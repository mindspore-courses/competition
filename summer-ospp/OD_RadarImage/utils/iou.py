import mindspore as ms
from mindspore import ops

from utils import bbox


def _check_coplanar(boxes: ms.Tensor, eps: float = 1e-4) -> None:
    # Define box planes
    _box_planes = [
        [0, 1, 2, 3],
        [3, 2, 6, 7],
        [0, 1, 5, 4],
        [0, 3, 7, 4],
        [1, 2, 6, 5],
        [4, 5, 6, 7],
    ]

    faces = ms.tensor(_box_planes, dtype=ms.int32)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    P, V = faces.shape
    # (B, P, 4, 3) -> (B, P, 3)
    v0, v1, v2, v3 = verts.reshape(B, P, V, 3).unbind(2)

    # Compute the normal
    e0 = ops.L2Normalize(axis=-1)(v1-v0)
    e1 = ops.L2Normalize(axis=-1)(v2-v0)
    normal = ops.L2Normalize(axis=-1)(ops.cross(e0, e1, dim=-1))

    # Check the fourth vertex is also on the same plane
    mat1 = (v3 - v0).view(B, 1, -1)  # (B, 1, P*3)
    mat2 = normal.view(B, -1, 1)  # (B, P*3, 1)

    return (mat1.bmm(mat2).abs() < eps).squeeze()


def _check_nonzero(boxes: ms.Tensor, eps: float = 1e-4) -> None:
    """
    Checks that the sides of the box have a non zero area
    """
    # Define box triangles
    _box_triangles = [
        [0, 1, 2],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [1, 5, 6],
        [1, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [3, 2, 6],
        [3, 6, 7],
        [0, 1, 5],
        [0, 4, 5],
    ]

    faces = ms.tensor(_box_triangles, dtype=ms.int32)
    verts = boxes.index_select(index=faces.view(-1), dim=1)
    B = boxes.shape[0]
    T, V = faces.shape
    # (B, T, 3, 3) -> (B, T, 3)
    v0, v1, v2 = verts.reshape(B, T, V, 3).unbind(2)

    normals = ops.cross(v1 - v0, v2 - v0, dim=-1)  # (B, T, 3)
    face_areas = normals.norm(dim=-1) / 2

    return ops.all((face_areas > eps), axis=1)


def iou3d(boxes1: ms.Tensor, boxes2: ms.Tensor) -> ms.Tensor:
    """Returns the intersection over union between the given boxes.

    Backward is not supported.

    Arguments:
        boxes1: Bounding box corners with shape (B, N, 8, 3).
        boxes2: Bounding box corners with shape (B, M, 8, 3).

    Returns:
        iou: Intersection over union with shape (B, N, M)
    """
    # Get input shapes: B, N, and M
    B = boxes1.shape[0]
    N = boxes1.shape[1]
    M = boxes2.shape[1]

    # Flatten inputs (B, N, 8, 3) -> (B * N, 8, 3)
    boxes1 = boxes1.flatten(0, 1)
    boxes2 = boxes2.flatten(0, 1)

    # Initialize iou
    iou_3d = ops.zeros((B*N,B*M), dtype=boxes1.dtype)

    # Check if inputs are empty
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return iou_3d.reshape((B, N, M))

    # Get mask for invalid boxes
    mask_1 = ops.logical_and(_check_nonzero(boxes1), _check_coplanar(boxes1))
    mask_2 = ops.logical_and(_check_nonzero(boxes2), _check_coplanar(boxes2))
    mask = ops.logical_and(*ops.meshgrid(mask_1, mask_2, indexing='ij'))
    # Check if inputs contain valid entries
    if not mask.any():
        return iou_3d.reshape((B, N, M))
    
    # Get intersection over union
    _, iou_3d_valid = box3d_overlap(boxes1[mask_1], boxes2[mask_2])
    
    # Insert valid iou values
    iou_3d[mask] = iou_3d_valid.flatten()
        
    # Reconstruct input shape
    iou_3d = iou_3d.reshape((B, N, M))

    return iou_3d


def giou3d(boxes1: ms.Tensor, boxes2: ms.Tensor) -> ms.Tensor:
    """Returns the generalized intersection over union between boxes.

    The Generalized Intersection over Union is given as
    GIoU = |A n B| / |A u B| - |C / (A u B)| / |C| =
           IoU - |C / (A u B)| / |C|

    with boxes A and B as well as thier minumin enclosing box C.

    Reference: https://giou.stanford.edu/

    Note: Backward is not supported.

    Arguments:
        boxes1: Bounding box corners with shape (B, N, 8, 3).
        boxes2: Bounding box corners with shape (B, M, 8, 3).

    Returns:
        giou: Generalized intersection over union with shape (B, N, M)
    """
    # Get input shapes: B, N, and M
    B = boxes1.shape[0]
    N = boxes1.shape[1]
    M = boxes2.shape[1]

    # Get minimal enclosing boxes (B, N, M, 8, 3)
    C = bbox.get_minimum_enclosing_box_corners(boxes1, boxes2)

    # Flatten inputs (B, N, 8, 3) -> (B * N, 8, 3)
    boxes1 = boxes1.flatten(0, 1)
    boxes2 = boxes2.flatten(0, 1)

    # Flatten minimal enclosing boxes (B, N, M, 8, 3) -> (B * N * M, 8, 3)
    C = C.flatten(0, 2)

    # Initialize iou, volume and union
    iou_3d = ops.zeros((B * N, B * M), dtype=boxes1.dtype)
    vol_3d = ops.zeros((B * N, B * M), dtype=boxes1.dtype)
    uni_3d = ops.zeros((B * N, B * M), dtype=boxes1.dtype)

    # Initialize enclosing volume
    evol_3d = -ops.ones((B * N, B * M), dtype=boxes1.dtype)

    # Check if inputs are empty
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return evol_3d.reshape((B, N, M))

    # Get mask for invalid boxes
    mask_1 = ops.logical_and(_check_nonzero(boxes1), _check_coplanar(boxes1))
    mask_2 = ops.logical_and(_check_nonzero(boxes2), _check_coplanar(boxes2))
    mask = ops.logical_and(*ops.meshgrid(mask_1, mask_2, indexing='ij'))

    # Check if inputs contain valid entries
    if not mask.any():
        return evol_3d.reshape((B, N, M))

    # Get intersection over union
    vol_3d_valid, iou_3d_valid = box3d_overlap(boxes1[mask_1], boxes2[mask_2])

    # Insert valid values
    iou_3d[mask] = iou_3d_valid.flatten()
    vol_3d[mask] = vol_3d_valid.flatten()

    # Calculate union
    zero_mask = (iou_3d != 0)
    uni_3d[zero_mask] = vol_3d[zero_mask] / iou_3d[zero_mask]

    # Reconstruct input shape
    iou_3d = iou_3d.reshape((B, N, M))
    vol_3d = vol_3d.reshape((B, N, M))
    uni_3d = uni_3d.reshape((B, N, M))

    # Calculate enclosing volume
    evol_3d_valid = bbox.get_box_volume_from_corners(C[mask.flatten()])

    # Insert valid values
    evol_3d[mask] = evol_3d_valid

    # Reconstruct input shape
    evol_3d = evol_3d.reshape((B, N, M))

    # Initialize giou
    giou = ops.zeros((B, N, M), dtype=boxes1.dtype)

    # Calculate giou
    zero_mask = (evol_3d != 0)
    giou[zero_mask] = \
        iou_3d[zero_mask] - (evol_3d[zero_mask] - uni_3d[zero_mask]) / evol_3d[zero_mask]

    return giou


def box3d_overlap(boxes1: ms.Tensor, boxes2: ms.Tensor) -> tuple:
    """
    Convert Cartesian coordinates to spherical coordinates (elevation version)

    Conventions:
    - r: Radius (distance to the origin)
    - phi: Azimuthal angle, the angle between the x-axis and the y-z plane, range [-180°, 180°] or [-π, π]
    - roh: Elevation angle, the angle between the x-y plane and the z-axis, range [-90°, 90°] or [-π/2, π/2]

    Parameters:
        x: Array of x-coordinates
        y: Array of y-coordinates
        z: Array of z-coordinates
        degrees: Whether to return angles in degrees (True) or radians (False)

    Returns:
        r: Radius
        phi: Azimuthal angle
        roh: Elevation angle
    """
    boxes1 = ms.Tensor(boxes1)
    boxes2 = ms.Tensor(boxes2)

    min_box1 = boxes1.min(dim=-2)[0]  # (B, N, 3) 
    max_box1 = boxes1.max(dim=-2)[0]
    min_box2 = boxes2.min(dim=-2)[0]  # (B, M, 3) 
    max_box2 = boxes2.max(dim=-2)[0]


    intersect_min = ops.maximum(min_box1.unsqueeze(-2), min_box2.unsqueeze(-3))  # (B, N, M, 3)
    intersect_max = ops.minimum(max_box1.unsqueeze(-2), max_box2.unsqueeze(-3))  # (B, N, M, 3)


    intersect_lengths = ops.clamp(intersect_max - intersect_min, min=0)  # (B, N, M, 3)
    intersection_volume = intersect_lengths.prod(dim=-1)  # (B, N, M)


    volume_box1 = (max_box1 - min_box1).prod(dim=-1)  # (B, N)
    volume_box2 = (max_box2 - min_box2).prod(dim=-1)  # (B, M)
    union_volume = volume_box1.unsqueeze(-1) + volume_box2.unsqueeze(-2) - intersection_volume  # (B, N, M)

    iou_3d = intersection_volume / (union_volume + 1e-6)

    return intersection_volume, iou_3d