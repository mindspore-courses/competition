import mindspore as ms
from mindspore import ops


def get_box_corners(center, size, angle):
    """Returns the corner points of a given bounidng box.

    The returnd corners are given in the following order:

          7------6
         /|     /|
        4------5 |
        | 3----|-2
        |/     |/
        0------1

    As an alternative to the diagram, this is a unit bounding
    box which has the correct vertex ordering:

        box_corner_vertices = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]

    Arguments:
        center: Center point (x, y, z) of the bounding boxes
            with shape (B, N, 3).
        size: Size of the bounding boxes (l, w, h)
            with shape (B, N, 3).
        angle: Orientation of the bounding boxes around the
            hight (z) dimension with shape (B, N). The angle
            is expected to be given in radians.

    Returns:
        corners: Corner points of the bounding boxes
            with shape (B, N, 8, 3)
    """

    # Get batch size and number of boxes
    B = center.shape[0]
    N = center.shape[1]
    # Get initial 3D bounding box corners (B, N, 8, 3)
    x_corners = (size[..., 0] / 2)[..., None] \
        *ms.tensor([-1, 1, 1, -1, -1, 1, 1, -1])
    y_corners = (size[..., 1] / 2)[..., None] \
        *ms.tensor([-1, -1, 1, 1, -1, -1, 1, 1])
    z_corners = (size[..., 2] / 2)[..., None] \
        *ms.tensor([-1, -1, -1, -1, 1, 1, 1, 1])
    corners = ops.stack((x_corners, y_corners, z_corners), axis=-1)
    if len(corners.shape)<4:
        corners=ops.expand_dims(corners,0)

    # Create rotation matrices from rotation angle
    rotation_matrices = ops.tile(ops.eye(3, 3),(B, N, 1, 1))
    rotation_matrices[:, :, 0, 0] = ops.cos(angle)
    rotation_matrices[:, :, 0, 1] = -ops.sin(angle)
    rotation_matrices[:, :, 1, 0] = ops.sin(angle)
    rotation_matrices[:, :, 1, 1] = ops.cos(angle)

    # Rotate 3D bounding box corners
    # corners = ops.einsum('bnij,bnkj->bnki', rotation_matrices, corners)
    corners = ops.BatchMatMul()(rotation_matrices, corners.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)

    # Translate 3D bounding box corners
    corners[..., 0] = corners[..., 0] + center[..., None, 0]
    corners[..., 1] = corners[..., 1] + center[..., None, 1]
    corners[..., 2] = corners[..., 2] + center[..., None, 2]

    return corners


def get_minimum_enclosing_box_corners(boxes1: ms.Tensor, boxes2: ms.Tensor) -> ms.Tensor:
    """Returns the minimum enclosing convex box given two boxes.

    Arguments:
        boxes1: Bounding box corners with shape (B, N, 8, 3).
        boxes2: Bounding box corners with shape (B, M, 8, 3).

    Returns:
        enclosings: Bounding box corners of the minimum
            enclosing convex boxes with shape (B, N, M, 8, 3).
    """
    # Get input shapes: B, N, and M
    B = boxes1.shape[0]
    N = boxes1.shape[1]
    M = boxes2.shape[1]

    # Flatten input boxes (B, N, 8, 3) -> (B * N, 8, 3)
    boxes1 = boxes1.flatten(0, 1)
    boxes2 = boxes2.flatten(0, 1)

    # Split inputs into spatial dimensions
    x1 = boxes1[..., 0]
    x2 = boxes2[..., 0]
    y1 = boxes1[..., 1]
    y2 = boxes2[..., 1]
    z1 = boxes1[..., 2]
    z2 = boxes2[..., 2]

    # Create combinations of all box coordinates (B * N, B * M, 16)
    x = ops.dstack((x1.unsqueeze(1).repeat(1, M, 1), x2.unsqueeze(0).repeat(N, 1, 1)))
    y = ops.dstack((y1.unsqueeze(1).repeat(1, M, 1), y2.unsqueeze(0).repeat(N, 1, 1)))
    z = ops.dstack((z1.unsqueeze(1).repeat(1, M, 1), z2.unsqueeze(0).repeat(N, 1, 1)))

    # Find corner coordinates of the minimum enclosing convex boxes (B * N, B * M)
    x_min, _ = ops.min(x, axis=-1)
    y_min, _ = ops.min(y, axis=-1)
    z_min, _ = ops.min(z, axis=-1)
    x_max, _ = ops.max(x, axis=-1)
    y_max, _ = ops.max(y, axis=-1)
    z_max, _ = ops.max(z, axis=-1)

    # Construct corners of the minimum enclosing convex boxes (B * N, B * M, 3)
    c1 = ops.dstack((x_min, y_min, z_min))
    c2 = ops.dstack((x_max, y_min, z_min))
    c3 = ops.dstack((x_max, y_max, z_min))
    c4 = ops.dstack((x_min, y_max, z_min))
    c5 = ops.dstack((x_min, y_min, z_max))
    c6 = ops.dstack((x_max, y_min, z_max))
    c7 = ops.dstack((x_max, y_max, z_max))
    c8 = ops.dstack((x_min, y_max, z_max))

    # Stack corners (B * N, B * M, 8, 3)
    enclosings = ops.stack((c1, c2, c3, c4, c5, c6, c7, c8), axis=2)

    # Reshape box corners
    enclosings = enclosings.reshape(B, N, M, 8, 3)

    return enclosings


def get_box_volume_from_corners(boxes: ms.Tensor) -> ms.Tensor:
    """Returns the volumes of the given boxes.

    The box corners mut be given in the following order:

          7------6
         /|     /|
        4------5 |
        | 3----|-2
        |/     |/
        0------1

    Arguments:
        boxes: Bounding box corners with shape (N, 8, 3).

    Returns:
        volumes: Bounding box volumes with shape (N,).
    """
    # Get side length of the boxes
    length = ms.mint.linalg.norm(boxes[..., 1, :] - boxes[..., 0, :], dim=-1)
    width = ms.mint.linalg.norm(boxes[..., 3, :] - boxes[..., 0, :], dim=-1)
    height = ms.mint.linalg.norm(boxes[..., 4, :] - boxes[..., 0, :], dim=-1)

    # Calculate box volumes
    volumes = length * width * height

    return volumes