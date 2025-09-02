from typing import Tuple

import mindspore as ms

from mindspore import nn,ops

# ms.set_context(enable_graph_kernel=False)

def cart2polar(x: ms.Tensor,
               y: ms.Tensor,
               degrees: bool = True) -> Tuple[ms.Tensor, ms.Tensor]:
    """Returns the polar coordinates for the given Cartesian coordinates.

    Convention of the polar angle phi in respect to the Cartesian axis.

                90
          135    y    45
                 |
        180      ---x    0

          -135       -45
                -90

    Arguments:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

    Returns:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the range. The angle is zero on the x-axis
    """

    # Convert coordiantes
    r = ms.mint.linalg.norm(ms.ops.dstack((x,y)), axis=0)
    phi =safe_atan2(y,x)

    if degrees:
        phi = ms.ops.rad2deg(phi)

    return r, phi

class Cart2Polar(nn.Cell):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def construct(self, batch: ms.Tensor):
        """Returns the polar coordinates for the given Cartesian coordinates.

        Arguments:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 2)

        Returns:
            batch: Batch of points in polar
                coordinates with shape
                (batch, points, 2)
        """
        return ms.ops.cat(cart2polar(*batch.split(1, self.dim), self.degrees), self.dim)
  
def cart2spher(x: ms.Tensor,
               y: ms.Tensor,
               z: ms.Tensor,
               degrees: bool = True) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    """Returns the spherical coordinates for the given Cartesian coordinates.

    Convention of the spherical angle phi (left) and roh (right)
    in respect to the Cartesian axis.

                90                          90
          135    y    45              45     z    45
                 |                           |
        180      ---x    0          0        ---y    0

          -135       -45              -45        -45
                -90                         -90

    Arguments:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
        z: Values of the z-dimension in Cartesian coordinates.
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

    Returns:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the y-z-plane. The angle is zero on the x-axis
            increases mathematical positivly.
        roh: Elevation (inclination) angle. Angle between the
            z-axis and the x-y-plane. The angle is zero on the
            x-y-plane, positive towards the positive z-axis and
            negative towards the negative z-axis.
    """
    # Convert coordinates
    r = ms.mint.linalg.norm(ops.dstack((x, y, z)), dim=-1).reshape_as(x)
    phi = ops.atan2(y, x)

    # Avoid devision by zero
    c = ops.zeros_like(z)
    mask = (r != 0)
    c[mask] = z[mask] / r[mask]

    roh = ops.arcsin(c)

    if degrees:
        phi = ops.rad2deg(phi)
        roh = ops.rad2deg(roh)

    return r, phi, roh

class Cart2Spher(nn.Cell):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def construct(self, batch: ms.Tensor):
        """Returns the spherical coordinates for the given Cartesian coordinates.

        Arguments:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 3)

        Returns:
            batch: Batch of points in spherical
                coordinates with shape
                (batch, points, 3)
        """
        return ms.ops.cat(cart2spher(*batch.split(1, self.dim), self.degrees), self.dim)

def polar2cart(r: ms.Tensor,
               phi: ms.Tensor,
               degrees: bool = True) -> Tuple[ms.Tensor, ms.Tensor]:
    """Returns the Cartesian coordinates for the given polar coordinates.

    Convention of the polar angle phi in respect to the Cartesian axis.

                90
          135    y    45
                 |
        180      ---x    0

          -135       -45
                -90

    Arguments:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the range. The angle is zero on the x-axis
            increases mathematical positivly.
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

    Returns:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
    """
    if degrees:
        x = r * ms.ops.cos(ms.ops.deg2rad(phi))
        y = r * ms.ops.sin(ms.ops.deg2rad(phi))
    else:
        x = r * ms.ops.cos(phi)
        y = r * ms.ops.sin(phi)

    return x, y

class Polar2Cart(nn.Cell):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def construct(self, batch: ms.Tensor):
        """Returns the Cartesian coordinates for the given polar coordinates.

        Arguments:
            batch: Batch of points in polar
                coordinates with shape
                (batch, points, 2)

        Returns:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 2)
        """
        return ops.cat(polar2cart(*batch.split(1, self.dim), self.degrees), self.dim)

def spher2cart(r: ms.Tensor,
               phi: ms.Tensor,
               roh: ms.Tensor,
               degrees: bool = True) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
    """Returns the Cartesian coordinates for the given spherical coordinates.

    Convention of the spherical angle phi (left) and roh (right)
    in respect to the Cartesian axis.

                90                          90
          135    y    45              45     z    45
                 |                           |
        180      ---x    0          0        ---y    0

          -135       -45              -45        -45
                -90                         -90

    Arguments:
        r: Range (radius) to the origin.
        phi: Azimuth angle. Angle between the x-axis and
            the y-z-plane. The angle is zero on the x-axis
            increases mathematical positivly.
        roh: Elevation (inclination) angle. Angle between the
            z-axis and the x-y-plane. The angle is zero on the
            x-y-plane, positive towards the positive z-axis and
            negative towards the negative z-axis.
        degrees: Whether the angular values are given in
            degrees (True) or radians (False).

    Returns:
        x: Values of the x-dimension in Cartesian coordinates.
        y: Values of the y-dimension in Cartesian coordinates.
        z: Values of the z-dimension in Cartesian coordinates.
    """
    if degrees:
        x = r * ms.ops.cos(ms.ops.deg2rad(phi)) * ms.ops.cos(ms.ops.deg2rad(roh))
        y = r * ms.ops.sin(ms.ops.deg2rad(phi)) * ms.ops.cos(ms.ops.deg2rad(roh))
        z = r * ms.ops.sin(ms.ops.deg2rad(roh))
    else:
        x = r * ms.ops.cos(phi) * ms.ops.cos(roh)
        y = r * ms.ops.sin(phi) * ms.ops.cos(roh)
        z = r * ms.ops.sin(roh)

    return x, y, z

class Spher2Cart(nn.Cell):
    def __init__(self,
                 dim: int = -1,
                 degrees: bool = True,
                 **kwargs):
        super().__init__()

        self.dim = dim
        self.degrees = degrees

    def construct(self, batch: ms.Tensor):
        """Returns the Cartesian coordinates for the given spherical coordinates.

        Arguments:
            batch: Batch of points in spherical
                coordinates with shape
                (batch, points, 3)

        Returns:
            batch: Batch of points in cartesian
                coordinates with shape
                (batch, points, 3)
        """
        return ops.cat(spher2cart(*batch.split(1, self.dim), self.degrees), self.dim)


def build_transformation(name: str, *args, **kwargs):
    if name is None:
        return None
    if 'polar2cart' in name.lower():
        return Polar2Cart(*args, **kwargs)
    if 'spher2cart' in name.lower():
        return Spher2Cart(*args, **kwargs)
    if 'cart2polar' in name.lower():
        return Cart2Polar(*args, **kwargs)
    if 'cart2spher' in name.lower():
        return Cart2Spher(*args, **kwargs)

# def safe_atan2(y, x):
#     """手动实现 atan2，避免依赖未注册的内核"""
#     eps = 1e-6
#     angle = ops.atan(y / (x + eps))
#     angle = ops.where(x < 0, angle + ms.numpy.pi * ops.sign(y), angle)
#     angle = ops.where((x == 0) & (y > 0), ms.numpy.pi / 2, angle)
#     angle = ops.where((x == 0) & (y < 0), -ms.numpy.pi / 2, angle)
#     return angle