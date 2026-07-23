import numpy as np
from scipy import ndimage

def rotate_frame(frame: np.ndarray, degrees: float, **kwargs) -> np.ndarray:
    """Rotate a 2d array of pixel data clockwise by an angle in degrees.
    """
    kwargs.setdefault('reshape', False)
    return ndimage.rotate(frame, -degrees, **kwargs)

def rotate_points(points: np.ndarray, degrees: float, origin=None) -> np.ndarray:
    """Rotate a 2d array of points (shape: n_points, 2) clockwise around an origin (x, y).
    """
    if len(points) == 0: return points

    origin = np.array([0, 0] if origin is None else origin)
    radians = np.radians(degrees)
    rot_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians),  np.cos(radians)],
    ])
    points_rot = (points - origin) @ rot_matrix.T + origin
    return points_rot