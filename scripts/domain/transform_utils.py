from enum import Enum
from dataclasses import dataclass
import numpy as np


class CoordinateSystem(Enum):
    UNITY = "Unity"
    OPEN3D = "Open3D"
    OPENGL = "OpenGL"


@dataclass
class Transforms:
    coordinate_system: CoordinateSystem

    positions: np.ndarray # shape=(N, 3), axis1=(x, y, z)
    rotations: np.ndarray # shape=(N, 4), axis1=(x, y, z, w)