from pathlib import Path
import numpy as np

from domain.depth_utils import convert_depth_to_linear


class DepthRepository:
    def __init__(self, project_root:Path, depth_dir: Path):
        self.project_root = project_root
        self.depth_dir = depth_dir


    def get_relaive_path(self, timestamp: int) -> str:
        depth_map_path = self.depth_dir / f"{timestamp}.raw"
        return depth_map_path.relative_to(self.project_root)


    def load(
        self,
        timestamp: int,
        width: int,
        height: int,
        near: float,
        far: float
    ) -> np.ndarray:
        depth_map_path = self.depth_dir / f"{timestamp}.raw"

        depth_array = np.fromfile(depth_map_path, dtype='<f4').reshape((height, width))
        depth_map = convert_depth_to_linear(depth_array, near, far)

        return depth_map
