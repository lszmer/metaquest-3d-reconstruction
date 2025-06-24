from typing import Optional
import numpy as np

from domain.utils.depth_utils import compute_depth_camera_params
from domain.models.side import Side
from domain.models.transforms import Transforms, CoordinateSystem
from domain.models.camera_dataset import DepthDataset
from infra.io.project_io_manager import ProjectIOManager


class DepthDatasetLoader:
    def __init__(
        self,
        project_io_manager: ProjectIOManager,
        side: Side
    ):
        self.project_io_manager = project_io_manager
        self.side = side
        self.dataset: Optional[DepthDataset] = None


    def load_dataset(self) -> DepthDataset:
        if self.dataset is not None:
            print(f"[Info] Depth dataset already loaded. Returning cached dataset...")
            return self.dataset

        if self.project_io_manager.depth_dataset_cache_exists(side=self.side):
            print(f"[Info] Depth dataset cache found. Loading cached dataset...")

            try:
                return self.project_io_manager.load_depth_dataset_cache(side=self.side)
            except Exception as e:
                print(f"[Error] Depth dataset cache is corrupted or invalid. Rebuilding cache from the original source...\n{e}")

        else:
            print(f"[Info] Depth dataset not found. Rebuilding cache from the original source...")

        df = self.project_io_manager.load_depth_descriptor(side=self.side)
        depth_repo = self.project_io_manager.get_depth_repo(side=self.side)

        depth_paths = []
        timestamps = []
        fxs = []
        fys = []
        cxs = []
        cys = []
        positions = []
        rotations = []
        widths = []
        heights = []
        nears = []
        fars = []

        for _, row in df.iterrows():

            timestamp = int(row['timestamp_ms'])
            width = int(row['width'])
            height = int(row['height'])

            near = float(row['near_z'])
            far = float(row['far_z'])

            left = float(row['fov_left_angle_tangent'])
            right = float(row['fov_right_angle_tangent'])
            top = float(row['fov_top_angle_tangent'])
            bottom = float(row['fov_down_angle_tangent'])

            position = np.array([
                row['create_pose_location_x'],
                row['create_pose_location_y'],
                row['create_pose_location_z'],
            ])

            rotation = np.array([
                row['create_pose_rotation_x'],
                row['create_pose_rotation_y'],
                row['create_pose_rotation_z'],
                row['create_pose_rotation_w'],
            ])

            fx, fy, cx, cy = compute_depth_camera_params(
                left, right, top, bottom, width, height
            )

            depth_path = depth_repo.get_relaive_path(timestamp=timestamp)

            depth_paths.append(str(depth_path))
            timestamps.append(timestamp)
            fxs.append(fx)
            fys.append(fy)
            cxs.append(cx)
            cys.append(cy)
            positions.append(position)
            rotations.append(rotation)
            widths.append(width)
            heights.append(height)
            nears.append(near)
            fars.append(far)

        self.dataset = DepthDataset(
            image_relative_paths=np.array(depth_paths),
            timestamps=np.array(timestamps),
            fx=np.array(fxs),
            fy=np.array(fys),
            cx=np.array(cxs),
            cy=np.array(cys),
            transforms=Transforms(
                coordinate_system=CoordinateSystem.UNITY,
                positions=np.array(positions),
                rotations=np.array(rotations)
            ),
            widths=np.array(widths),
            heights=np.array(heights),
            nears=np.array(nears),
            fars=np.array(fars)
        )

        self.project_io_manager.save_depth_dataset_cache(
            side=self.side,
            dataset=self.dataset
        )

        return self.dataset