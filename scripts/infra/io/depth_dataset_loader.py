from pathlib import Path
import numpy as np
import pandas as pd

from domain.depth_utils import compute_depth_camera_params
from domain.transform_utils import Transforms, CoordinateSystem
from domain.camera_dataset import DepthDataset
from infra.io.depth_repository import DepthRepository


class DepthDatasetLoader:
    def __init__(
        self,
        depth_dataset_cache_path: Path,
        descriptor_csv_path: Path,
        depth_repo: DepthRepository
    ):
        self.depth_dataset_cache_path = depth_dataset_cache_path
        self.descriptor_csv_path = descriptor_csv_path
        self.depth_repo = depth_repo


    def load_dataset(self) -> DepthDataset:
        if self.depth_dataset_cache_path.exists():
            print(f"[Info] Depth dataset cache ({self.depth_dataset_cache_path}) detected. Loading cached dataset...")

            try:
                return DepthDataset.load(self.depth_dataset_cache_path)
            except Exception as e:
                print(f"[Error] Depth dataset cache ({self.depth_dataset_cache_path}) is corrupted or invalid. Rebuilding cache from the original source...\n{e}")

        else:
            print(f"[Info] Depth dataset not found. Rebuilding cache from the original source...")
        
        df = pd.read_csv(self.descriptor_csv_path)

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

            depth_path = self.depth_repo.get_relaive_path(timestamp=timestamp)

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

        dataset = DepthDataset(
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

        dataset.save(self.depth_dataset_cache_path)

        return dataset