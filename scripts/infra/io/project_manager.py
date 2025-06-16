from pathlib import Path
from enum import Enum
from typing import Optional
import numpy as np

import config.path_config as path_config
from domain.models.camera_dataset import DepthDataset
from infra.io.yuv_repository import YUVRepository
from infra.io.rgb_repository import RGBRepository
from infra.io.depth_repository import DepthRepository
from infra.io.depth_dataset_loader import DepthDatasetLoader


class Side(Enum):
    LEFT = "left"
    RIGHT = "right"


class ProjectManager:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir

        self.left_yuv_repo = YUVRepository(
            yuv_dir=project_dir / path_config.LEFT_CAMERA_YUV_IMAGE_DIR,
            format_json=project_dir / path_config.LEFT_CAMERA_IMAGE_FORMAT_JSON
        )
        self.right_yuv_repo = YUVRepository(
            yuv_dir=project_dir / path_config.RIGHT_CAMERA_YUV_IMAGE_DIR,
            format_json=project_dir / path_config.RIGHT_CAMERA_IMAGE_FORMAT_JSON
        )

        self.left_rgb_repo = RGBRepository(project_dir / path_config.LEFT_CAMERA_RGB_IMAGE_DIR)
        self.right_rgb_repo = RGBRepository(project_dir / path_config.RIGHT_CAMERA_RGB_IMAGE_DIR)

        self.left_depth_grey_repo = RGBRepository(project_dir / path_config.LEFT_DEPTH_GRAY_IMAGE_DIR)
        self.right_depth_grey_repo = RGBRepository(project_dir / path_config.RIGHT_DEPTH_GRAY_IMAGE_DIR)

        self.left_depth_repo = DepthRepository(
            project_root=project_dir,
            depth_dir=project_dir / path_config.LEFT_DEPTH_RAW_DATA_DIR
        )
        self.right_depth_repo = DepthRepository(
            project_root=project_dir,
            depth_dir=project_dir / path_config.RIGHT_DEPTH_RAW_DATA_DIR
        )
        self.left_depth_data_loader = DepthDatasetLoader(
            depth_dataset_cache_path=project_dir / path_config.LEFT_DEPTH_DATASET_CACHE,
            descriptor_csv_path=project_dir / path_config.LEFT_DEPTH_DESCRIPTOR_CSV,
            depth_repo=self.left_depth_repo,
        )
        self.right_depth_data_loader = DepthDatasetLoader(
            depth_dataset_cache_path=project_dir / path_config.RIGHT_DEPTH_DATASET_CACHE,
            descriptor_csv_path=project_dir / path_config.RIGHT_DEPTH_DESCRIPTOR_CSV,
            depth_repo=self.right_depth_repo,
        )


    def get_yuv_repo(self, side: Side) -> YUVRepository:
        if side == Side.LEFT:
            return self.left_yuv_repo
        else:
            return self.right_yuv_repo


    def get_rgb_repo(self, side: Side) -> RGBRepository:
        if side == Side.LEFT:
            return self.left_rgb_repo
        else:
            return self.right_rgb_repo
        

    def get_depth_grey_repo(self, side: Side) -> RGBRepository:
        if side == Side.LEFT:
            return self.left_depth_grey_repo
        else:
            return self.right_depth_grey_repo


    def get_depth_dataset(self, side: Side) -> DepthDataset:
        if side == Side.LEFT:
            return self.left_depth_data_loader.load_dataset()
        else:
            return self.right_depth_data_loader.load_dataset()
        

    def load_depth_map_by_index(
        self,
        side: Side,
        index: int,
        dataset: Optional[DepthDataset] = None,
    ):
        if dataset is None:
            dataset = self.get_depth_dataset(side=side)

        return self.load_depth_map(
            side=side,
            timestamp=dataset.timestamps[index],
            width=dataset.widths[index],
            height=dataset.heights[index],
            near=dataset.nears[index],
            far=dataset.fars[index]
        )

    
    def load_depth_map(
        self, 
        side: Side,
        timestamp: int,
        width: int,
        height: int,
        near: float,
        far: float,
    ) -> np.ndarray:
        if side == Side.LEFT:
            return self.left_depth_repo.load(
                timestamp=timestamp,
                width=width, height=height,
                near=near, far=far
            )
        else:
            return self.right_depth_repo.load(
                timestamp=timestamp,
                width=width, height=height,
                near=near, far=far
            )            