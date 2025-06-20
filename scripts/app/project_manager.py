from pathlib import Path

from domain.models.side import Side
from infra.io.project_io_manager import ProjectIOManager
from app.dataset import DepthDatasetLoader
from app.pipeline import convert_yuv_directory_to_png, convert_depth_directory_to_linear


class ProjectManager:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.io_manager = ProjectIOManager(project_dir)

        self.left_depth_dataset_loader = DepthDatasetLoader(
            project_io_manager=self.io_manager,
            side=Side.LEFT
        )
        self.right_depth_dataset_loader = DepthDatasetLoader(
            project_io_manager=self.io_manager,
            side=Side.RIGHT
        )

    
    def load_depth_dataset(self, side: Side):
        print(f"[Info] Loading depth dataset for {side} camera...")
        if side == Side.LEFT:
            return self.left_depth_dataset_loader.load_dataset()
        else:
            return self.right_depth_dataset_loader.load_dataset()


    def convert_yuv_to_rgb(
        self,
        apply_filter: bool = False,
        blur_threshold: float = 50.0,
        exposure_threshold_low: float = 0.1,
        exposure_threshold_high: float = 0.1
    ):
        for side in Side:
            print(f"[Info] Converting {side} camera images...")

            yuv_repo = self.io_manager.get_yuv_repo(side)
            rgb_repo = self.io_manager.get_rgb_repo(side)

            convert_yuv_directory_to_png(
                yuv_repo=yuv_repo,
                rgb_repo=rgb_repo,
                apply_filter=apply_filter,
                blur_threshold=blur_threshold,
                exposure_threshold_low=exposure_threshold_low,
                exposure_threshold_high=exposure_threshold_high
            )

    
    def convert_depth_to_linear_map(self, clip_near: float = 0.1, clip_far: float = 100.0):
        for side in Side:
            print(f"[Info] Converting {side} camera depth images to linear format...")

            dataset = self.load_depth_dataset(side=side)
            convert_depth_directory_to_linear(
                project_io_manager=self.io_manager,
                side=side,
                dataset=dataset,
                clip_near=clip_near,
                clip_far=clip_far
            )

