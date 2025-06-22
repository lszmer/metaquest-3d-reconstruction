from typing import Optional

import numpy as np
from tqdm import tqdm

from domain.models.camera_dataset import RGBDDataset
from domain.models.side import Side
from app.helper.depth_reprojector import DepthReprojector
from infra.io.project_io_manager import ProjectIOManager
from app.dataset.depth_dataset_loader import DepthDatasetLoader
from app.dataset.color_dataset_loader import ColorDatasetLoader


class RGBDDatasetLoader:
    def __init__(
        self,
        project_io_manager: ProjectIOManager,
        side: Side,
        depth_dataset_loader: DepthDatasetLoader,
        color_dataset_loader: ColorDatasetLoader,
    ):
        self.project_io_manager = project_io_manager
        self.side = side
        self.depth_dataset_loader = depth_dataset_loader
        self.color_dataset_loader = color_dataset_loader

        self.dataset: Optional[RGBDDataset] = None

    
    def load_dataset(self):
        if self.dataset is not None:
            print(f"[Info] RGBD dataset already loaded. Returning cached dataset...")
            return self.dataset

        if self.project_io_manager.color_dataset_cache_exists(side=self.side):
            print(f"[Info] RGBD dataset cache found. Loading cached dataset...")

            try:
                return self.project_io_manager.load_color_dataset_cache(side=self.side)
            except Exception as e:
                print(f"[Error] RGBD dataset cache is corrupted or invalid. Rebuilding cache from the original source...\n{e}")

        else:
            print(f"[Info] RGBD dataset not found. Rebuilding cache from the original source...")

        color_dataset = self.color_dataset_loader.load_dataset()
        depth_dataset = self.depth_dataset_loader.load_dataset()

        color_intrinsics = color_dataset.get_intrinsic_matrices()
        depth_intrinsics = depth_dataset.get_intrinsic_matrices()

        color_extrinsics_wc = color_dataset.transforms.extrinsics_wc
        depth_extrinsics_wc = depth_dataset.transforms.extrinsics_wc

        N = len(color_dataset.timestamps)

        source_color_indices = []
        source_depth_timestamps = []
        source_depth_maps = []
        source_depth_intrinsics = []
        source_depth_extrinsics = []
        source_color_intrinsics = []
        source_color_extrinsics = []
        source_color_map_sizes = []

        for i in tqdm(range(N), desc="Aligning RGB-D frames"):
            color_timestamp = color_dataset.timestamps[i]
            color_intrinsic = color_intrinsics[i]
            color_extrinsic_wc = color_extrinsics_wc[i]
            color_map_size = (color_dataset.heights[0], color_dataset.widths[0])

            nearest_depth_index = depth_dataset.find_nearest_index(timestamp=color_timestamp)

            depth_timestamp = depth_dataset.timestamps[nearest_depth_index]
            if abs(color_timestamp - depth_timestamp) > 100: # milliseconds
                continue

            depth_map = self.project_io_manager.load_depth_map_by_index(
                side=self.side,
                index=nearest_depth_index,
                dataset=depth_dataset
            )

            depth_intrinsic = depth_intrinsics[nearest_depth_index]
            depth_extrinsic_wc = depth_extrinsics_wc[nearest_depth_index]

            source_depth_timestamps.append(depth_timestamp)
            source_color_indices.append(i)
            source_depth_maps.append(depth_map)
            source_depth_intrinsics.append(depth_intrinsic)
            source_depth_extrinsics.append(depth_extrinsic_wc)
            source_color_intrinsics.append(color_intrinsic)
            source_color_extrinsics.append(color_extrinsic_wc)
            source_color_map_sizes.append(color_map_size)

        reproject_batch = DepthReprojector.reproject_depth(
            depth_maps=np.array(source_depth_maps),
            depth_intrinsics=np.array(source_depth_intrinsics),
            depth_extrinsics=np.array(source_depth_extrinsics),
            rgb_intrinsics=np.array(source_color_intrinsics),
            rgb_extrinsics=np.array(source_color_extrinsics),
            out_sizes=np.array(source_color_map_sizes),
            batch_size=1
        )

        for indices, results in reproject_batch:
            N = len(results)
            
            if indices[0] >= 15:
                exit()
            if indices[0] % 6 != 5:
                continue

            for i in range(N):
                index = indices[i]
                v = results[i]

                color_index = source_color_indices[index]
                timestamp = color_dataset.timestamps[color_index]

                import matplotlib.pyplot as plt
                print(f"timestamp={timestamp}")
                print(f"depth timestamp={source_depth_timestamps[index]}")
                print(f"delta time={timestamp - source_depth_timestamps[index]}")
                print(f"vmin={np.min(v)}, vmax={np.max(v)}")
                print(f"Depth: \nintrinsic={source_depth_intrinsics[index]} \nextrinsic={source_depth_extrinsics[index]}\nextrinsic_inv={np.linalg.inv(source_depth_extrinsics[index])}")
                print(f"Color: \nintrinsic={source_color_intrinsics[index]} \nextrinsic={source_color_extrinsics[index]}\nextrinsic_inv={np.linalg.inv(source_color_extrinsics[index])}")
                delta_transform = source_depth_extrinsics[index] @ np.linalg.inv(source_color_extrinsics[index])
                from scipy.spatial.transform import Rotation as R
                delta_rot = R.from_matrix(delta_transform[:3, :3]).as_euler('zxy')
                print(f"Delta transform: {delta_transform}\n{delta_rot}")

                img = self.project_io_manager.load_color_map_by_index(side=self.side, index=color_index, dataset=color_dataset)

                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(img)
                axes[0].set_title("Original Image")
                axes[0].axis('off')

                im = axes[1].imshow(v, cmap='viridis', vmax=3, vmin=0)
                axes[1].set_title("Depth Map")
                axes[1].axis('off')

                #fig.colorbar(im)
                plt.tight_layout()
                plt.show()

        self.dataset = RGBDDataset(
            image_relative_paths=color_dataset.image_relative_paths,
            timestamps=color_dataset.timestamps,
            fx=color_dataset.fx,
            fy=color_dataset.fy,
            cx=color_dataset.cx,
            cy=color_dataset.cy,
            transforms=color_dataset.transforms,
            widths=color_dataset.widths,
            heights=color_dataset.heights,
            depth_relative_paths=depth_dataset.image_relative_paths
        )

        self.project_io_manager.save_rgbd_dataset_cache(side=self.side, dataset=self.dataset)
        print(f"[Info] RGBD dataset loaded successfully. Dataset size: {len(self.dataset.image_relative_paths)}")

        return self.dataset
