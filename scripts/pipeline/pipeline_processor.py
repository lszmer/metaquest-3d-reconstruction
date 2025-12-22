from pathlib import Path
from config.pipeline_configs import PipelineConfigs
from dataio.data_io import DataIO
from models.side import Side
from processing.depth_conversion.convert_depth_to_linear import convert_depth_directory
from processing.reconstruction.reconstruct_scene import reconstruct_scene
from processing.yuv_conversion.convert_yuv_dir import convert_yuv_directory


class PipelineProcessor:
    def __init__(self, project_dir: Path, config_yml_path: Path, force_recompute: bool = False):
        self.data_io = DataIO(project_dir=project_dir)
        self.pipeline_configs = PipelineConfigs.parse_config_yml(config_yml_path)
        self.project_dir = project_dir
        # When True, do not trust any cached datasets or previously generated outputs.
        # Always recompute per run (used for batch processing).
        self.force_recompute = force_recompute


    def convert_yuv_to_rgb(self):
        # YUV→RGB conversion should *always* reuse existing RGB where possible.
        # Even in batch / force_recompute mode we only want to fill in missing
        # RGB frames, not regenerate everything from scratch.
        # Check if RGB images already exist for all YUV timestamps
        all_rgb_exist = True
        missing_count = 0
        total_yuv_count = 0

        for side in Side:
            # Get all YUV timestamps
            yuv_timestamps = set(self.data_io.color.get_yuv_timestamps(side))
            total_yuv_count += len(yuv_timestamps)

            # Check if RGB directory exists
            rgb_dir = self.data_io.color.image_path_config.get_rgb_dir(side)
            if not rgb_dir.exists():
                all_rgb_exist = False
                missing_count += len(yuv_timestamps)
                continue

            # Get all existing RGB timestamps
            rgb_timestamps = set(self.data_io.color.get_rgb_timestamps(side))

            # Check if all YUV timestamps have corresponding RGB images
            missing = yuv_timestamps - rgb_timestamps
            if missing:
                all_rgb_exist = False
                missing_count += len(missing)

        if all_rgb_exist and total_yuv_count > 0:
            print(f"[Info] All RGB images already exist. Skipping YUV to RGB conversion.")
            for side in Side:
                rgb_count = len(self.data_io.color.get_rgb_timestamps(side))
                print(f"[Info] {side.name}: {rgb_count} RGB images found")
            return

        if missing_count > 0:
            print(f"[Info] Found {missing_count} missing RGB images. Converting YUV to RGB...")
        elif total_yuv_count == 0:
            print(f"[Info] No YUV images found. Skipping conversion.")
            return

        convert_yuv_directory(image_io=self.data_io.color, config=self.pipeline_configs.yuv_to_rgb)

    
    def convert_depth_to_linear(self):
        # In "force_recompute" mode, always run the conversion for all frames
        # and do not skip based on existing linear-depth PNGs.
        if not self.force_recompute:
            # Check if linear-depth PNGs already exist for all depth frames
            all_linear_exist = True
            missing_count = 0
            total_depth_frames = 0

            for side in Side:
                # Use the cached/built depth dataset to get expected timestamps
                depth_dataset = self.data_io.depth.load_depth_dataset(side=side, use_cache=True)
                depth_timestamps = set(int(t) for t in depth_dataset.timestamps.tolist())
                total_depth_frames += len(depth_timestamps)

                # Get any existing linear-depth timestamps
                linear_timestamps = set(self.data_io.depth.get_linear_depth_timestamps(side=side))

                missing = depth_timestamps - linear_timestamps
                if missing:
                    all_linear_exist = False
                    missing_count += len(missing)

            if all_linear_exist and total_depth_frames > 0:
                print(f"[Info] All linear depth images already exist. Skipping depth-to-linear conversion.")
                for side in Side:
                    linear_count = len(self.data_io.depth.get_linear_depth_timestamps(side=side))
                    print(f"[Info] {side.name}: {linear_count} linear depth images found")
                return

            if missing_count > 0:
                print(f"[Info] Found {missing_count} missing linear depth images. Converting depth to linear...")
            elif total_depth_frames == 0:
                print(f"[Info] No depth frames found. Skipping depth-to-linear conversion.")
                return

        # Ensure we do not use cached depth datasets when force_recompute is enabled.
        if self.force_recompute:
            self.pipeline_configs.depth_to_linear.use_cache = False

        convert_depth_directory(
            depth_data_io=self.data_io.depth,
            depth_to_linear_config=self.pipeline_configs.depth_to_linear
        )


    def reconstruct_scene(self):
        # When forcing recomputation, ensure that reconstruction does not load or
        # reuse any cached intermediate data structures (datasets, fragments, VBG, etc.).
        if self.force_recompute:
            recon_cfg = self.pipeline_configs.reconstruction
            recon_cfg.use_dataset_cache = False
            recon_cfg.use_fragment_dataset_cache = False
            recon_cfg.use_optimized_dataset_cache = False
            recon_cfg.use_colorless_vbg_cache = False

        reconstruct_scene(data_io=self.data_io, config=self.pipeline_configs.reconstruction)
    
    
    def run_full_pipeline(self):
        """
        Run the complete pipeline: YUV->RGB, depth->linear, and reconstruction.
        """
        print("\n" + "="*80)
        print("Running Full Pipeline")
        print("="*80)
        
        print("\n[Step 1/3] Converting YUV to RGB...")
        self.convert_yuv_to_rgb()
        
        print("\n[Step 2/3] Converting depth to linear...")
        self.convert_depth_to_linear()
        
        print("\n[Step 3/3] Reconstructing scene...")
        self.reconstruct_scene()
        
        print("\n" + "="*80)
        print("Pipeline Complete!")
        print("="*80)