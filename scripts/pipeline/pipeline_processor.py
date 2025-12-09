from pathlib import Path
from config.pipeline_configs import PipelineConfigs
from dataio.data_io import DataIO
from models.side import Side
from processing.depth_conversion.convert_depth_to_linear import convert_depth_directory
from processing.reconstruction.reconstruct_scene import reconstruct_scene
from processing.yuv_conversion.convert_yuv_dir import convert_yuv_directory


class PipelineProcessor:
    def __init__(self, project_dir: Path, config_yml_path: Path):
        self.data_io = DataIO(project_dir=project_dir)
        self.pipeline_configs = PipelineConfigs.parse_config_yml(config_yml_path)
        self.project_dir = project_dir


    def convert_yuv_to_rgb(self):
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
        convert_depth_directory(depth_data_io=self.data_io.depth, depth_to_linear_config=self.pipeline_configs.depth_to_linear)


    def reconstruct_scene(self):
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