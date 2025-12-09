#!/usr/bin/env python3
"""
Full pipeline runner that orchestrates the complete 3D reconstruction workflow:
1. Convert YUV to RGB
2. Convert depth to linear
3. Reconstruct scene
4. Convert PLY mesh to FBX
"""

# latest_folder=$(adb shell "ls -1t /sdcard/Android/data/com.CHL.RealityLog/files/" | head -n 1)
# adb shell "ls /sdcard/Android/data/com.CHL.RealityLog/files/$latest_folder/left_depth"
# adb pull "/sdcard/Android/data/com.CHL.RealityLog/files/$latest_folder" ~/Documents/QuestRealityCapture


import argparse
from pathlib import Path
from typing import Optional
import subprocess
import re
import time
import sys

from pipeline.pipeline_processor import PipelineProcessor


def find_latest_session(base_dir: Path):
    """Find the latest session directory matching the pattern YYYYMMDD_HHMMSS."""
    candidates = [d for d in base_dir.iterdir() if d.is_dir() and re.match(r'\d{8}_\d{6}', d.name)]
    if not candidates:
        raise RuntimeError(f"No session directory matching pattern found in {base_dir}")
    latest = max(candidates, key=lambda d: d.stat().st_mtime)
    return latest


def run_pipeline(project_dir: Path, config_path: Path) -> float:
    """
    Run the full reconstruction pipeline.
    
    Returns:
        Total visualization time in seconds (to exclude from timing)
    """
    view_seconds = 0.0
    script_dir = Path(__file__).resolve().parent
    
    # Initialize pipeline processor
    processor = PipelineProcessor(project_dir=project_dir, config_yml_path=config_path)
    
    # Step 1: Convert YUV to RGB
    print("\n" + "="*80)
    print("STEP 1: Converting YUV to RGB")
    print("="*80)
    try:
        processor.convert_yuv_to_rgb()
    except Exception as e:
        print(f"[Error] Failed to convert YUV to RGB: {e}")
        raise
    
    # Step 2: Convert depth to linear
    print("\n" + "="*80)
    print("STEP 2: Converting depth to linear")
    print("="*80)
    try:
        processor.convert_depth_to_linear()
    except Exception as e:
        print(f"[Error] Failed to convert depth to linear: {e}")
        raise
    
    # Step 3: Reconstruct scene
    print("\n" + "="*80)
    print("STEP 3: Reconstructing scene")
    print("="*80)
    try:
        # Run reconstruction in subprocess to capture visualization time
        cmd = [
            sys.executable,
            str(script_dir / "reconstruct_scene.py"),
            "--project_dir", str(project_dir),
            "--config", str(config_path)
        ]
        print(f"Running: {' '.join(cmd)}")
        
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="")
                # Capture visualization time if reported
                if "[VIS] COLORLESS_VIEW_SECONDS:" in line or "[VIS] COLORED_VIEW_SECONDS:" in line:
                    try:
                        val = float(line.strip().split(":")[-1])
                        view_seconds += val
                    except Exception:
                        pass
            ret = proc.wait()
            if ret != 0:
                raise subprocess.CalledProcessError(ret, cmd)
    except Exception as e:
        print(f"[Error] Failed to reconstruct scene: {e}")
        raise
    
    return view_seconds


def convert_reconstruction_mesh_to_fbx(project_dir: Path) -> None:
    """Convert the reconstructed color mesh PLY into FBX using Aspose utility."""
    color_mesh_path = project_dir / "reconstruction" / "color_mesh.ply"
    if not color_mesh_path.exists():
        print(f"[Info] No color mesh found at {color_mesh_path}, skipping FBX export.")
        return
    
    print(f"\n[Info] Converting reconstructed mesh to FBX via Aspose utility: {color_mesh_path}")
    script_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(script_dir / "utils" / "convert_ply_to_fbx_aspose.py"),
        str(color_mesh_path),
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"[Info] FBX conversion complete: {color_mesh_path.with_suffix('.fbx')}")
    except FileNotFoundError as e:
        print(f"[Warning] Failed to launch FBX conversion script: {e}")
        print(f"[Warning] Make sure aspose-3d is installed: pip install aspose-3d")
    except subprocess.CalledProcessError as e:
        print(f"[Warning] FBX conversion script exited with code {e.returncode}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full 3D reconstruction pipeline including FBX conversion."
    )
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        help="Path to project/session directory"
    )
    parser.add_argument(
        "--session_dir", "-s",
        type=Path,
        help="Specify session dir directly. If not given, auto-select latest from project_dir."
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to pipeline config YAML file. Defaults to config/pipeline_config.yml relative to script."
    )
    parser.add_argument(
        "--skip-fbx",
        action="store_true",
        help="Skip FBX conversion step"
    )
    
    args = parser.parse_args()
    
    # Determine project directory
    if args.session_dir:
        project_dir = args.session_dir.resolve()
    elif args.project_dir:
        # Look for latest session in given project dir
        project_dir = find_latest_session(args.project_dir.resolve())
        print(f"[Info] No --session_dir specified. Found latest session: {project_dir}")
    else:
        # Try to use a default location or require explicit input
        parser.error("Either --project_dir or --session_dir must be specified")
    
    # Ensure project_dir is absolute and resolved
    project_dir = project_dir.resolve()
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        script_dir = Path(__file__).resolve().parent
        config_path = script_dir.parent / "config" / "pipeline_config.yml"
        if not config_path.exists():
            parser.error(f"Config file not found: {config_path}. Please specify --config")
    
    print(f"[Info] Project directory: {project_dir}")
    print(f"[Info] Config file: {config_path}")
    
    # Run pipeline
    start_ts = time.time()
    try:
        view_seconds = run_pipeline(project_dir, config_path)
    except Exception as e:
        print(f"\n[Error] Pipeline failed: {e}")
        sys.exit(1)
    
    # Convert to FBX
    if not args.skip_fbx:
        try:
            convert_reconstruction_mesh_to_fbx(project_dir)
        except Exception as e:
            print(f"[Warning] FBX conversion failed: {e}")
    
    end_ts = time.time()
    
    # Timing summary
    left_dir = project_dir / "left_camera_rgb"
    right_dir = project_dir / "right_camera_rgb"
    left_count = len(list(left_dir.glob("*.png"))) if left_dir.exists() else 0
    right_count = len(list(right_dir.glob("*.png"))) if right_dir.exists() else 0
    image_count = max(left_count, right_count)
    
    total_seconds = end_ts - start_ts
    adjusted_seconds = max(0.0, total_seconds - view_seconds)
    secs_per_capture = (adjusted_seconds / image_count) if image_count > 0 else float('nan')
    
    summary_lines = [
        f"Project directory: {project_dir}",
        f"Total runtime (s): {total_seconds:.3f}",
        f"Visualization time excluded (s): {view_seconds:.3f}",
        f"Adjusted runtime (s): {adjusted_seconds:.3f}",
        f"Images (max per side): {image_count}",
        f"Seconds per capture (adjusted): {secs_per_capture:.6f}",
        ""
    ]
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print("\n".join(summary_lines))
    
    out_path = project_dir / "pipeline_runtime.txt"
    try:
        out_path.write_text("\n".join(summary_lines), encoding="utf-8")
        print(f"[Info] Wrote timing summary to: {out_path}")
    except Exception as e:
        print(f"[Warning] Failed to write timing summary: {e}")


if __name__ == "__main__":
    main()

