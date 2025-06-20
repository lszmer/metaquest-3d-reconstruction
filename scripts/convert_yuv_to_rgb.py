import argparse
from pathlib import Path

from app import ProjectManager


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory containing QRC data."
    )
    parser.add_argument(
        "--filter", action="store_true", default=False,
        help="Enable image quality filtering."
    )
    parser.add_argument(
        "--blur_threshold", type=float, default=50.0,
        help="Blur threshold (Laplacian variance). Lower means more blur. Default: 50.0"
    )
    parser.add_argument(
        "--exposure_threshold_low", type=float, default=0.1,
        help="Cumulative histogram threshold to detect underexposure. Default: 0.1"
    )
    parser.add_argument(
        "--exposure_threshold_high", type=float, default=0.1,
        help="Cumulative histogram threshold to detect overexposure. Default: 0.1"
    )
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.project_dir}")

    return args


def main(args):
    project_manager = ProjectManager(project_dir=args.project_dir)

    print("[Info] Converting YUV to RGB...")
    project_manager.convert_yuv_to_rgb(
        apply_filter=args.filter,
        blur_threshold=args.blur_threshold,
        exposure_threshold_low=args.exposure_threshold_low,
        exposure_threshold_high=args.exposure_threshold_high
    )
    print("[Info] Conversion completed.")


if __name__ == "__main__":
    args = parse_args()

    print(f"[Info] Project Directory: {args.project_dir}")

    main(args)