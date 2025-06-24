from pathlib import Path
import argparse
import os

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
        "--clip_near",
        type=float,
        default=0.1,
        help="Near clipping plane distance for depth conversion. Default: 0.1"
    )
    parser.add_argument(
        "--clip_far",
        type=float,
        default=10.0,
        help="Far clipping plane distance for depth conversion. Default: 10.0"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.project_dir):
        parser.error(f"Input directory does not exist: {args.project_dir}")

    return args


def main(args):
    clip_near = args.clip_near
    clip_far = args.clip_far
    print(f"[Info] Clip: near={clip_near}, far={clip_far}")

    project_manager = ProjectManager(args.project_dir)
    project_manager.convert_depth_to_linear_map(
        clip_near=clip_near,
        clip_far=clip_far
    )
    print("[Info] Depth conversion completed.")



if __name__ == "__main__":
    args = parse_args()

    print(f"[Info] Project Directory: {args.project_dir}")

    main(args)
