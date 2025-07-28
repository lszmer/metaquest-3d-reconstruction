import argparse
from pathlib import Path

from dataio.data_io import DataIO
from processing.test.visualize_camera_tragectories import visualize_camera_trajectories


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory containing QRC data."
    )
    args = parser.parse_args()

    if not args.project_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.project_dir}")

    return args


def main(args):
    data_io = DataIO(project_dir=args.project_dir)
    visualize_camera_trajectories(data_io=data_io)


if __name__ == "__main__":
    args = parse_args()

    print(f"[Info] Project Directory: {args.project_dir}")
    main(args)