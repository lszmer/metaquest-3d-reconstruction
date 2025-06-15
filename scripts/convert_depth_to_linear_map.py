from pathlib import Path
from tqdm import tqdm
import argparse
import os

from infra.io.project_manager import ProjectManager, Side


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
    project_manager = ProjectManager(args.project_dir)

    clip_near = args.clip_near
    clip_far = args.clip_far
    print(f"[Info] Clip: near={clip_near}, far={clip_far}")

    for side in Side:
        dataset = project_manager.get_depth_dataset(side)
        depth_grey_repo = project_manager.get_depth_grey_repo(side)

        num_frames = len(dataset.timestamps)

        for i in tqdm(range(num_frames), total=num_frames, desc="Converting depth images"):
            timestamp = dataset.timestamps[i]

            depth_map = project_manager.load_depth_map(
                side=side,
                timestamp=timestamp,
                width=dataset.widths[i],
                height=dataset.heights[i],
                near=dataset.nears[i],
                far=dataset.fars[i]
            )

            depth_grey_repo.save(
                file_stem=timestamp,
                bgr_img=(depth_map - clip_near) / (clip_far - clip_near) * 255.0
            )

        print(f"[Info] Converted depth images for {side} camera to linear format.")


if __name__ == "__main__":
    args = parse_args()

    print(f"[Info] Project Directory: {args.project_dir}")

    main(args)
