#!/usr/bin/env python3
"""
Trim all recordings referenced in the recording length report to a maximum
duration (default: 20.0 seconds) or YUV frame count, while keeping all 
time-dependent files consistent (YUV/RGB images, depth maps & descriptors, 
HMD poses, confidence maps, linear depth, color-aligned depth) and clearing 
derived caches.

Session directories are automatically loaded from recording_length_report.csv.

Console Usage Examples:
    # Trim by duration (default: 17 seconds) - safe mode
    python analysis/processing/trim_recordings.py

    # Trim to specific duration with custom report
    python analysis/processing/trim_recordings.py \
        --length-report analysis/data/recording_length_report.csv \
        --max-duration-s 20.0

    # Trim by YUV frame count instead of duration
    python analysis/processing/trim_recordings.py \
        --mode frames \
        --yuv-frame-limit 923

    # Preview what would be trimmed without making changes
    python analysis/processing/trim_recordings.py \
        --dry-run \
        --max-duration-s 15.0

    # Trim recordings from a different experiment
    python analysis/processing/trim_recordings.py \
        --length-report analysis/data/recording_lengths.csv \
        --max-duration-s 17.1

    # Trim to shorter duration for testing
    python analysis/processing/trim_recordings.py \
        --max-duration-s 10.0 \
        --dry-run
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_LENGTH_REPORT = (
    Path(__file__).resolve().parent / "recording_length_report.csv"
)
DEFAULT_MAX_DURATION_S = 17.


@dataclass
class TimeStats:
    start_ms: Optional[int]
    end_ms: Optional[int]
    count: int

    @property
    def duration_s(self) -> Optional[float]:
        if self.start_ms is None or self.end_ms is None:
            return None
        return (self.end_ms - self.start_ms) / 1000.0


def _parse_timestamp_stem(stem: str) -> Optional[int]:
    """
    Parse integer timestamp from a filename stem, handling macOS sidecar
    prefixes that sometimes appear on external drives.
    """
    if stem.startswith("._"):
        stem = stem[2:]
    elif stem.startswith("_"):
        stem = stem.lstrip("_")
    return int(stem) if stem.isdigit() else None


def _gather_file_timestamps(dir_path: Path, suffix: str) -> TimeStats:
    timestamps: list[int] = []
    if not dir_path.exists():
        return TimeStats(None, None, 0)

    for file_path in sorted(dir_path.glob(f"*{suffix}")):
        ts = _parse_timestamp_stem(file_path.stem)
        if ts is None:
            continue
        timestamps.append(ts)

    if not timestamps:
        return TimeStats(None, None, 0)

    return TimeStats(min(timestamps), max(timestamps), len(timestamps))


def _gather_depth_timestamps(session_dir: Path, side_prefix: str) -> TimeStats:
    descriptor_path = session_dir / f"{side_prefix}_depth_descriptors.csv"
    if not descriptor_path.exists():
        return TimeStats(None, None, 0)

    timestamps: list[int] = []
    with descriptor_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = "timestamp_ms"
        if col not in (reader.fieldnames or []):
            return TimeStats(None, None, 0)
        for row in reader:
            try:
                timestamps.append(int(row[col]))
            except (TypeError, ValueError):
                continue

    if not timestamps:
        return TimeStats(None, None, 0)

    return TimeStats(min(timestamps), max(timestamps), len(timestamps))


def _gather_hmd_pose_timestamps(session_dir: Path) -> TimeStats:
    pose_path = session_dir / "hmd_poses.csv"
    if not pose_path.exists():
        return TimeStats(None, None, 0)

    timestamps: list[int] = []
    with pose_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        col = "unix_time" if "unix_time" in (reader.fieldnames or []) else None
        if col is None:
            return TimeStats(None, None, 0)
        for row in reader:
            try:
                timestamps.append(int(float(row[col])))
            except (TypeError, ValueError):
                continue

    if not timestamps:
        return TimeStats(None, None, 0)

    return TimeStats(min(timestamps), max(timestamps), len(timestamps))


def _combine_modalities(modalities: Iterable[TimeStats]) -> TimeStats:
    starts = [m.start_ms for m in modalities if m.start_ms is not None]
    ends = [m.end_ms for m in modalities if m.end_ms is not None]
    counts = [m.count for m in modalities if m.count > 0]

    if not starts or not ends:
        return TimeStats(None, None, sum(counts))

    return TimeStats(min(starts), max(ends), sum(counts))


def _trim_timestamped_dir(
    dir_path: Path, suffix: str, cutoff_ms: int, dry_run: bool
) -> bool:
    """
    Trim timestamped files in a directory.
    Returns True if any files would be deleted, False otherwise.
    """
    if not dir_path.exists():
        return False
    
    files_to_delete = []
    for file_path in dir_path.glob(f"*{suffix}"):
        ts = _parse_timestamp_stem(file_path.stem)
        if ts is None:
            continue
        if ts > cutoff_ms:
            files_to_delete.append(file_path)
    
    if not files_to_delete:
        return False
    
    if dry_run:
        # In dry-run, only print the directory once
        print(f"[DRY] Would delete {len(files_to_delete)} file(s) from {dir_path}")
    else:
        for file_path in files_to_delete:
            try:
                file_path.unlink()
                print(f"[Trim] Deleted {file_path}")
            except OSError as e:
                print(f"[Error] Failed to delete {file_path}: {e}")
    
    return True


def _trim_depth_modalities(session_dir: Path, side_prefix: str, cutoff_ms: int, dry_run: bool) -> None:
    # 1) Filter depth descriptors CSV
    descriptor_path = session_dir / f"{side_prefix}_depth_descriptors.csv"
    kept_timestamps: set[int] = set()
    if descriptor_path.exists():
        with descriptor_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            rows = list(reader)

        new_rows: list[dict[str, str]] = []
        for row in rows:
            try:
                ts = int(row["timestamp_ms"])
            except (KeyError, ValueError, TypeError):
                continue
            if ts <= cutoff_ms:
                new_rows.append(row)
                kept_timestamps.add(ts)

        if dry_run:
            removed = len(rows) - len(new_rows)
            if removed > 0:
                print(f"[DRY] Would modify {descriptor_path} (keep {len(new_rows)} / {len(rows)} rows, remove {removed})")
        else:
            with descriptor_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(new_rows)
            print(
                f"[Trim] Rewrote {descriptor_path} with "
                f"{len(new_rows)} / {len(rows)} rows (cutoff_ms={cutoff_ms})"
            )

    # 2) Depth maps (.raw) – delete files whose timestamp is > cutoff or no longer in descriptors
    depth_dir = session_dir / f"{side_prefix}_depth"
    files_to_delete = []
    if depth_dir.exists():
        for file_path in depth_dir.glob("*.raw"):
            ts = _parse_timestamp_stem(file_path.stem)
            if ts is None:
                continue
            if ts > cutoff_ms or (kept_timestamps and ts not in kept_timestamps):
                files_to_delete.append(file_path)
        
        if files_to_delete:
            if dry_run:
                print(f"[DRY] Would delete {len(files_to_delete)} file(s) from {depth_dir}")
            else:
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        print(f"[Trim] Deleted {file_path}")
                    except OSError as e:
                        print(f"[Error] Failed to delete {file_path}: {e}")

    # 3) Confidence maps
    conf_dir = session_dir / f"{side_prefix}_depth_confidence"
    _trim_timestamped_dir(conf_dir, ".npz", cutoff_ms, dry_run)

    # 4) Linear depth
    linear_dir = session_dir / f"{side_prefix}_depth_linear"
    _trim_timestamped_dir(linear_dir, ".png", cutoff_ms, dry_run)


def _trim_hmd_poses(session_dir: Path, cutoff_ms: int, dry_run: bool) -> None:
    pose_path = session_dir / "hmd_poses.csv"
    if not pose_path.exists():
        return

    with pose_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    new_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            ts = int(float(row["unix_time"]))
        except (KeyError, ValueError, TypeError):
            continue
        if ts <= cutoff_ms:
            new_rows.append(row)

    removed = len(rows) - len(new_rows)
    if removed > 0:
        if dry_run:
            print(f"[DRY] Would modify {pose_path} (keep {len(new_rows)} / {len(rows)} rows, remove {removed})")
        else:
            with pose_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(new_rows)
            print(
                f"[Trim] Rewrote {pose_path} with "
                f"{len(new_rows)} / {len(rows)} rows (cutoff_ms={cutoff_ms})"
            )


def _trim_color_aligned_depth(session_dir: Path, cutoff_ms: int, dry_run: bool) -> None:
    # Top-level color-aligned depth dirs (as per RGBDPathConfig)
    for side_prefix in ("left", "right"):
        cad_dir = session_dir / f"{side_prefix}_color_aligned_depth"
        _trim_timestamped_dir(cad_dir, ".npy", cutoff_ms, dry_run)

    # In case aligned depth is stored under reconstruction/aligned_depth
    aligned_dir = session_dir / "reconstruction" / "aligned_depth"
    _trim_timestamped_dir(aligned_dir, ".npy", cutoff_ms, dry_run)


def _remove_tree(path: Path, dry_run: bool) -> None:
    if not path.exists():
        return
    if dry_run:
        print(f"[DRY] Would remove directory tree {path}")
        return

    # Remove files then directories bottom-up
    for p in sorted(path.rglob("*"), reverse=True):
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        except OSError as e:
            print(f"[Error] Failed to remove {p}: {e}")
    try:
        path.rmdir()
    except OSError:
        pass
    print(f"[Trim] Removed directory tree {path}")


def _clear_caches_and_reconstruction(session_dir: Path, dry_run: bool) -> None:
    # Remove dataset caches, fragment caches, and reconstruction outputs so they
    # can be rebuilt for the truncated recordings.
    for rel in ("dataset", "cache", "reconstruction"):
        _remove_tree(session_dir / rel, dry_run=dry_run)


def analyze_session_time_bounds(session_dir: Path) -> TimeStats:
    yuv_left = _gather_file_timestamps(session_dir / "left_camera_raw", ".yuv")
    yuv_right = _gather_file_timestamps(session_dir / "right_camera_raw", ".yuv")
    rgb_left = _gather_file_timestamps(session_dir / "left_camera_rgb", ".png")
    rgb_right = _gather_file_timestamps(session_dir / "right_camera_rgb", ".png")

    depth_left = _gather_depth_timestamps(session_dir, "left")
    depth_right = _gather_depth_timestamps(session_dir, "right")
    hmd = _gather_hmd_pose_timestamps(session_dir)

    return _combine_modalities(
        [yuv_left, yuv_right, rgb_left, rgb_right, depth_left, depth_right, hmd]
    )


def compute_cutoff_from_yuv_frames(
    session_dir: Path, frame_limit: int
) -> Optional[int]:
    """
    Determine cutoff timestamp based on the N-th earliest YUV frame across
    left/right streams. Returns the cutoff timestamp in ms, or None if there
    are fewer than `frame_limit` valid YUV frames.
    """
    timestamps: list[int] = []
    for dir_name in ("left_camera_raw", "right_camera_raw"):
        # _gather_file_timestamps only returns min/max, so we need to re-list here
        dir_path = session_dir / dir_name
        if not dir_path.exists():
            continue
        for file_path in sorted(dir_path.glob("*.yuv")):
            ts = _parse_timestamp_stem(file_path.stem)
            if ts is None:
                continue
            timestamps.append(ts)

    if not timestamps:
        return None

    timestamps = sorted(set(timestamps))
    if len(timestamps) < frame_limit:
        return None

    return timestamps[frame_limit - 1]


def read_session_dirs(report_csv: Path) -> list[Path]:
    with report_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        dirs = [Path(row["session_dir"]) for row in reader if "session_dir" in row]
    # Deduplicate while preserving order
    seen = set()
    unique_dirs: list[Path] = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)
    return unique_dirs


def trim_session(
    session_dir: Path,
    max_duration_s: float,
    dry_run: bool,
    mode: str,
    yuv_frame_limit: int,
) -> None:
    if not session_dir.exists():
        print(f"[Skip] Session directory does not exist: {session_dir}")
        return

    if mode == "time":
        # For time-based trimming, we need to determine the start time from YUV/RGB files
        # specifically, not from all modalities (HMD might start earlier)
        yuv_left = _gather_file_timestamps(session_dir / "left_camera_raw", ".yuv")
        yuv_right = _gather_file_timestamps(session_dir / "right_camera_raw", ".yuv")
        rgb_left = _gather_file_timestamps(session_dir / "left_camera_rgb", ".png")
        rgb_right = _gather_file_timestamps(session_dir / "right_camera_rgb", ".png")
        
        # Find the earliest start time from YUV/RGB files
        yuv_rgb_starts = [
            ts for ts in [
                yuv_left.start_ms, yuv_right.start_ms,
                rgb_left.start_ms, rgb_right.start_ms
            ] if ts is not None
        ]
        
        if not yuv_rgb_starts:
            print(f"[Skip] No YUV or RGB files found to determine start time for {session_dir}")
            return
        
        yuv_rgb_start_ms = min(yuv_rgb_starts)
        
        # Also get overall bounds for duration check
        bounds = analyze_session_time_bounds(session_dir)
        if bounds.start_ms is None or bounds.end_ms is None:
            print(f"[Skip] Could not determine time bounds for {session_dir}")
            return

        cutoff_ms = yuv_rgb_start_ms + int(max_duration_s * 1000.0)
        duration = bounds.duration_s
        if duration is None or duration <= max_duration_s:
            print(f"[Info] Session already <= {max_duration_s:.1f}s ({duration or 0:.3f}s): {session_dir}")
            return

        print(
            f"[Trim] Session {session_dir} duration {duration:.3f}s "
            f"-> cutoff at {max_duration_s:.3f}s from YUV/RGB start {yuv_rgb_start_ms} (cutoff_ms={cutoff_ms})"
        )
    elif mode == "frames":
        cutoff_ts = compute_cutoff_from_yuv_frames(session_dir, frame_limit=yuv_frame_limit)
        if cutoff_ts is None:
            print(
                f"[Skip] Not enough YUV frames (< {yuv_frame_limit}) to trim by frame count "
                f"for {session_dir}"
            )
            return
        cutoff_ms = cutoff_ts
        print(
            f"[Trim] Session {session_dir} by YUV frames: "
            f"cutoff at frame #{yuv_frame_limit} timestamp={cutoff_ms}"
        )
    else:
        print(f"[Error] Unknown trim mode '{mode}' for session {session_dir}")
        return

    # YUV/RGB images
    _trim_timestamped_dir(session_dir / "left_camera_raw", ".yuv", cutoff_ms, dry_run)
    _trim_timestamped_dir(session_dir / "right_camera_raw", ".yuv", cutoff_ms, dry_run)
    _trim_timestamped_dir(session_dir / "left_camera_rgb", ".png", cutoff_ms, dry_run)
    _trim_timestamped_dir(session_dir / "right_camera_rgb", ".png", cutoff_ms, dry_run)

    # Depth + descriptors + confidence + linear depth
    _trim_depth_modalities(session_dir, "left", cutoff_ms, dry_run)
    _trim_depth_modalities(session_dir, "right", cutoff_ms, dry_run)

    # HMD poses
    _trim_hmd_poses(session_dir, cutoff_ms, dry_run)

    # Color-aligned depth (top-level and under reconstruction)
    _trim_color_aligned_depth(session_dir, cutoff_ms, dry_run)

    # Derived caches and reconstruction outputs
    _clear_caches_and_reconstruction(session_dir, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trim sessions to a maximum duration (default: 20.0 s), updating all "
            "time-dependent files and clearing caches. Can trim a single session "
            "directory or all sessions from a CSV report."
        )
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Single session directory to trim (overrides --length-report). Example: /Volumes/Intenso/Fog/20251209_144306",
    )
    parser.add_argument(
        "--length-report",
        type=Path,
        default=DEFAULT_LENGTH_REPORT,
        help="Path to recording_length_report.csv (from analyze_recording_lengths.py). Ignored if --session-dir is provided.",
    )
    parser.add_argument(
        "--max-duration-s",
        type=float,
        default=DEFAULT_MAX_DURATION_S,
        help="Maximum recording duration in seconds (default: 20.0)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["time", "frames"],
        default="time",
        help="Trimming mode: 'time' (by duration) or 'frames' (by YUV frame count).",
    )
    parser.add_argument(
        "--yuv-frame-limit",
        type=int,
        default=923,
        help="In 'frames' mode, keep only the first N YUV frames (default: 923).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not delete or modify anything; only print actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.session_dir is not None:
        # Single session directory mode
        session_dir = Path(args.session_dir).resolve()
        if not session_dir.exists():
            print(f"[Error] Session directory does not exist: {session_dir}")
            return
        print(f"[Info] Trimming single session: {session_dir}")
        session_dirs = [session_dir]
    else:
        # CSV report mode
        session_dirs = read_session_dirs(args.length_report)
        if not session_dirs:
            print(f"[Warning] No session_dir entries found in {args.length_report}")
            return
        print(f"[Info] Found {len(session_dirs)} session directories from {args.length_report}")
    
    for session_dir in session_dirs:
        trim_session(
            session_dir,
            max_duration_s=args.max_duration_s,
            dry_run=args.dry_run,
            mode=args.mode,
            yuv_frame_limit=args.yuv_frame_limit,
        )


if __name__ == "__main__":
    main()


