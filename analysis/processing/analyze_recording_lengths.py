#!/usr/bin/env python3
"""
Analyze recording durations for experiment sessions.

This script calculates accurate recording durations by examining multiple data sources
within session directories. It can analyze either a single specified session or batch
process all sessions listed in a master fog/no-fog report.

Key features:
- Examines multiple data sources: YUV frames, RGB frames, depth descriptors, HMD pose logs
- Calculates start/end timestamps from filename patterns and metadata
- Handles macOS sidecar files and timestamp parsing edge cases
- Provides detailed console output for single session analysis
- Generates comprehensive CSV reports for batch analysis
- Validates data integrity by cross-checking multiple timestamp sources

Console Usage Examples:
    # Analyze a single session directory with detailed console output
    python analysis/processing/analyze_recording_lengths.py \
        --session-dir /Volumes/Intenso/Fog/20251209_144306

    # Batch analyze recording lengths using default master report
    python analysis/processing/analyze_recording_lengths.py

    # Specify custom master report and output file for batch processing
    python analysis/processing/analyze_recording_lengths.py \
        --master-report analysis/data/master_fog_no_fog_report.csv \
        --output analysis/data/recording_lengths.csv

    # Process recordings from a different experiment
    python analysis/processing/analyze_recording_lengths.py \
        --master-report /data/experiment2/master_report.csv \
        --output /results/experiment2/durations.csv
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_MASTER_REPORT = (
    Path(__file__).resolve().parent / "master_fog_no_fog_report.csv"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "recording_length_report.csv"


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


def analyze_session(session_dir: Path) -> dict[str, str]:
    yuv_left = _gather_file_timestamps(session_dir / "left_camera_raw", ".yuv")
    yuv_right = _gather_file_timestamps(session_dir / "right_camera_raw", ".yuv")
    rgb_left = _gather_file_timestamps(session_dir / "left_camera_rgb", ".png")
    rgb_right = _gather_file_timestamps(session_dir / "right_camera_rgb", ".png")

    depth_left = _gather_depth_timestamps(session_dir, "left")
    depth_right = _gather_depth_timestamps(session_dir, "right")

    hmd = _gather_hmd_pose_timestamps(session_dir)

    combined = _combine_modalities(
        [yuv_left, yuv_right, rgb_left, rgb_right, depth_left, depth_right, hmd]
    )

    def fmt_time(value: Optional[int]) -> str:
        return "" if value is None else str(value)

    def fmt_float(value: Optional[float]) -> str:
        return "" if value is None else f"{value:.3f}"

    def to_row(prefix: str, stats: TimeStats) -> dict[str, str]:
        return {
            f"{prefix}_start_ms": fmt_time(stats.start_ms),
            f"{prefix}_end_ms": fmt_time(stats.end_ms),
            f"{prefix}_duration_s": fmt_float(stats.duration_s),
            f"{prefix}_count": str(stats.count),
        }

    row: dict[str, str] = {
        "session_dir": str(session_dir),
        "session_dir_exists": str(session_dir.exists()),
    }
    row |= to_row("yuv_left", yuv_left)
    row |= to_row("yuv_right", yuv_right)
    row |= to_row("rgb_left", rgb_left)
    row |= to_row("rgb_right", rgb_right)
    row |= to_row("depth_left", depth_left)
    row |= to_row("depth_right", depth_right)
    row |= to_row("hmd", hmd)
    row |= to_row("overall", combined)

    return row


def print_session_analysis(session_dir: Path, row: dict[str, str]) -> None:
    """Print detailed session analysis results to console in a legible format."""
    print("=" * 80)
    print(f"RECORDING LENGTH ANALYSIS: {session_dir.name}")
    print("=" * 80)
    print(f"Session Directory: {session_dir}")
    print(f"Directory Exists: {row.get('session_dir_exists', 'Unknown')}")
    print()

    # Overall duration
    overall_duration = row.get('overall_duration_s', '')
    if overall_duration:
        duration_sec = float(overall_duration)
        print("OVERALL RECORDING DURATION:")
        print(f"  Total Duration: {duration_sec:.3f} seconds ({duration_sec/60:.2f} minutes)")
        print()

    # Individual modalities
    print("MODALITY BREAKDOWN:")
    modalities = [
        ("YUV Left Camera", "yuv_left"),
        ("YUV Right Camera", "yuv_right"),
        ("RGB Left Camera", "rgb_left"),
        ("RGB Right Camera", "rgb_right"),
        ("Depth Left Camera", "depth_left"),
        ("Depth Right Camera", "depth_right"),
        ("HMD Poses", "hmd"),
    ]

    for name, prefix in modalities:
        duration = row.get(f'{prefix}_duration_s', '')
        count = row.get(f'{prefix}_count', '0')
        start = row.get(f'{prefix}_start_ms', '')
        end = row.get(f'{prefix}_end_ms', '')

        print(f"  {name}:")
        if duration:
            print(f"    Duration: {float(duration):.3f} seconds")
        else:
            print("    Duration: No data available")
        print(f"    Sample Count: {count}")
        if start and end:
            print(f"    Time Range: {start}ms - {end}ms")
        print()

    # Summary
    if overall_duration:
        duration_sec = float(overall_duration)
        status = "✓ Within expected range (≤20s)" if duration_sec <= 20.0 else f"⚠ Longer than expected ({duration_sec:.1f}s > 20s)"
        print(f"STATUS: {status}")

    print("=" * 80)


def read_session_dirs(report_csv: Path) -> list[Path]:
    with report_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        candidate_cols = [c for c in fieldnames if c and c.endswith("_session_dir")]
        if "session_dir" in fieldnames:
            candidate_cols.insert(0, "session_dir")

        dirs: list[Path] = []
        seen: set[str] = set()
        for row in reader:
            for col in candidate_cols:
                raw_val = row.get(col, "")
                if raw_val is None:
                    continue
                path_str = str(raw_val).strip()
                if not path_str or path_str in seen:
                    continue
                seen.add(path_str)
                dirs.append(Path(path_str))
    return dirs


def write_report(rows: list[dict[str, str]], output_csv: Path) -> None:
    if not rows:
        print("[Info] No rows to write.")
        return

    fieldnames = list(rows[0].keys())
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Info] Wrote recording length report: {output_csv} ({len(rows)} sessions)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze recording durations for experiment sessions. "
            "Can process a single session directory or batch process from a master report."
        )
    )

    # Single session mode
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Single session directory to analyze (shows detailed console output)",
    )

    # Batch mode
    parser.add_argument(
        "--master-report",
        type=Path,
        default=DEFAULT_MASTER_REPORT,
        help="Path to master_fog_no_fog_report.csv produced by analyze_fog_no_fog_mapping.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path for duration summary (batch mode only)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Single session mode
    if args.session_dir:
        if not args.session_dir.exists():
            print(f"[Error] Session directory does not exist: {args.session_dir}")
            return

        print(f"[Info] Analyzing single session: {args.session_dir}")
        row = analyze_session(args.session_dir)
        print_session_analysis(args.session_dir, row)
        return

    # Batch mode (existing functionality)
    session_dirs = read_session_dirs(args.master_report)
    if not session_dirs:
        print(f"[Warning] No session_dir entries found in {args.master_report}")
        return

    rows = [analyze_session(session_dir) for session_dir in session_dirs]
    write_report(rows, args.output)

    over_20s = [r for r in rows if r.get("overall_duration_s") and float(r["overall_duration_s"]) > 20.0]
    if over_20s:
        print(f"[Info] {len(over_20s)} session(s) longer than 20.0s (see output CSV).")
    else:
        print("[Info] All sessions are <= 20.0s or missing duration data.")


if __name__ == "__main__":
    main()


