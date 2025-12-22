#!/usr/bin/env python3
"""
Compute recording durations for every session listed in the master fog/no_fog
report produced by `analyze_fog_no_fog_mapping.py`.

For each `session_dir`, the script inspects YUV, RGB, depth descriptors, and
HMD pose logs to estimate start/end timestamps and durations (seconds).
Outputs a CSV summary.
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
            "Compute recording durations (seconds) for sessions listed in "
            "master_fog_no_fog_report.csv."
        )
    )
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
        help="Output CSV path for duration summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

