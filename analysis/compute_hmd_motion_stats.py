"""Summarize body and head movement from Quest HMD pose logs.

Default mode: analyze one explicit capture directory.
Optional mode: auto-iterate captures under /Volumes/Intenso for Fog/NoFog/All.

Examples:
    # Single capture (default style)
    python analysis/compute_hmd_motion_stats.py \
        --session_dir /Volumes/Intenso/NoFog/20251209_153834

    # Sweep all NoFog captures under /Volumes/Intenso/NoFog
    python analysis/compute_hmd_motion_stats.py --mode NoFog

    # Sweep Fog and NoFog captures
    python analysis/compute_hmd_motion_stats.py --mode All

Each capture directory must contain an `hmd_poses.csv`.
Per-capture summaries are written to:
    <capture_dir>/analysis/hmd_movement_summary.csv
You can additionally emit an aggregate CSV via --aggregate_csv.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


REQUIRED_COLUMNS = [
    "unix_time",
    "pos_x",
    "pos_y",
    "pos_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "rot_w",
]


@dataclass
class MovementSummary:
    capture_name: str
    capture_path: str
    num_samples: int
    duration_seconds: float
    sampling_hz: float
    body_distance_m: float
    body_net_displacement_m: float
    body_avg_speed_kmh: float
    body_peak_speed_kmh: float
    head_cumulative_radians: float
    head_avg_angular_speed_rad_s: float
    head_peak_angular_speed_rad_s: float
    yaw_range_rad: float
    pitch_range_rad: float
    roll_range_rad: float


def infer_time_scale_to_seconds(timestamps: np.ndarray) -> float:
    """Infer divisor to convert timestamp deltas to seconds."""
    if len(timestamps) < 2:
        return 1.0

    deltas = np.diff(timestamps)
    median_dt = float(np.median(np.abs(deltas)))

    # Heuristic based on expected Quest logs (usually milliseconds).
    if median_dt > 1e6:
        return 1e9  # nanoseconds to seconds
    if median_dt > 1e3:
        return 1e6  # microseconds to seconds
    if median_dt > 10:
        return 1e3  # milliseconds to seconds
    return 1.0  # already seconds


def compute_head_angles(quaternions: np.ndarray, dt_seconds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-step rotation angles (rad) and angular speeds (rad/s)."""
    if len(quaternions) < 2:
        return np.array([]), np.array([])

    rots = R.from_quat(quaternions)
    rel_rots = rots[1:] * rots[:-1].inv()
    angles = rel_rots.magnitude()

    safe_dt = np.clip(dt_seconds, a_min=1e-9, a_max=None)
    angular_speeds = angles / safe_dt
    return angles, angular_speeds


def compute_body_distance(positions: np.ndarray, dt_seconds: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Return per-step speeds (m/s), total distance, net displacement, and peak speed (m/s)."""
    if len(positions) < 2:
        return np.array([]), 0.0, 0.0, 0.0

    deltas = positions[1:] - positions[:-1]
    step_distances = np.linalg.norm(deltas, axis=1)
    total_distance = float(step_distances.sum())
    net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))

    safe_dt = np.clip(dt_seconds, a_min=1e-9, a_max=None)
    speeds = step_distances / safe_dt
    peak_speed = float(np.max(speeds)) if len(speeds) else 0.0
    return speeds, total_distance, net_displacement, peak_speed


def compute_euler_ranges(quaternions: np.ndarray) -> tuple[float, float, float]:
    """Return yaw/pitch/roll ranges in radians."""
    if len(quaternions) == 0:
        return 0.0, 0.0, 0.0

    eulers = R.from_quat(quaternions).as_euler("xyz", degrees=False)
    ranges = eulers.max(axis=0) - eulers.min(axis=0)
    yaw_range, pitch_range, roll_range = (float(r) for r in ranges)
    return yaw_range, pitch_range, roll_range


def load_pose_dataframe(hmd_csv_path: Path) -> pd.DataFrame:
    if not hmd_csv_path.exists():
        raise FileNotFoundError(f"HMD poses CSV not found: {hmd_csv_path}")

    df = pd.read_csv(hmd_csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{hmd_csv_path} missing required columns: {missing}")

    df = df.sort_values("unix_time").reset_index(drop=True)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df


def summarize_capture(capture_dir: Path) -> MovementSummary:
    hmd_csv = capture_dir / "hmd_poses.csv"
    df = load_pose_dataframe(hmd_csv)

    timestamps = df["unix_time"].to_numpy(dtype=float)
    time_scale = infer_time_scale_to_seconds(timestamps)

    dt_seconds = np.diff(timestamps) / time_scale if len(timestamps) > 1 else np.array([])
    duration_seconds = float((timestamps[-1] - timestamps[0]) / time_scale) if len(timestamps) > 1 else 0.0
    sampling_hz = float(1.0 / np.median(dt_seconds)) if len(dt_seconds) else 0.0

    positions = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    quaternions = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy(dtype=float)

    body_speeds, total_distance, net_displacement, peak_speed = compute_body_distance(positions, dt_seconds)
    angles, angular_speeds = compute_head_angles(quaternions, dt_seconds)
    yaw_range, pitch_range, roll_range = compute_euler_ranges(quaternions)

    head_cumulative = float(angles.sum()) if len(angles) else 0.0
    head_peak_speed = float(np.max(angular_speeds)) if len(angular_speeds) else 0.0
    head_avg_speed = float(head_cumulative / duration_seconds) if duration_seconds > 0 else 0.0
    body_avg_speed_mps = float(total_distance / duration_seconds) if duration_seconds > 0 else 0.0
    body_peak_speed_mps = peak_speed

    # Convert linear speeds to km/h for reporting
    mps_to_kmh = 3.6
    body_avg_speed_kmh = body_avg_speed_mps * mps_to_kmh
    body_peak_speed_kmh = body_peak_speed_mps * mps_to_kmh

    return MovementSummary(
        capture_name=capture_dir.name,
        capture_path=str(capture_dir),
        num_samples=len(df),
        duration_seconds=duration_seconds,
        sampling_hz=sampling_hz,
        body_distance_m=total_distance,
        body_net_displacement_m=net_displacement,
        body_avg_speed_kmh=body_avg_speed_kmh,
        body_peak_speed_kmh=body_peak_speed_kmh,
        head_cumulative_radians=head_cumulative,
        head_avg_angular_speed_rad_s=head_avg_speed,
        head_peak_angular_speed_rad_s=head_peak_speed,
        yaw_range_rad=yaw_range,
        pitch_range_rad=pitch_range,
        roll_range_rad=roll_range,
    )


def write_summary_csv(rows: Sequence[MovementSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([asdict(r) for r in rows])
    df.to_csv(output_path, index=False)
    print(f"[info] wrote summary: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute HMD body/head movement statistics.")

    # Default path-based usage (one session)
    parser.add_argument(
        "--session_dir",
        type=Path,
        default=None,
        help="Explicit capture directory containing hmd_poses.csv (default mode).",
    )

    # Mode-based sweep under /Volumes/Intenso
    parser.add_argument(
        "--mode",
        choices=["Fog", "NoFog", "All"],
        default=None,
        help="When set, auto-iterate captures under /Volumes/Intenso/<mode>. Overrides --session_dir.",
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path("/Volumes/Intenso"),
        help="Root directory that contains Fog/NoFog subfolders.",
    )

    parser.add_argument(
        "--aggregate_csv",
        type=Path,
        default=None,
        help="Optional CSV path to store aggregated summary across all captures.",
    )
    return parser.parse_args()


def collect_capture_dirs(args: argparse.Namespace) -> list[Path]:
    """Resolve capture directories from args."""
    if args.mode is None:
        if args.session_dir is None:
            raise ValueError("Provide --session_dir or choose --mode Fog|NoFog|All.")
        return [args.session_dir]

    modes = ["Fog", "NoFog"] if args.mode == "All" else [args.mode]
    capture_dirs: list[Path] = []
    for mode in modes:
        parent = args.root_dir / mode
        if not parent.exists():
            print(f"[warn] mode '{mode}' directory not found: {parent}")
            continue
        for child in parent.iterdir():
            if child.is_dir() and (child / "hmd_poses.csv").exists():
                capture_dirs.append(child)

    if not capture_dirs:
        raise FileNotFoundError("No capture directories found for the given mode/root_dir.")
    return sorted(capture_dirs)


def main(capture_dirs: Iterable[Path], aggregate_csv: Path | None) -> None:
    summaries: List[MovementSummary] = []
    for capture_dir in capture_dirs:
        summary = summarize_capture(capture_dir)
        summaries.append(summary)

        capture_output = capture_dir / "analysis" / "hmd_movement_summary.csv"
        write_summary_csv([summary], capture_output)

    if aggregate_csv:
        write_summary_csv(summaries, aggregate_csv)


if __name__ == "__main__":
    args = parse_args()
    capture_dirs = collect_capture_dirs(args)
    main(capture_dirs, args.aggregate_csv)

