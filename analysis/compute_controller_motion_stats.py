#!/usr/bin/env python3
"""Summarize hand controller movement from Quest controller pose logs.

Processes all sessions listed in master_fog_no_fog_report.csv and computes
motion statistics for both left and right controllers, including inter-hand
coordination metrics.

Example:
    python analysis/compute_controller_motion_stats.py \\
        --master-report analysis/master_fog_no_fog_report.csv \\
        --output analysis/controller_analysis.csv
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
class ControllerMovementSummary:
    """Motion statistics for a single controller (left or right)."""
    capture_name: str
    capture_path: str
    participant: str | None
    condition: str | None
    hand: str  # "left" or "right"
    num_samples: int
    duration_seconds: float
    sampling_hz: float
    
    # Linear motion
    total_distance_m: float
    net_displacement_m: float
    avg_speed_kmh: float
    peak_speed_kmh: float
    avg_acceleration_ms2: float
    peak_acceleration_ms2: float
    
    # Angular motion
    cumulative_rotation_rad: float
    avg_angular_speed_rad_s: float
    peak_angular_speed_rad_s: float
    
    # Spatial extent
    workspace_volume_m3: float  # bounding box volume
    workspace_extent_x_m: float
    workspace_extent_y_m: float
    workspace_extent_z_m: float
    
    # Quality metrics
    tracking_gaps: int  # number of gaps > 100ms
    jitter_stddev_m: float  # position jitter (stddev of second derivative)


@dataclass
class InterHandSummary:
    """Metrics describing coordination between left and right hands."""
    capture_name: str
    capture_path: str
    participant: str | None
    condition: str | None
    
    # Distance metrics
    avg_inter_hand_distance_m: float
    min_inter_hand_distance_m: float
    max_inter_hand_distance_m: float
    inter_hand_distance_stddev_m: float
    
    # Relative motion
    avg_relative_speed_kmh: float
    peak_relative_speed_kmh: float
    
    # Coordination
    movement_correlation: float  # correlation of speeds (0-1)
    synchronization_score: float  # how synchronized movements are (0-1)


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


def compute_linear_motion_stats(
    positions: np.ndarray, dt_seconds: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    """
    Compute linear motion statistics.
    
    Returns:
        total_distance_m, net_displacement_m, avg_speed_kmh, peak_speed_kmh,
        avg_acceleration_ms2, peak_acceleration_ms2
    """
    if len(positions) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Compute distances and speeds
    deltas = positions[1:] - positions[:-1]
    step_distances = np.linalg.norm(deltas, axis=1)
    total_distance = float(step_distances.sum())
    net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))

    safe_dt = np.clip(dt_seconds, a_min=1e-9, a_max=None)
    speeds = step_distances / safe_dt
    peak_speed_mps = float(np.max(speeds)) if len(speeds) else 0.0
    
    duration = float(dt_seconds.sum()) if len(dt_seconds) > 0 else 0.0
    avg_speed_mps = total_distance / duration if duration > 0 else 0.0
    
    # Convert to km/h
    mps_to_kmh = 3.6
    avg_speed_kmh = avg_speed_mps * mps_to_kmh
    peak_speed_kmh = peak_speed_mps * mps_to_kmh
    
    # Compute accelerations
    if len(speeds) > 1:
        accelerations = np.diff(speeds) / safe_dt[1:]
        avg_acceleration = float(np.mean(np.abs(accelerations))) if len(accelerations) else 0.0
        peak_acceleration = float(np.max(np.abs(accelerations))) if len(accelerations) else 0.0
    else:
        avg_acceleration = 0.0
        peak_acceleration = 0.0
    
    return total_distance, net_displacement, avg_speed_kmh, peak_speed_kmh, avg_acceleration, peak_acceleration


def compute_angular_motion_stats(
    quaternions: np.ndarray, dt_seconds: np.ndarray
) -> tuple[float, float, float]:
    """
    Compute angular motion statistics.
    
    Returns:
        cumulative_rotation_rad, avg_angular_speed_rad_s, peak_angular_speed_rad_s
    """
    if len(quaternions) < 2:
        return 0.0, 0.0, 0.0

    rots = R.from_quat(quaternions)
    rel_rots = rots[1:] * rots[:-1].inv()
    angles = rel_rots.magnitude()
    
    cumulative_rotation = float(angles.sum())
    
    safe_dt = np.clip(dt_seconds, a_min=1e-9, a_max=None)
    angular_speeds = angles / safe_dt
    
    duration = float(dt_seconds.sum()) if len(dt_seconds) > 0 else 0.0
    avg_angular_speed = cumulative_rotation / duration if duration > 0 else 0.0
    peak_angular_speed = float(np.max(angular_speeds)) if len(angular_speeds) else 0.0
    
    return cumulative_rotation, avg_angular_speed, peak_angular_speed


def compute_workspace_stats(positions: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute workspace statistics (bounding box).
    
    Returns:
        workspace_volume_m3, extent_x_m, extent_y_m, extent_z_m
    """
    if len(positions) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    extents = max_pos - min_pos
    
    extent_x = float(extents[0])
    extent_y = float(extents[1])
    extent_z = float(extents[2])
    volume = extent_x * extent_y * extent_z
    
    return volume, extent_x, extent_y, extent_z


def compute_tracking_quality(timestamps: np.ndarray, positions: np.ndarray) -> tuple[int, float]:
    """
    Compute tracking quality metrics.
    
    Returns:
        tracking_gaps (count of gaps > 100ms), jitter_stddev_m
    """
    if len(timestamps) < 2:
        return 0, 0.0
    
    # Count gaps > 100ms
    dt = np.diff(timestamps) / 1000.0  # convert to seconds
    gaps = np.sum(dt > 0.1)
    
    # Compute jitter (stddev of second derivative of position)
    if len(positions) >= 3:
        # First derivative (velocity)
        velocities = np.diff(positions, axis=0)
        # Second derivative (acceleration/jerk)
        accelerations = np.diff(velocities, axis=0)
        # Jitter is stddev of acceleration magnitude
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        jitter = float(np.std(accel_magnitudes)) if len(accel_magnitudes) > 0 else 0.0
    else:
        jitter = 0.0
    
    return int(gaps), jitter


def load_controller_dataframe(controller_csv_path: Path) -> pd.DataFrame:
    """Load and validate controller pose CSV."""
    if not controller_csv_path.exists():
        raise FileNotFoundError(f"Controller poses CSV not found: {controller_csv_path}")

    if controller_csv_path.stat().st_size == 0:
        raise ValueError(f"Controller poses CSV is empty: {controller_csv_path}")

    try:
        df = pd.read_csv(controller_csv_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Controller poses CSV is empty or has no valid data: {controller_csv_path}")

    if df.empty:
        raise ValueError(f"Controller poses CSV contains no rows: {controller_csv_path}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{controller_csv_path} missing required columns: {missing}")

    df = df.sort_values("unix_time").reset_index(drop=True)
    df = df.dropna(subset=REQUIRED_COLUMNS)
    
    if df.empty:
        raise ValueError(f"{controller_csv_path} has no valid rows after filtering NaN values")

    return df


def summarize_controller(
    capture_dir: Path,
    hand: str,
    participant_mapping: dict[str, tuple[str, str]] | None = None
) -> ControllerMovementSummary:
    """Summarize motion statistics for a single controller."""
    controller_csv = capture_dir / f"{hand}_controller_poses.csv"
    df = load_controller_dataframe(controller_csv)

    timestamps = df["unix_time"].to_numpy(dtype=float)
    time_scale = infer_time_scale_to_seconds(timestamps)

    dt_seconds = np.diff(timestamps) / time_scale if len(timestamps) > 1 else np.array([])
    duration_seconds = float((timestamps[-1] - timestamps[0]) / time_scale) if len(timestamps) > 1 else 0.0
    sampling_hz = float(1.0 / np.median(dt_seconds)) if len(dt_seconds) > 0 and np.median(dt_seconds) > 0 else 0.0

    positions = df[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
    quaternions = df[["rot_x", "rot_y", "rot_z", "rot_w"]].to_numpy(dtype=float)

    # Compute statistics
    total_distance, net_displacement, avg_speed_kmh, peak_speed_kmh, avg_accel, peak_accel = \
        compute_linear_motion_stats(positions, dt_seconds)
    
    cumulative_rotation, avg_angular_speed, peak_angular_speed = \
        compute_angular_motion_stats(quaternions, dt_seconds)
    
    workspace_volume, extent_x, extent_y, extent_z = \
        compute_workspace_stats(positions)
    
    tracking_gaps, jitter = \
        compute_tracking_quality(timestamps, positions)

    # Look up participant and condition
    participant = None
    condition = None
    if participant_mapping:
        capture_path_str = str(capture_dir)
        if capture_path_str in participant_mapping:
            participant, condition = participant_mapping[capture_path_str]
        else:
            # Try to infer condition from path
            if "/Fog/" in capture_path_str:
                condition = "Fog"
            elif "/NoFog/" in capture_path_str:
                condition = "NoFog"

    return ControllerMovementSummary(
        capture_name=capture_dir.name,
        capture_path=str(capture_dir),
        participant=participant,
        condition=condition,
        hand=hand,
        num_samples=len(df),
        duration_seconds=duration_seconds,
        sampling_hz=sampling_hz,
        total_distance_m=total_distance,
        net_displacement_m=net_displacement,
        avg_speed_kmh=avg_speed_kmh,
        peak_speed_kmh=peak_speed_kmh,
        avg_acceleration_ms2=avg_accel,
        peak_acceleration_ms2=peak_accel,
        cumulative_rotation_rad=cumulative_rotation,
        avg_angular_speed_rad_s=avg_angular_speed,
        peak_angular_speed_rad_s=peak_angular_speed,
        workspace_volume_m3=workspace_volume,
        workspace_extent_x_m=extent_x,
        workspace_extent_y_m=extent_y,
        workspace_extent_z_m=extent_z,
        tracking_gaps=tracking_gaps,
        jitter_stddev_m=jitter,
    )


def compute_inter_hand_stats(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    capture_dir: Path,
    participant_mapping: dict[str, tuple[str, str]] | None = None
) -> InterHandSummary:
    """Compute inter-hand coordination metrics."""
    # Align timestamps (interpolate to common timebase)
    left_times = left_df["unix_time"].to_numpy()
    right_times = right_df["unix_time"].to_numpy()
    
    # Use intersection of time ranges
    min_time = max(left_times.min(), right_times.min())
    max_time = min(left_times.max(), right_times.max())
    
    if min_time >= max_time:
        # No overlap
        return InterHandSummary(
            capture_name=capture_dir.name,
            capture_path=str(capture_dir),
            participant=None,
            condition=None,
            avg_inter_hand_distance_m=0.0,
            min_inter_hand_distance_m=0.0,
            max_inter_hand_distance_m=0.0,
            inter_hand_distance_stddev_m=0.0,
            avg_relative_speed_kmh=0.0,
            peak_relative_speed_kmh=0.0,
            movement_correlation=0.0,
            synchronization_score=0.0,
        )
    
    # Filter to overlapping time range
    left_mask = (left_times >= min_time) & (left_times <= max_time)
    right_mask = (right_times >= min_time) & (right_times <= max_time)
    
    left_positions = left_df[left_mask][["pos_x", "pos_y", "pos_z"]].to_numpy()
    right_positions = right_df[right_mask][["pos_x", "pos_y", "pos_z"]].to_numpy()
    left_times_filtered = left_times[left_mask]
    right_times_filtered = right_times[right_mask]
    
    # Interpolate to common timestamps (use left timestamps as reference)
    common_times = left_times_filtered
    right_positions_interp = np.zeros_like(left_positions)
    
    for i, t in enumerate(common_times):
        # Find closest right timestamp
        idx = np.argmin(np.abs(right_times_filtered - t))
        right_positions_interp[i] = right_positions[idx]
    
    # Compute inter-hand distances
    inter_hand_distances = np.linalg.norm(left_positions - right_positions_interp, axis=1)
    avg_distance = float(np.mean(inter_hand_distances))
    min_distance = float(np.min(inter_hand_distances))
    max_distance = float(np.max(inter_hand_distances))
    stddev_distance = float(np.std(inter_hand_distances))
    
    # Compute relative speeds
    left_deltas = left_positions[1:] - left_positions[:-1]
    right_deltas = right_positions_interp[1:] - right_positions_interp[:-1]
    left_speeds = np.linalg.norm(left_deltas, axis=1)
    right_speeds = np.linalg.norm(right_deltas, axis=1)
    
    # Relative speed (magnitude of velocity difference)
    relative_velocities = left_deltas - right_deltas
    relative_speeds = np.linalg.norm(relative_velocities, axis=1)
    
    # Convert to km/h (assuming ~90Hz sampling, approximate)
    dt_approx = 1.0 / 90.0  # approximate
    relative_speeds_kmh = relative_speeds / dt_approx * 3.6
    avg_relative_speed = float(np.mean(relative_speeds_kmh))
    peak_relative_speed = float(np.max(relative_speeds_kmh))
    
    # Movement correlation (correlation of speeds)
    if len(left_speeds) > 1 and len(right_speeds) > 1:
        # Align lengths
        min_len = min(len(left_speeds), len(right_speeds))
        left_speeds_aligned = left_speeds[:min_len]
        right_speeds_aligned = right_speeds[:min_len]
        
        correlation_matrix = np.corrcoef(left_speeds_aligned, right_speeds_aligned)
        movement_correlation = float(correlation_matrix[0, 1]) if not np.isnan(correlation_matrix[0, 1]) else 0.0
    else:
        movement_correlation = 0.0
    
    # Synchronization score (inverse of relative speed, normalized)
    # Higher relative speed = lower synchronization
    sync_score = 1.0 / (1.0 + avg_relative_speed / 10.0)  # normalize by dividing by 10 km/h
    synchronization_score = float(np.clip(sync_score, 0.0, 1.0))
    
    # Look up participant and condition
    participant = None
    condition = None
    if participant_mapping:
        capture_path_str = str(capture_dir)
        if capture_path_str in participant_mapping:
            participant, condition = participant_mapping[capture_path_str]
        else:
            if "/Fog/" in capture_path_str:
                condition = "Fog"
            elif "/NoFog/" in capture_path_str:
                condition = "NoFog"
    
    return InterHandSummary(
        capture_name=capture_dir.name,
        capture_path=str(capture_dir),
        participant=participant,
        condition=condition,
        avg_inter_hand_distance_m=avg_distance,
        min_inter_hand_distance_m=min_distance,
        max_inter_hand_distance_m=max_distance,
        inter_hand_distance_stddev_m=stddev_distance,
        avg_relative_speed_kmh=avg_relative_speed,
        peak_relative_speed_kmh=peak_relative_speed,
        movement_correlation=movement_correlation,
        synchronization_score=synchronization_score,
    )


def load_participant_mapping(master_report_csv: Path | None) -> dict[str, tuple[str, str]]:
    """Load participant and condition mapping from master report CSV."""
    if master_report_csv is None or not master_report_csv.exists():
        return {}
    
    try:
        df = pd.read_csv(master_report_csv)
        if "session_dir" not in df.columns or "participant" not in df.columns or "condition" not in df.columns:
            return {}
        
        mapping = {}
        for _, row in df.iterrows():
            session_dir = str(row["session_dir"])
            participant = str(row["participant"])
            condition = str(row["condition"])
            mapping[session_dir] = (participant, condition)
        
        return mapping
    except Exception as e:
        print(f"[warn] Could not load participant mapping: {e}")
        return {}


def read_session_dirs(master_report_csv: Path) -> list[Path]:
    """Read session directories from master report CSV."""
    if not master_report_csv.exists():
        raise FileNotFoundError(f"Master report CSV not found: {master_report_csv}")
    
    df = pd.read_csv(master_report_csv)
    if "session_dir" not in df.columns:
        raise ValueError(f"Master report CSV missing 'session_dir' column: {master_report_csv}")
    
    session_dirs = []
    for session_dir_str in df["session_dir"].dropna():
        session_path = Path(session_dir_str)
        if session_path.exists():
            session_dirs.append(session_path)
        else:
            print(f"[warn] Skipping missing session_dir: {session_path}")
    
    return session_dirs


def write_summary_csv(
    controller_summaries: Sequence[ControllerMovementSummary],
    inter_hand_summaries: Sequence[InterHandSummary],
    output_path: Path
) -> None:
    """Write combined summary CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrames
    controller_df = pd.DataFrame([asdict(s) for s in controller_summaries])
    inter_hand_df = pd.DataFrame([asdict(s) for s in inter_hand_summaries])
    
    # Merge on capture_name, participant, condition
    merged_df = pd.merge(
        controller_df,
        inter_hand_df,
        on=["capture_name", "capture_path", "participant", "condition"],
        how="outer",
        suffixes=("", "_interhand")
    )
    
    # Reorder columns for readability
    key_cols = ["capture_name", "capture_path", "participant", "condition", "hand"]
    other_cols = [c for c in merged_df.columns if c not in key_cols]
    merged_df = merged_df[key_cols + other_cols]
    
    merged_df.to_csv(output_path, index=False)
    print(f"[info] Wrote controller analysis: {output_path} ({len(controller_summaries)} controller summaries, {len(inter_hand_summaries)} inter-hand summaries)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute controller motion statistics for all sessions in master report."
    )
    parser.add_argument(
        "--master-report",
        type=Path,
        default=Path(__file__).parent / "master_fog_no_fog_report.csv",
        help="Path to master_fog_no_fog_report.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "controller_analysis.csv",
        help="Output CSV path for controller analysis",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load session directories
    session_dirs = read_session_dirs(args.master_report)
    if not session_dirs:
        print(f"[warn] No session directories found in {args.master_report}")
        return
    
    print(f"[info] Found {len(session_dirs)} session directories")
    
    # Load participant mapping
    participant_mapping = load_participant_mapping(args.master_report)
    
    controller_summaries: List[ControllerMovementSummary] = []
    inter_hand_summaries: List[InterHandSummary] = []
    skipped: List[tuple[Path, str]] = []
    
    for session_dir in session_dirs:
        left_path = session_dir / "left_controller_poses.csv"
        right_path = session_dir / "right_controller_poses.csv"
        
        # Process left controller
        if left_path.exists():
            try:
                left_summary = summarize_controller(session_dir, "left", participant_mapping)
                controller_summaries.append(left_summary)
            except Exception as e:
                skipped.append((session_dir, f"left controller: {e}"))
                print(f"[warn] Skipping left controller for {session_dir.name}: {e}")
        else:
            skipped.append((session_dir, "left_controller_poses.csv not found"))
        
        # Process right controller
        if right_path.exists():
            try:
                right_summary = summarize_controller(session_dir, "right", participant_mapping)
                controller_summaries.append(right_summary)
            except Exception as e:
                skipped.append((session_dir, f"right controller: {e}"))
                print(f"[warn] Skipping right controller for {session_dir.name}: {e}")
        else:
            skipped.append((session_dir, "right_controller_poses.csv not found"))
        
        # Process inter-hand metrics (if both controllers exist)
        if left_path.exists() and right_path.exists():
            try:
                left_df = load_controller_dataframe(left_path)
                right_df = load_controller_dataframe(right_path)
                inter_hand_summary = compute_inter_hand_stats(
                    left_df, right_df, session_dir, participant_mapping
                )
                inter_hand_summaries.append(inter_hand_summary)
            except Exception as e:
                skipped.append((session_dir, f"inter-hand: {e}"))
                print(f"[warn] Skipping inter-hand analysis for {session_dir.name}: {e}")
    
    if skipped:
        print(f"\n[info] Skipped {len(skipped)} session(s) due to errors")
    
    # Write output
    write_summary_csv(controller_summaries, inter_hand_summaries, args.output)
    print(f"[info] Processed {len(controller_summaries)} controller summaries successfully")


if __name__ == "__main__":
    main()

