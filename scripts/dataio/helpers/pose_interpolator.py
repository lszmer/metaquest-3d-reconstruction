from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R, Slerp


class PoseInterpolator:
    def __init__(self, pose_csv_path: Path):
        self.pose_csv_path = pose_csv_path
        self._df: Optional[pd.DataFrame] = None


    @property
    def hmd_pose_df(self) -> pd.DataFrame:
        if self._df is None:
            df = pd.read_csv(self.pose_csv_path, on_bad_lines='skip').dropna()
            df = df.sort_values('unix_time')
            df = df.reset_index(drop=True)

            self._df = df
        
        return self._df


    def find_nearest_frames(self, timestamp: int, window_ms: int = 30) -> tuple[Optional[pd.Series], Optional[pd.Series]]:
        """
        Find nearest pose frames before and after the given timestamp.
        
        Args:
            timestamp: Unix timestamp in microseconds
            window_ms: Maximum time window in milliseconds to search for nearby poses
        
        Returns:
            Tuple of (previous_pose, next_pose) Series, or (None, None) if no poses found within window
        """
        # Convert window from milliseconds to microseconds
        # Timestamps in hmd_poses.csv are in microseconds (13 digits)
        window_us = window_ms * 1000
        
        before = self.hmd_pose_df[self.hmd_pose_df['unix_time'] <= timestamp]
        after = self.hmd_pose_df[self.hmd_pose_df['unix_time'] >= timestamp]

        prev = before.iloc[-1] if not before.empty and abs(timestamp - before.iloc[-1]['unix_time']) <= window_us else None
        next = after.iloc[0] if not after.empty and abs(after.iloc[0]['unix_time'] - timestamp) <= window_us else None

        return prev, next


    def interpolate_pose(self, timestamp: int) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """
        Interpolate pose at the given timestamp.
        
        If timestamp is before first pose or after last pose, uses the nearest pose
        (extrapolation). Otherwise, interpolates between nearest poses.
        """
        prev, next = self.find_nearest_frames(timestamp)
        
        # If no poses found at all, return None
        if prev is None and next is None:
            return None
        
        # If only one pose found (extrapolation case)
        if prev is None:
            # Use the next pose (timestamp is before first pose)
            pos = np.array([next['pos_x'], next['pos_y'], next['pos_z']])
            rot = np.array([next['rot_x'], next['rot_y'], next['rot_z'], next['rot_w']])
            return pos, rot
        elif next is None:
            # Use the previous pose (timestamp is after last pose)
            pos = np.array([prev['pos_x'], prev['pos_y'], prev['pos_z']])
            rot = np.array([prev['rot_x'], prev['rot_y'], prev['rot_z'], prev['rot_w']])
            return pos, rot
        
        # Both poses found - interpolate
        t0 = prev['unix_time']
        t1 = next['unix_time']
        alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

        pos0 = np.array([prev['pos_x'], prev['pos_y'], prev['pos_z']])
        pos1 = np.array([next['pos_x'], next['pos_y'], next['pos_z']])
        pos_interp = (1 - alpha) * pos0 + alpha * pos1

        rots = R.from_quat( [
            [prev['rot_x'], prev['rot_y'], prev['rot_z'], prev['rot_w']],
            [next['rot_x'], next['rot_y'], next['rot_z'], next['rot_w']],
        ] )
        rot_interp = Slerp([0, 1], rots)(alpha).as_quat()

        return pos_interp, rot_interp