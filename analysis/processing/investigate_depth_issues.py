#!/usr/bin/env python3
"""
Diagnostic script to investigate depth map conversion issues.

This script analyzes raw depth files, their descriptors, and the resulting
linear depth maps to identify potential issues in the conversion process.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from config.project_path_config import ProjectPathConfig
from dataio.depth_data_io import DepthDataIO
from models.side import Side


def analyze_raw_depth_file(
    raw_path: Path, width: int, height: int
) -> dict:
    """Analyze a raw depth file and return statistics."""
    if not raw_path.exists():
        return {"error": "File not found"}
    
    try:
        depth_array = np.fromfile(raw_path, dtype='<f4')
        expected_size = width * height
        
        if len(depth_array) != expected_size:
            return {
                "error": f"Size mismatch: expected {expected_size}, got {len(depth_array)}",
                "file_size_bytes": raw_path.stat().st_size,
                "expected_size": expected_size * 4,  # 4 bytes per float32
                "actual_size": len(depth_array) * 4
            }
        
        depth_map = depth_array.reshape((height, width))
        
        return {
            "min": float(np.nanmin(depth_map)),
            "max": float(np.nanmax(depth_map)),
            "mean": float(np.nanmean(depth_map)),
            "median": float(np.nanmedian(depth_map)),
            "std": float(np.nanstd(depth_map)),
            "has_nan": bool(np.isnan(depth_map).any()),
            "has_inf": bool(np.isinf(depth_map).any()),
            "zero_count": int(np.sum(depth_map == 0)),
            "one_count": int(np.sum(depth_map == 1)),
            "valid_count": int(np.sum((depth_map > 0) & (depth_map < 1) & np.isfinite(depth_map))),
            "file_size_bytes": raw_path.stat().st_size
        }
    except Exception as e:
        return {"error": str(e)}


def analyze_linear_depth_file(linear_path: Path) -> dict:
    """Analyze a linear depth PNG file and return statistics."""
    if not linear_path.exists():
        return {"error": "File not found"}
    
    try:
        img = cv2.imread(str(linear_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {"error": "Failed to load image"}
        
        # Convert to float for analysis
        img_float = img.astype(np.float32)
        
        return {
            "shape": img.shape,
            "dtype": str(img.dtype),
            "min": float(np.min(img_float)),
            "max": float(np.max(img_float)),
            "mean": float(np.mean(img_float)),
            "median": float(np.median(img_float)),
            "std": float(np.std(img_float)),
            "zero_count": int(np.sum(img == 0)),
            "max_count": int(np.sum(img == 255)),
            "file_size_bytes": linear_path.stat().st_size
        }
    except Exception as e:
        return {"error": str(e)}


def check_depth_conversion(
    depth_data_io: DepthDataIO,
    side: Side,
    timestamp: int,
    width: int,
    height: int,
    near: float,
    far: float
) -> dict:
    """Check if a depth map can be loaded and converted correctly."""
    result = {
        "timestamp": timestamp,
        "side": side.name,
        "width": width,
        "height": height,
        "near": near,
        "far": far
    }
    
    # Check raw depth file
    raw_path = depth_data_io.depth_path_config.get_depth_map_path(side=side, timestamp=timestamp)
    result["raw_analysis"] = analyze_raw_depth_file(raw_path, width, height)
    
    # Check if conversion would work
    if "error" not in result["raw_analysis"]:
        try:
            depth_array = np.fromfile(raw_path, dtype='<f4').reshape((height, width))
            
            # Check validation
            is_valid = depth_data_io.is_depth_map_valid(depth_array)
            result["passes_validation"] = is_valid
            
            if is_valid:
                # Try conversion
                from utils.depth_utils import convert_depth_to_linear
                linear_depth = convert_depth_to_linear(depth_array, near, far)
                
                result["conversion"] = {
                    "success": True,
                    "linear_min": float(np.nanmin(linear_depth)),
                    "linear_max": float(np.nanmax(linear_depth)),
                    "linear_mean": float(np.nanmean(linear_depth)),
                    "has_nan": bool(np.isnan(linear_depth).any()),
                    "has_inf": bool(np.isinf(linear_depth).any()),
                    "negative_count": int(np.sum(linear_depth < 0))
                }
            else:
                result["conversion"] = {"success": False, "reason": "Failed validation"}
        except Exception as e:
            result["conversion"] = {"success": False, "error": str(e)}
    
    # Check linear depth file if it exists
    linear_path = depth_data_io.depth_path_config.get_linear_depth_dir(side=side) / f"{timestamp}.png"
    result["linear_analysis"] = analyze_linear_depth_file(linear_path)
    
    return result


def investigate_recording(
    session_dir: Path,
    start_timestamp: Optional[int] = None,
    end_timestamp: Optional[int] = None,
    side: Side = Side.LEFT
):
    """Investigate depth issues in a recording."""
    print(f"[Info] Investigating depth issues in: {session_dir}")
    print(f"[Info] Side: {side.name}")
    
    # Initialize path config and depth data IO
    project_path_config = ProjectPathConfig(project_dir=session_dir)
    depth_data_io = DepthDataIO(depth_path_config=project_path_config.depth)
    
    # Load depth descriptors
    descriptors = depth_data_io.load_depth_descriptors(side=side)
    print(f"[Info] Found {len(descriptors)} depth frames in descriptors")
    
    # Filter by timestamp range if provided
    if start_timestamp or end_timestamp:
        if start_timestamp:
            descriptors = descriptors[descriptors['timestamp_ms'] >= start_timestamp]
        if end_timestamp:
            descriptors = descriptors[descriptors['timestamp_ms'] <= end_timestamp]
        print(f"[Info] Filtered to {len(descriptors)} frames in timestamp range")
    
    # Analyze frames
    issues_found = []
    normal_frames = []
    
    print("\n[Info] Analyzing depth frames...")
    for idx, (_, row) in enumerate(descriptors.iterrows()):
        timestamp = int(row['timestamp_ms'])
        width = int(row['width'])
        height = int(row['height'])
        near = float(row['near_z'])
        far = float(row['far_z'])
        
        if idx % 50 == 0:
            print(f"  Processing frame {idx+1}/{len(descriptors)} (timestamp: {timestamp})")
        
        analysis = check_depth_conversion(
            depth_data_io, side, timestamp, width, height, near, far
        )
        
        # Check for issues
        has_issue = False
        issue_reasons = []
        
        # Check raw file issues
        if "error" in analysis["raw_analysis"]:
            has_issue = True
            issue_reasons.append(f"Raw file: {analysis['raw_analysis']['error']}")
        
        # Check descriptor values
        if np.isnan(near) or np.isnan(far) or np.isinf(near) or np.isinf(far):
            has_issue = True
            issue_reasons.append(f"Invalid near/far: near={near}, far={far}")
        
        if far <= near:
            has_issue = True
            issue_reasons.append(f"far <= near: near={near}, far={far}")
        
        # Check conversion issues
        if "conversion" in analysis:
            if not analysis["conversion"].get("success", False):
                has_issue = True
                issue_reasons.append(f"Conversion failed: {analysis['conversion']}")
            elif analysis["conversion"].get("has_nan", False) or analysis["conversion"].get("has_inf", False):
                has_issue = True
                issue_reasons.append("Conversion produced NaN/Inf")
            elif analysis["conversion"].get("negative_count", 0) > 0:
                has_issue = True
                issue_reasons.append(f"Conversion produced {analysis['conversion']['negative_count']} negative values")
        
        # Check linear depth file issues
        if "error" in analysis["linear_analysis"]:
            has_issue = True
            issue_reasons.append(f"Linear file: {analysis['linear_analysis']['error']}")
        
        if has_issue:
            analysis["issue_reasons"] = issue_reasons
            issues_found.append(analysis)
        else:
            normal_frames.append(analysis)
    
    # Print summary
    print(f"\n[Summary]")
    print(f"  Total frames analyzed: {len(descriptors)}")
    print(f"  Normal frames: {len(normal_frames)}")
    print(f"  Frames with issues: {len(issues_found)}")
    
    if issues_found:
        print(f"\n[Issues Found]")
        print(f"  First issue at timestamp: {issues_found[0]['timestamp']}")
        print(f"  Last issue at timestamp: {issues_found[-1]['timestamp']}")
        print(f"\n  Issue breakdown:")
        
        issue_types = {}
        for issue in issues_found:
            for reason in issue.get("issue_reasons", []):
                issue_type = reason.split(":")[0]
                issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        for issue_type, count in sorted(issue_types.items()):
            print(f"    {issue_type}: {count} frames")
        
        # Show detailed info for first few issues
        print(f"\n  Detailed analysis of first 5 issues:")
        for i, issue in enumerate(issues_found[:5]):
            print(f"\n  Issue #{i+1} (timestamp: {issue['timestamp']}):")
            print(f"    Reasons: {', '.join(issue.get('issue_reasons', []))}")
            if "raw_analysis" in issue and "error" not in issue["raw_analysis"]:
                raw = issue["raw_analysis"]
                print(f"    Raw depth: min={raw['min']:.4f}, max={raw['max']:.4f}, mean={raw['mean']:.4f}")
                print(f"    Raw depth: valid={raw['valid_count']}, zeros={raw['zero_count']}, ones={raw['one_count']}")
            print(f"    Descriptor: near={issue['near']}, far={issue['far']}, size={issue['width']}x{issue['height']}")
            if "conversion" in issue:
                conv = issue["conversion"]
                if conv.get("success"):
                    print(f"    Conversion: min={conv['linear_min']:.4f}, max={conv['linear_max']:.4f}")
                else:
                    print(f"    Conversion: {conv}")
    
    # Compare problematic range with normal range
    if issues_found and normal_frames:
        print(f"\n[Comparison]")
        print(f"  Normal frames (sample of {min(10, len(normal_frames))}):")
        for frame in normal_frames[:10]:
            if "raw_analysis" in frame and "error" not in frame["raw_analysis"]:
                raw = frame["raw_analysis"]
                print(f"    ts={frame['timestamp']}: near={frame['near']:.4f}, far={frame['far']:.4f}, "
                      f"raw_range=[{raw['min']:.4f}, {raw['max']:.4f}]")
    
    return issues_found, normal_frames


def main():
    parser = argparse.ArgumentParser(
        description="Investigate depth map conversion issues"
    )
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Session directory to investigate"
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["left", "right"],
        default="left",
        help="Camera side to investigate (default: left)"
    )
    parser.add_argument(
        "--start-timestamp",
        type=int,
        help="Start timestamp to investigate (inclusive)"
    )
    parser.add_argument(
        "--end-timestamp",
        type=int,
        help="End timestamp to investigate (inclusive)"
    )
    
    args = parser.parse_args()
    
    side = Side.LEFT if args.side == "left" else Side.RIGHT
    
    issues, normal = investigate_recording(
        args.session_dir,
        start_timestamp=args.start_timestamp,
        end_timestamp=args.end_timestamp,
        side=side
    )


if __name__ == "__main__":
    main()

