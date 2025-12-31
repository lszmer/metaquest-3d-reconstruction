#!/usr/bin/env python3
"""
Find optimal timestamp offset for YUV/RGB files to maximize overlap with depth/HMD recordings.

This script tests different offset values and reports which offset produces the best
overlapping time window between all recording modalities.

Console Usage Examples:
    # Test offset range around a known value
    python analysis/processing/find_optimal_timestamp_offset.py \
        --session-dir /Volumes/Intenso/Fog/20251209_144306 \
        --min-offset -200000 \
        --max-offset -150000 \
        --step 1000

    # Quick test with default range
    python analysis/processing/find_optimal_timestamp_offset.py \
        --session-dir /Volumes/Intenso/Fog/20251209_144306

    # Fine-tune around a specific offset
    python analysis/processing/find_optimal_timestamp_offset.py \
        --session-dir /Volumes/Intenso/Fog/20251209_144306 \
        --min-offset -181000 \
        --max-offset -179000 \
        --step 100
"""

import argparse
from pathlib import Path
from typing import Optional

# Import from analyze_recording_lengths
import sys
sys.path.insert(0, str(Path(__file__).parent))
from analyze_recording_lengths import analyze_session, TimeStats


def test_offset(session_dir: Path, offset_ms: int) -> tuple[float, dict[str, str]]:
    """Test a specific offset and return overlap duration and full analysis."""
    row = analyze_session(session_dir, offset_ms)
    overall_duration = row.get('overall_duration_s', '')
    duration_sec = float(overall_duration) if overall_duration else 0.0
    return duration_sec, row


def find_optimal_offset(
    session_dir: Path,
    min_offset_ms: int = -300000,
    max_offset_ms: int = 0,
    step_ms: int = 5000
) -> dict:
    """Find the offset that produces the maximum overlap duration."""
    print(f"Testing offsets from {min_offset_ms}ms to {max_offset_ms}ms (step: {step_ms}ms)")
    print("=" * 80)
    
    best_offset = None
    best_duration = 0.0
    best_row = None
    results = []
    
    offset = min_offset_ms
    while offset <= max_offset_ms:
        duration, row = test_offset(session_dir, offset)
        results.append({
            'offset_ms': offset,
            'duration_s': duration,
            'has_overlap': duration > 0
        })
        
        if duration > best_duration:
            best_duration = duration
            best_offset = offset
            best_row = row
        
        # Progress indicator
        if (offset - min_offset_ms) % (step_ms * 10) == 0:
            print(f"  Testing offset {offset}ms: {duration:.3f}s overlap")
        
        offset += step_ms
    
    return {
        'best_offset_ms': best_offset,
        'best_duration_s': best_duration,
        'best_row': best_row,
        'all_results': results
    }


def print_results(session_dir: Path, results: dict) -> None:
    """Print the results of the offset search."""
    print("\n" + "=" * 80)
    print("OPTIMAL OFFSET ANALYSIS RESULTS")
    print("=" * 80)
    
    best_offset = results['best_offset_ms']
    best_duration = results['best_duration_s']
    
    if best_offset is None or best_duration == 0:
        print("⚠ No overlapping time window found in tested range")
        print("  Try expanding the offset range with --min-offset and --max-offset")
        return
    
    print(f"\nOPTIMAL OFFSET: -{abs(best_offset)}ms ({best_offset}ms)")
    print(f"Maximum Overlap Duration: {best_duration:.3f} seconds ({best_duration/60:.2f} minutes)")
    print()
    
    # Show top 5 results
    sorted_results = sorted(
        [r for r in results['all_results'] if r['has_overlap']],
        key=lambda x: x['duration_s'],
        reverse=True
    )[:5]
    
    if len(sorted_results) > 1:
        print("Top 5 Offset Values:")
        for i, result in enumerate(sorted_results, 1):
            print(f"  {i}. Offset {result['offset_ms']}ms: {result['duration_s']:.3f}s overlap")
        print()
    
    # Show detailed analysis for best offset
    if results['best_row']:
        row = results['best_row']
        print("DETAILED ANALYSIS AT OPTIMAL OFFSET:")
        print("-" * 80)
        
        yuv_start = row.get('yuv_left_start_ms', '')
        depth_start = row.get('depth_left_start_ms', '')
        hmd_start = row.get('hmd_start_ms', '')
        overall_start = row.get('overall_start_ms', '')
        overall_end = row.get('overall_end_ms', '')
        
        if yuv_start and depth_start and hmd_start:
            print(f"YUV Start (after offset): {yuv_start}ms")
            print(f"Depth Start: {depth_start}ms")
            print(f"HMD Start: {hmd_start}ms")
            print(f"Overlap Window: {overall_start}ms - {overall_end}ms")
            print()
            
            # Calculate remaining misalignments
            if yuv_start and depth_start:
                yuv_int = int(yuv_start)
                depth_int = int(depth_start)
                remaining_diff = depth_int - yuv_int
                if abs(remaining_diff) > 100:
                    print(f"⚠ Remaining YUV-Depth misalignment: {remaining_diff}ms")
            
            if yuv_start and hmd_start:
                yuv_int = int(yuv_start)
                hmd_int = int(hmd_start)
                remaining_diff = hmd_int - yuv_int
                if abs(remaining_diff) > 100:
                    print(f"⚠ Remaining YUV-HMD misalignment: {remaining_diff}ms")
    
    print("=" * 80)
    print(f"\nRECOMMENDED COMMAND:")
    print(f"  python analysis/processing/rename_timestamps.py \\")
    print(f"      --session-dir {session_dir} \\")
    print(f"      --offset-ms {best_offset}")


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal timestamp offset for YUV/RGB files to maximize overlap"
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        required=True,
        help="Session directory to analyze",
    )
    parser.add_argument(
        "--min-offset",
        type=int,
        default=-300000,
        help="Minimum offset to test in milliseconds (default: -300000)",
    )
    parser.add_argument(
        "--max-offset",
        type=int,
        default=0,
        help="Maximum offset to test in milliseconds (default: 0)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5000,
        help="Step size for offset testing in milliseconds (default: 5000)",
    )
    
    args = parser.parse_args()
    
    if not args.session_dir.exists():
        print(f"[Error] Session directory does not exist: {args.session_dir}")
        return
    
    if args.min_offset > args.max_offset:
        print("[Error] min-offset must be less than or equal to max-offset")
        return
    
    print(f"[Info] Finding optimal offset for: {args.session_dir}")
    print()
    
    results = find_optimal_offset(
        args.session_dir,
        args.min_offset,
        args.max_offset,
        args.step
    )
    
    print_results(args.session_dir, results)


if __name__ == "__main__":
    main()

