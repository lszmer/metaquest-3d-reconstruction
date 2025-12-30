#!/usr/bin/env python3
"""
Analyze why overall duration differs from individual modality durations.
"""

import csv
from pathlib import Path

def analyze_session(session_dir: Path):
    """Analyze timing for a specific session."""
    print(f"\n{'='*80}")
    print(f"Analysis for: {session_dir}")
    print('='*80)
    
    # Read the recording length report
    report_path = Path(__file__).parent / "recording_length_report.csv"
    with report_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["session_dir"] == str(session_dir):
                print(f"\nIndividual Modality Durations:")
                print(f"  YUV Left:   {row.get('yuv_left_duration_s', 'N/A')}s")
                print(f"  YUV Right:  {row.get('yuv_right_duration_s', 'N/A')}s")
                print(f"  RGB Left:   {row.get('rgb_left_duration_s', 'N/A')}s")
                print(f"  RGB Right:  {row.get('rgb_right_duration_s', 'N/A')}s")
                print(f"  Depth Left: {row.get('depth_left_duration_s', 'N/A')}s")
                print(f"  Depth Right:{row.get('depth_right_duration_s', 'N/A')}s")
                print(f"  HMD:        {row.get('hmd_duration_s', 'N/A')}s")
                print(f"\nOverall Duration: {row.get('overall_duration_s', 'N/A')}s")
                print(f"\nTimestamps:")
                print(f"  Overall start: {row.get('overall_start_ms', 'N/A')}")
                print(f"  Overall end:   {row.get('overall_end_ms', 'N/A')}")
                print(f"  HMD start:     {row.get('hmd_start_ms', 'N/A')}")
                print(f"  HMD end:       {row.get('hmd_end_ms', 'N/A')}")
                print(f"  RGB start:     {row.get('rgb_left_start_ms', 'N/A')}")
                print(f"  RGB end:       {row.get('rgb_left_end_ms', 'N/A')}")
                
                # Calculate the issue
                try:
                    overall_start = int(row.get('overall_start_ms', 0))
                    overall_end = int(row.get('overall_end_ms', 0))
                    hmd_start = int(row.get('hmd_start_ms', 0)) if row.get('hmd_start_ms') else None
                    hmd_end = int(row.get('hmd_end_ms', 0)) if row.get('hmd_end_ms') else None
                    rgb_start = int(row.get('rgb_left_start_ms', 0)) if row.get('rgb_left_start_ms') else None
                    rgb_end = int(row.get('rgb_left_end_ms', 0)) if row.get('rgb_left_end_ms') else None
                    
                    if hmd_start and rgb_start:
                        gap = (rgb_start - hmd_start) / 1000.0
                        print(f"\n⚠️  ISSUE DETECTED:")
                        print(f"  RGB starts {gap:.3f}s after HMD starts")
                        if gap > 1.0:
                            print(f"  This creates a large gap that inflates the overall duration!")
                            print(f"  Overall duration = (latest_end - earliest_start)")
                            print(f"                   = ({overall_end} - {overall_start}) / 1000")
                            print(f"                   = {(overall_end - overall_start)/1000:.3f}s")
                            print(f"  But actual recording duration should be ~{row.get('hmd_duration_s', 'N/A')}s (HMD) or ~{row.get('rgb_left_duration_s', 'N/A')}s (RGB)")
                    
                    if rgb_start:
                        print(f"\n✅ Trimming to 17.1s from RGB start:")
                        cutoff = rgb_start + int(17.1 * 1000)
                        print(f"  Cutoff timestamp: {cutoff}")
                        print(f"  This will keep:")
                        if hmd_start:
                            hmd_kept = min(hmd_end or cutoff, cutoff) - max(hmd_start, rgb_start)
                            print(f"    - HMD: {max(0, hmd_kept/1000):.3f}s (from RGB start)")
                        rgb_kept = min(rgb_end or cutoff, cutoff) - rgb_start
                        print(f"    - RGB: {rgb_kept/1000:.3f}s")
                        print(f"  ✓ This should work correctly!")
                except (ValueError, TypeError) as e:
                    print(f"  Could not calculate: {e}")
                
                return
    
    print(f"Session not found in recording_length_report.csv")

if __name__ == "__main__":
    analyze_session(Path("/Volumes/Intenso/Fog/20251209_144306"))
    analyze_session(Path("/Volumes/Intenso/NoFog/20251212_191211"))

