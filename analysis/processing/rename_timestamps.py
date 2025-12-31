#!/usr/bin/env python3
"""
Rename YUV and RGB files with corrected timestamps by applying an offset.

This script renames files in the session directory by subtracting the specified
offset from their timestamp-based filenames. This aligns YUV/RGB recordings with
depth/HMD recordings.

Console Usage Examples:
    # Apply offset to align timestamps (dry-run first)
    python analysis/processing/rename_timestamps.py \
        --session-dir /Volumes/Intenso/Fog/20251209_144306 \
        --offset-ms 180100 \
        --dry-run

    # Actually rename files after verifying with dry-run
    python analysis/processing/rename_timestamps.py \
        --session-dir /Volumes/Intenso/Fog/20251209_144306 \
        --offset-ms 180100

    # Apply offset to multiple session directories
    python analysis/processing/rename_timestamps.py \
        --session-dir /Volumes/Intenso/Fog/20251209_144306 \
        --session-dir /Volumes/Intenso/Fog/20251209_150000 \
        --offset-ms 180100 \
        --dry-run
"""

import argparse
from pathlib import Path
from typing import Optional


def parse_timestamp_from_stem(stem: str) -> Optional[int]:
    """
    Parse integer timestamp from a filename stem, handling macOS sidecar prefixes.
    """
    if stem.startswith("._"):
        stem = stem[2:]
    elif stem.startswith("_"):
        stem = stem.lstrip("_")
    return int(stem) if stem.isdigit() else None


def apply_timestamp_offset(timestamp: int, offset_ms: int) -> int:
    """Apply offset to timestamp (subtract offset)."""
    return timestamp - offset_ms


def rename_files_in_directory(
    dir_path: Path,
    suffix: str,
    offset_ms: int,
    dry_run: bool = True
) -> tuple[int, int]:
    """
    Rename files in directory by applying timestamp offset.
    
    Returns:
        (success_count, error_count) tuple
    """
    if not dir_path.exists():
        return 0, 0
    
    success_count = 0
    error_count = 0
    
    for file_path in sorted(dir_path.glob(f"*{suffix}")):
        original_stem = file_path.stem
        timestamp = parse_timestamp_from_stem(original_stem)
        
        if timestamp is None:
            print(f"  ⚠ Skipping {file_path.name} - cannot parse timestamp")
            error_count += 1
            continue
        
        new_timestamp = apply_timestamp_offset(timestamp, offset_ms)
        new_stem = str(new_timestamp)
        new_name = f"{new_stem}{suffix}"
        new_path = file_path.parent / new_name
        
        if new_path == file_path:
            # No change needed
            continue
        
        if new_path.exists() and not dry_run:
            print(f"  ⚠ Skipping {file_path.name} - target {new_name} already exists")
            error_count += 1
            continue
        
        if dry_run:
            print(f"  Would rename: {file_path.name} → {new_name}")
        else:
            try:
                file_path.rename(new_path)
                print(f"  ✓ Renamed: {file_path.name} → {new_name}")
                success_count += 1
            except Exception as e:
                print(f"  ✗ Error renaming {file_path.name}: {e}")
                error_count += 1
    
    return success_count, error_count


def process_session(
    session_dir: Path,
    offset_ms: int,
    dry_run: bool = True
) -> None:
    """Process a single session directory."""
    print("=" * 80)
    print(f"PROCESSING SESSION: {session_dir.name}")
    print("=" * 80)
    print(f"Session Directory: {session_dir}")
    print(f"Offset Applied: -{abs(offset_ms)}ms ({offset_ms}ms)")
    print(f"Mode: {'DRY RUN (no files will be modified)' if dry_run else 'LIVE (files will be renamed)'}")
    print()
    
    if not session_dir.exists():
        print(f"[Error] Session directory does not exist: {session_dir}")
        return
    
    directories_to_process = [
        ("YUV Left Camera", session_dir / "left_camera_raw", ".yuv"),
        ("YUV Right Camera", session_dir / "right_camera_raw", ".yuv"),
        ("RGB Left Camera", session_dir / "left_camera_rgb", ".png"),
        ("RGB Right Camera", session_dir / "right_camera_rgb", ".png"),
    ]
    
    total_success = 0
    total_errors = 0
    
    for name, dir_path, suffix in directories_to_process:
        print(f"{name} ({dir_path}):")
        success, errors = rename_files_in_directory(dir_path, suffix, offset_ms, dry_run)
        total_success += success
        total_errors += errors
        print()
    
    print("-" * 80)
    if dry_run:
        print(f"DRY RUN SUMMARY:")
        print(f"  Files that would be renamed: {total_success}")
        print(f"  Files with errors: {total_errors}")
        print()
        print("Run without --dry-run to actually rename files")
    else:
        print(f"RENAME SUMMARY:")
        print(f"  Files renamed: {total_success}")
        print(f"  Files with errors: {total_errors}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Rename YUV/RGB files with corrected timestamps"
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        action="append",
        required=True,
        help="Session directory to process (can be specified multiple times)",
    )
    parser.add_argument(
        "--offset-ms",
        type=int,
        required=True,
        help="Offset in milliseconds to subtract from YUV/RGB timestamps",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be renamed without actually renaming (default: True)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Actually rename files (overrides --dry-run)",
    )
    
    args = parser.parse_args()
    
    dry_run = args.dry_run and not args.force
    
    if not dry_run:
        response = input(
            f"\n⚠ WARNING: This will rename files in {len(args.session_dir)} session(s).\n"
            f"Offset: {args.offset_ms}ms\n"
            f"Are you sure you want to proceed? (yes/no): "
        )
        if response.lower() != "yes":
            print("Cancelled.")
            return
    
    for session_dir in args.session_dir:
        process_session(session_dir, args.offset_ms, dry_run)
        print()


if __name__ == "__main__":
    main()

