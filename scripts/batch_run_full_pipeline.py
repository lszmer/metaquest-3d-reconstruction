#!/usr/bin/env python3
"""
Batch runner that orchestrates the full 3D reconstruction pipeline over
multiple project/session directories, sitting on top of `run_full_pipeline.py`.

Typical use case:
    - You download multiple Quest capture sessions into folders like
      `/Volumes/Intenso/Fog` and `/Volumes/Intenso/NoFog`.
    - Each immediate subdirectory inside those folders is a project/session dir.
    - This script iterates over all such subdirectories and runs the full
      pipeline for each, one after another.

Example:
    python scripts/batch_run_full_pipeline.py \
        --base-dir /Volumes/Intenso/Fog \
        --base-dir /Volumes/Intenso/NoFog \
        --config config/pipeline_config.yml \
        --skip-fbx
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List
import time


def discover_project_dirs(base_dirs: List[Path]) -> List[Path]:
    """Return a sorted list of all immediate subdirectories of the given bases."""
    project_dirs: List[Path] = []
    for base in base_dirs:
        if not base.exists():
            print(f"[Warning] Base directory does not exist, skipping: {base}")
            continue
        if not base.is_dir():
            print(f"[Warning] Base path is not a directory, skipping: {base}")
            continue

        for child in base.iterdir():
            if child.is_dir():
                project_dirs.append(child.resolve())

    # Sort for deterministic order
    project_dirs = sorted(set(project_dirs))
    return project_dirs


def run_single_project(
    run_script: Path,
    session_dir: Path,
    config: Path | None,
    skip_fbx: bool,
) -> int:
    """
    Run `run_full_pipeline.py` once for a given session/project directory.

    Returns the subprocess return code.
    """
    cmd = [sys.executable, str(run_script), "--session_dir", str(session_dir)]

    if config is not None:
        cmd.extend(["--config", str(config)])

    if skip_fbx:
        cmd.append("--skip-fbx")

    print("\n" + "=" * 80)
    print(f"[Batch] Running pipeline for: {session_dir}")
    print(f"[Batch] Command: {' '.join(cmd)}")
    print("=" * 80)

    start_ts = time.time()
    try:
        proc = subprocess.run(cmd, check=False)
        rc = proc.returncode
    except Exception as e:  # pragma: no cover - defensive
        print(f"[Batch][Error] Failed to launch pipeline for {session_dir}: {e}")
        return 1
    finally:
        elapsed = time.time() - start_ts
        print(f"[Batch] Elapsed time for {session_dir}: {elapsed:.2f} s")

    if rc != 0:
        print(f"[Batch][Error] Pipeline failed for {session_dir} with code {rc}")
    else:
        print(f"[Batch] Pipeline completed successfully for {session_dir}")

    return rc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-run the full 3D reconstruction pipeline over multiple "
            "project/session directories, using scripts/run_full_pipeline.py."
        )
    )

    parser.add_argument(
        "--base-dir",
        "-b",
        type=Path,
        action="append",
        help=(
            "Base directory containing multiple project/session subdirectories. "
            "Can be specified multiple times. If omitted, defaults to "
            "/Volumes/Intenso/Fog and /Volumes/Intenso/NoFog."
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to pipeline config YAML to forward to "
            "run_full_pipeline.py. If omitted, that script will use its default."
        ),
    )
    parser.add_argument(
        "--skip-fbx",
        action="store_true",
        help="Forward --skip-fbx to run_full_pipeline.py for each project.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the discovered project/session dirs, do not run anything.",
    )

    args = parser.parse_args()

    # Determine base directories
    if args.base_dir:
        base_dirs = [p.resolve() for p in args.base_dir]
    else:
        base_dirs = [
            Path("/Volumes/Intenso/Fog").resolve(),
            Path("/Volumes/Intenso/NoFog").resolve(),
        ]

    print("[Batch] Base directories:")
    for b in base_dirs:
        print(f"  - {b}")

    # Discover projects
    project_dirs = discover_project_dirs(base_dirs)
    if not project_dirs:
        print("[Batch][Error] No project/session directories found under the given bases.")
        sys.exit(1)

    print("\n[Batch] Discovered project/session directories:")
    for p in project_dirs:
        print(f"  - {p}")

    if args.dry_run:
        print("\n[Batch] Dry run requested. Exiting without running pipelines.")
        return

    # Resolve run_full_pipeline.py path (same directory as this script)
    script_dir = Path(__file__).resolve().parent
    run_script = script_dir / "run_full_pipeline.py"
    if not run_script.exists():
        print(f"[Batch][Error] Could not find run_full_pipeline.py at: {run_script}")
        sys.exit(1)

    # Optional config path normalization
    config_path: Path | None = args.config.resolve() if args.config is not None else None

    print(f"\n[Batch] Using run_full_pipeline script: {run_script}")
    if config_path is not None:
        print(f"[Batch] Forwarding config: {config_path}")
    print(f"[Batch] skip_fbx = {args.skip_fbx}")

    # Run all projects sequentially and collect results
    failures: list[tuple[Path, int]] = []
    overall_start = time.time()

    for idx, proj in enumerate(project_dirs, start=1):
        print("\n" + "#" * 80)
        print(f"[Batch] [{idx}/{len(project_dirs)}] Starting: {proj}")
        print("#" * 80)

        rc = run_single_project(
            run_script=run_script,
            session_dir=proj,
            config=config_path,
            skip_fbx=args.skip_fbx,
        )
        if rc != 0:
            failures.append((proj, rc))

    overall_elapsed = time.time() - overall_start

    print("\n" + "=" * 80)
    print("[Batch] BATCH PIPELINE RUN COMPLETE")
    print("=" * 80)
    print(f"[Batch] Total elapsed batch time: {overall_elapsed:.2f} s")
    print(f"[Batch] Total projects processed: {len(project_dirs)}")
    print(f"[Batch] Successful: {len(project_dirs) - len(failures)}")
    print(f"[Batch] Failed: {len(failures)}")

    if failures:
        print("\n[Batch] Failures:")
        for proj, rc in failures:
            print(f"  - {proj} (exit code {rc})")
        sys.exit(1)


if __name__ == "__main__":
    main()


