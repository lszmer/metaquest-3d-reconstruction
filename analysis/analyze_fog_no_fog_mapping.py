#!/usr/bin/env python3
"""
Build a consolidated fog/no_fog experiment report.

- Reads the mapping CSV (Name, NoFog, Fog).
- Matches sessions to `/Volumes/Intenso/{NoFog,Fog}/<session>`.
- Extracts the best-available reconstruction completion timestamp.
- Captures total/adjusted runtimes (if present in pipeline_runtime.txt).
- Adds placeholders for `evaluate_fbx_mesh` output and notes mesh presence.
- Writes a flat CSV (and optional XLSX) for downstream analysis.
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Defaults are absolute to avoid ambiguity.
DEFAULT_MAPPING = Path(
    "/Users/linus/Library/CloudStorage/GoogleDrive-linus.zimmer@cdtm.com/.shortcut-targets-by-id/1cO8BYGH4cISVBx0VR7aQZerZKRpnQ7QW/CHL_SnappingGrid/Analysis/mapping_fognofog_folders.csv"
)
DEFAULT_ROOT = Path("/Volumes/Intenso")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "master_fog_no_fog_report.csv"

FIELDNAMES = [
    "participant",
    "condition",
    "session_id",
    "paired_session_id",
    "paired_condition",
    "session_dir",
    "session_dir_exists",
    "pipeline_runtime_path",
    "completion_time_utc",
    "completion_time_source",
    "completion_fallback_artifact",
    "runtime_total_seconds",
    "runtime_adjusted_seconds",
    "runtime_secs_per_capture",
    "runtime_source",
    "color_mesh_fbx_path",
    "color_mesh_ply_path",
    "color_mesh_present",
    "evaluate_fbx_mesh_command_placeholder",
    "evaluate_report_path_placeholder",
    "evaluate_quality_score_placeholder",
    "notes",
]


def read_mapping(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Name", "NoFog", "Fog"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapping CSV missing required columns: {missing}")
        return [row for row in reader]


def _parse_pipeline_runtime(runtime_path: Path) -> Dict[str, Optional[str]]:
    """
    Parse pipeline_runtime.txt for timestamps and runtimes.
    Returns keys: completion_time_utc, start_time_utc, runtime_total_seconds,
    runtime_adjusted_seconds, runtime_secs_per_capture, runtime_source.
    """
    result: Dict[str, Optional[str]] = {
        "completion_time_utc": None,
        "start_time_utc": None,
        "runtime_total_seconds": None,
        "runtime_adjusted_seconds": None,
        "runtime_secs_per_capture": None,
        "runtime_source": None,
    }
    if not runtime_path.exists():
        return result
    
    for line in runtime_path.read_text(encoding="utf-8").splitlines():
        normalized = line.strip().lower()
        if normalized.startswith("completion timestamp (utc):"):
            result["completion_time_utc"] = line.split(":", 1)[1].strip()
            result["runtime_source"] = "pipeline_runtime_timestamp"
        elif normalized.startswith("start timestamp (utc):"):
            result["start_time_utc"] = line.split(":", 1)[1].strip()
        elif normalized.startswith("total runtime (s):"):
            result["runtime_total_seconds"] = line.split(":", 1)[1].strip()
            result["runtime_source"] = result["runtime_source"] or "pipeline_runtime_fields"
        elif normalized.startswith("adjusted runtime (s):"):
            result["runtime_adjusted_seconds"] = line.split(":", 1)[1].strip()
            result["runtime_source"] = result["runtime_source"] or "pipeline_runtime_fields"
        elif normalized.startswith("seconds per capture (adjusted):"):
            result["runtime_secs_per_capture"] = line.split(":", 1)[1].strip()
            result["runtime_source"] = result["runtime_source"] or "pipeline_runtime_fields"
    
    return result


def find_completion_timestamp(session_dir: Path) -> Tuple[Optional[str], str, Dict[str, Optional[str]], Optional[str]]:
    """
    Return:
    - completion timestamp (UTC ISO string or None)
    - source string
    - parsed runtime info dict
    - fallback artifact label (if mtime used)
    Priority:
    1) Parsed timestamp inside pipeline_runtime.txt (new format).
    2) Latest mtime among pipeline_runtime.txt and key reconstruction artifacts.
    """
    runtime_path = session_dir / "pipeline_runtime.txt"
    runtime_info = _parse_pipeline_runtime(runtime_path)
    if runtime_info.get("completion_time_utc"):
        return runtime_info["completion_time_utc"], "pipeline_runtime_timestamp", runtime_info, None
    
    candidates: List[Tuple[float, str]] = []
    candidate_paths = [
        ("pipeline_runtime.txt", runtime_path),
        ("reconstruction/color_mesh.fbx", session_dir / "reconstruction" / "color_mesh.fbx"),
        ("reconstruction/color_mesh.ply", session_dir / "reconstruction" / "color_mesh.ply"),
        ("reconstruction/mesh/color_mesh.ply", session_dir / "reconstruction" / "mesh" / "color_mesh.ply"),
        ("reconstruction/mesh", session_dir / "reconstruction" / "mesh"),
        ("reconstruction/tsdf", session_dir / "reconstruction" / "tsdf"),
    ]
    for label, path in candidate_paths:
        if path.exists():
            try:
                mtime = path.stat().st_mtime
                candidates.append((mtime, label))
            except OSError:
                continue
    
    if not candidates:
        return None, "", runtime_info, None
    
    mtime, label = max(candidates, key=lambda x: x[0])
    completion_iso = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
    return completion_iso, f"mtime:{label}", runtime_info, label


def build_rows(mapping_rows: List[Dict[str, str]], root_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    
    for entry in mapping_rows:
        participant = entry["Name"].strip()
        sessions = {"NoFog": entry["NoFog"].strip(), "Fog": entry["Fog"].strip()}
        for condition, session_id in sessions.items():
            paired_condition = "Fog" if condition == "NoFog" else "NoFog"
            paired_session_id = sessions[paired_condition]
            session_dir = root_dir / condition / session_id
            exists = session_dir.exists()
            
            completion_ts, completion_source, runtime_info, fallback_label = (None, "", {}, None)
            if exists:
                completion_ts, completion_source, runtime_info, fallback_label = find_completion_timestamp(session_dir)
            
            runtime_path = session_dir / "pipeline_runtime.txt"
            color_mesh_fbx = session_dir / "reconstruction" / "color_mesh.fbx"
            color_mesh_ply = session_dir / "reconstruction" / "color_mesh.ply"
            color_mesh_present = color_mesh_fbx.exists() or color_mesh_ply.exists()
            
            eval_cmd = ""
            if color_mesh_fbx.exists():
                eval_cmd = f"python scripts/evaluate_fbx_mesh.py \"{color_mesh_fbx}\""
            elif color_mesh_ply.exists():
                eval_cmd = f"python scripts/evaluate_fbx_mesh.py \"{color_mesh_ply}\""
            
            rows.append(
                {
                    "participant": participant,
                    "condition": condition,
                    "session_id": session_id,
                    "paired_session_id": paired_session_id,
                    "paired_condition": paired_condition,
                    "session_dir": str(session_dir),
                    "session_dir_exists": str(exists),
                    "pipeline_runtime_path": str(runtime_path) if runtime_path.exists() else "",
                    "completion_time_utc": completion_ts or "",
                    "completion_time_source": completion_source,
                    "completion_fallback_artifact": fallback_label or "",
                    "runtime_total_seconds": runtime_info.get("runtime_total_seconds") or "",
                    "runtime_adjusted_seconds": runtime_info.get("runtime_adjusted_seconds") or "",
                    "runtime_secs_per_capture": runtime_info.get("runtime_secs_per_capture") or "",
                    "runtime_source": runtime_info.get("runtime_source") or "",
                    "color_mesh_fbx_path": str(color_mesh_fbx) if color_mesh_fbx.exists() else "",
                    "color_mesh_ply_path": str(color_mesh_ply) if color_mesh_ply.exists() else "",
                    "color_mesh_present": str(color_mesh_present),
                    "evaluate_fbx_mesh_command_placeholder": eval_cmd,
                    "evaluate_report_path_placeholder": "",
                    "evaluate_quality_score_placeholder": "",
                    "notes": "",
                }
            )
    
    return rows


def write_csv(rows: List[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Info] Wrote CSV: {output_path} ({len(rows)} rows)")


def maybe_write_excel(rows: List[Dict[str, str]], xlsx_path: Optional[Path]) -> None:
    if xlsx_path is None:
        return
    try:
        import pandas as pd  # type: ignore
    except Exception:
        print("[Warning] pandas not installed; skipping Excel output.")
        return
    xlsx_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_excel(xlsx_path, index=False)
    print(f"[Info] Wrote Excel: {xlsx_path} ({len(rows)} rows)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a fog/no_fog master report with reconstruction timestamps, runtimes, and evaluation placeholders."
    )
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="Path to mapping_fognofog_folders.csv")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Root directory containing Fog/ and NoFog/")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--xlsx-output", type=Path, help="Optional Excel output path (.xlsx). Requires pandas.")
    return parser.parse_args()


def main():
    args = parse_args()
    
    mapping_rows = read_mapping(args.mapping)
    rows = build_rows(mapping_rows, args.root)
    
    write_csv(rows, args.output)
    maybe_write_excel(rows, args.xlsx_output)
    
    missing = [row for row in rows if row["session_dir_exists"] == "False"]
    if missing:
        print(f"[Warning] {len(missing)} session(s) listed in mapping were not found under {args.root}.")
        for row in missing:
            print(f"  - {row['condition']} missing: {row['session_dir']}")
    else:
        print("[Info] All mapped sessions were found on disk.")


if __name__ == "__main__":
    main()

