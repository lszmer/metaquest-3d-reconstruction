#!/usr/bin/env python3
"""
Test script to verify the quality metrics are added to master_fog_no_fog_report.csv
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional, Any, NamedTuple
from dataclasses import dataclass, asdict

# Mock the QualityScores class
@dataclass
class MockQualityScores:
    name: str
    Q_raw: float
    Q_norm: float
    S_geom: float
    S_smooth: float
    S_complete: float
    S_color: float
    S_shape: float
    S_topology: float
    S_bonuses: float

def update_master_fog_report_test(master_csv: Path) -> None:
    """Test version of the update function"""
    print(f"[info] Testing update of: {master_csv}")

    # Read existing CSV
    with master_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = list(reader)
        fieldnames = list(reader.fieldnames or [])

    # Mock some quality scores
    mock_scores = [
        MockQualityScores("Lena Ehrenreich_20251209_142927__20251209_144306_fog", 0.7, 0.284691, 0.6, 0.5, 0.8, 0.9, 0.4, 0.3, 0.2),
        MockQualityScores("Lena Ehrenreich_20251209_142927__20251209_144306_nofog", 0.8, 0.393964, 0.7, 0.6, 0.9, 0.8, 0.5, 0.4, 0.3)
    ]

    # Mock pair metadata
    pair_meta = [{
        "participant": "Lena Ehrenreich",
        "pair_id": "20251209_142927__20251209_144306",
        "fog_name": "Lena Ehrenreich_20251209_142927__20251209_144306_fog",
        "nofog_name": "Lena Ehrenreich_20251209_142927__20251209_144306_nofog"
    }]

    # Fast lookup for scores and pair metadata
    scores_by_name: Dict[str, MockQualityScores] = {s.name: s for s in mock_scores}
    meta_index: Dict[Tuple[str, str], Dict[str, str]] = {}
    for m in pair_meta:
        participant = (m.get("participant") or "").strip()
        pair_id = (m.get("pair_id") or "").strip()
        if participant and pair_id:
            meta_index[(participant, pair_id)] = m

    # Define all the quality metric columns we want to add
    quality_columns = [
        "Q_raw", "Q_norm",  # Overall scores
        "S_geom", "S_smooth", "S_complete", "S_color",  # Main sub-scores
        "S_shape", "S_topology", "S_bonuses"  # Detailed sub-scores
    ]

    # Add columns for both fog and nofog conditions
    new_columns = []
    for condition in ["fog", "nofog"]:
        for col in quality_columns:
            new_col = f"{condition}_{col}"
            if new_col not in fieldnames:
                fieldnames.append(new_col)
                new_columns.append(new_col)

    delta_col = "relative_quality_delta_nofog_minus_fog"
    if delta_col not in fieldnames:
        fieldnames.append(delta_col)
        new_columns.append(delta_col)

    print(f"[info] Adding {len(new_columns)} new columns: {new_columns}")

    for row in rows:
        participant = (row.get("participant") or "").strip()
        pair_id = (row.get("pair_id") or "").strip()
        meta = meta_index.get((participant, pair_id))
        if not meta:
            continue

        fog_name = meta.get("fog_name") or ""
        nofog_name = meta.get("nofog_name") or ""
        fog_score = scores_by_name.get(fog_name)
        nofog_score = scores_by_name.get(nofog_name)

        # Only update if we have both sides evaluated so far
        if fog_score is None or nofog_score is None:
            continue

        print(f"[info] Updating row for participant: {participant}")

        # Fill comprehensive quality metrics for fog condition
        row["fog_Q_raw"] = f"{fog_score.Q_raw:.6f}"
        row["fog_Q_norm"] = f"{fog_score.Q_norm:.6f}"
        row["fog_S_geom"] = f"{fog_score.S_geom:.6f}"
        row["fog_S_smooth"] = f"{fog_score.S_smooth:.6f}"
        row["fog_S_complete"] = f"{fog_score.S_complete:.6f}"
        row["fog_S_color"] = f"{fog_score.S_color:.6f}"
        row["fog_S_shape"] = f"{fog_score.S_shape:.6f}"
        row["fog_S_topology"] = f"{fog_score.S_topology:.6f}"
        row["fog_S_bonuses"] = f"{fog_score.S_bonuses:.6f}"

        # Fill comprehensive quality metrics for nofog condition
        row["nofog_Q_raw"] = f"{nofog_score.Q_raw:.6f}"
        row["nofog_Q_norm"] = f"{nofog_score.Q_norm:.6f}"
        row["nofog_S_geom"] = f"{nofog_score.S_geom:.6f}"
        row["nofog_S_smooth"] = f"{nofog_score.S_smooth:.6f}"
        row["nofog_S_complete"] = f"{nofog_score.S_complete:.6f}"
        row["nofog_S_color"] = f"{nofog_score.S_color:.6f}"
        row["nofog_S_shape"] = f"{nofog_score.S_shape:.6f}"
        row["nofog_S_topology"] = f"{nofog_score.S_topology:.6f}"
        row["nofog_S_bonuses"] = f"{nofog_score.S_bonuses:.6f}"

        # Fill existing placeholder columns with the normalized scores (backward compatibility)
        if "fog_evaluate_quality_score_placeholder" in row:
            row["fog_evaluate_quality_score_placeholder"] = f"{fog_score.Q_norm:.6f}"
        if "nofog_evaluate_quality_score_placeholder" in row:
            row["nofog_evaluate_quality_score_placeholder"] = f"{nofog_score.Q_norm:.6f}"

        delta = nofog_score.Q_norm - fog_score.Q_norm
        row[delta_col] = f"{delta:.6f}"

    # Save updated CSV
    with master_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[info] Updated master report saved to: {master_csv}")

    # Show new columns
    print("\nNew quality metric columns added:")
    quality_cols = [col for col in fieldnames if any(col.startswith(prefix) for prefix in ["fog_Q", "fog_S", "nofog_Q", "nofog_S"])]
    for col in quality_cols:
        print(f"  - {col}")

if __name__ == "__main__":
    master_csv = Path("analysis/master_fog_no_fog_report.csv")
    update_master_fog_report_test(master_csv)
