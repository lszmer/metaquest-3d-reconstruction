#!/usr/bin/env python3
"""
Fix pairwise matching in pairwise_summary.csv by correctly pairing fog and nofog meshes
based on participant and session IDs from quality_scores.csv.
"""

import pandas as pd
from pathlib import Path
import argparse


def extract_participant_pair_id(name: str, master_report_df: pd.DataFrame = None) -> tuple[str, str]:
    """
    Extract participant and pair identifier from mesh name.
    
    Names follow pattern: {participant}_{nofog_session}__{fog_session}_{condition}
    Returns: (participant, pair_id)
    """
    if name.endswith("_fog"):
        base = name[:-4]
    elif name.endswith("_nofog"):
        base = name[:-6]
    else:
        return None, None
    
    # Split on double underscore
    parts = base.split("__")
    if len(parts) != 2:
        return None, None
    
    nofog_part = parts[0]  # {participant}_{nofog_session}
    fog_session = parts[1]  # {fog_session}
    
    # Extract nofog_session (last 15 characters: YYYYMMDD_HHMMSS)
    # Find the last underscore before the date pattern
    nofog_session = nofog_part[-15:] if len(nofog_part) >= 15 else None
    if nofog_session and len(nofog_session.split("_")) == 2:
        participant = nofog_part[:-16] if len(nofog_part) > 15 else nofog_part  # Remove trailing underscore
    else:
        # Fallback: find last underscore
        last_underscore = nofog_part.rfind("_")
        if last_underscore == -1:
            return None, None
        participant = nofog_part[:last_underscore]
        nofog_session = nofog_part[last_underscore + 1:]
    
    pair_id = f"{nofog_session}__{fog_session}"
    
    # If we have master report, try to match participant name more accurately
    if master_report_df is not None:
        # Try to find matching participant by pair_id
        matching_rows = master_report_df[master_report_df["pair_id"] == pair_id]
        if len(matching_rows) > 0:
            participant = matching_rows.iloc[0]["participant"]
    
    return participant, pair_id


def fix_pairwise_matching(
    quality_scores_csv: Path,
    pairwise_summary_csv: Path,
    output_csv: Path,
    master_report_csv: Path = None
):
    """Fix pairwise matching by correctly pairing fog and nofog meshes."""
    
    # Load master report if available for accurate participant names
    master_report_df = None
    if master_report_csv and master_report_csv.exists():
        print(f"Loading master report from {master_report_csv}...")
        master_report_df = pd.read_csv(master_report_csv)
    
    # Load quality scores
    print(f"Loading quality scores from {quality_scores_csv}...")
    quality_df = pd.read_csv(quality_scores_csv)
    
    # Extract participant and pair_id for each mesh
    quality_df["participant"] = quality_df["name"].apply(
        lambda x: extract_participant_pair_id(x, master_report_df)[0] if extract_participant_pair_id(x, master_report_df)[0] else None
    )
    quality_df["pair_id"] = quality_df["name"].apply(
        lambda x: extract_participant_pair_id(x, master_report_df)[1] if extract_participant_pair_id(x, master_report_df)[1] else None
    )
    quality_df["condition"] = quality_df["name"].apply(
        lambda x: "Fog" if x.endswith("_fog") else ("NoFog" if x.endswith("_nofog") else None)
    )
    
    # Filter out rows where we couldn't extract info
    quality_df = quality_df[
        quality_df["participant"].notna() & 
        quality_df["pair_id"].notna() & 
        quality_df["condition"].notna()
    ].copy()
    
    print(f"Found {len(quality_df)} valid mesh records")
    print(f"Unique participants: {quality_df['participant'].nunique()}")
    print(f"Unique pairs: {quality_df['pair_id'].nunique()}")
    
    # Group by participant and pair_id to find matching pairs
    pairs = []
    for (participant, pair_id), group in quality_df.groupby(["participant", "pair_id"]):
        fog_row = group[group["condition"] == "Fog"]
        nofog_row = group[group["condition"] == "NoFog"]
        
        if len(fog_row) == 0:
            print(f"Warning: No fog mesh found for {participant} ({pair_id})")
            continue
        if len(nofog_row) == 0:
            print(f"Warning: No nofog mesh found for {participant} ({pair_id})")
            continue
        
        fog_full_name = fog_row.iloc[0]["name"]
        nofog_full_name = nofog_row.iloc[0]["name"]
        fog_q_norm = fog_row.iloc[0]["Q_norm"]
        nofog_q_norm = nofog_row.iloc[0]["Q_norm"]
        delta = nofog_q_norm - fog_q_norm

        # Extract just the date ID from the mesh name
        # Pattern: {participant}_{nofog_session}__{fog_session}_{condition}
        # We want: fog_session for fog_name, nofog_session for nofog_name

        # Split fog name: e.g., "Lena Ehrenreich_20251209_142927__20251209_144306_fog"
        fog_base = fog_full_name[:-4] if fog_full_name.endswith("_fog") else fog_full_name
        fog_parts = fog_base.split("__")
        if len(fog_parts) == 2:
            fog_session = fog_parts[1]  # "20251209_144306"
        else:
            fog_session = "unknown"

        # Split nofog name: e.g., "Lena Ehrenreich_20251209_142927__20251209_144306_nofog"
        nofog_base = nofog_full_name[:-6] if nofog_full_name.endswith("_nofog") else nofog_full_name
        nofog_parts = nofog_base.split("__")
        if len(nofog_parts) == 2:
            nofog_part = nofog_parts[0]  # "Lena Ehrenreich_20251209_142927"
            # Extract just the date ID (last 15 characters)
            nofog_session = nofog_part[-15:] if len(nofog_part) >= 15 and "_" in nofog_part[-15:] else "unknown"
        else:
            nofog_session = "unknown"

        pairs.append({
            "participant": participant,
            "pair_id": pair_id,
            "fog_name": fog_session,
            "fog_Q_norm": f"{fog_q_norm:.6f}",
            "nofog_name": nofog_session,
            "nofog_Q_norm": f"{nofog_q_norm:.6f}",
            "delta_nofog_minus_fog": f"{delta:.6f}",
            "delta_fog_minus_nofog": f"{-delta:.6f}"
        })
    
    # Create DataFrame and sort by participant
    fixed_df = pd.DataFrame(pairs)
    fixed_df = fixed_df.sort_values("participant").reset_index(drop=True)
    
    print(f"\nFound {len(fixed_df)} correctly matched pairs")
    
    # Save fixed CSV
    fixed_df.to_csv(output_csv, index=False)
    print(f"Fixed pairwise summary saved to: {output_csv}")
    
    # Print comparison
    if pairwise_summary_csv.exists():
        print("\nComparing with original pairwise_summary.csv...")
        original_df = pd.read_csv(pairwise_summary_csv)
        print(f"Original pairs: {len(original_df)}")
        print(f"Fixed pairs: {len(fixed_df)}")
        
        # Check for mismatches
        mismatches = 0
        for idx, row in fixed_df.iterrows():
            if idx < len(original_df):
                orig_row = original_df.iloc[idx]
                if (row["fog_name"] != orig_row["fog_name"] or 
                    row["nofog_name"] != orig_row["nofog_name"]):
                    mismatches += 1
                    if mismatches <= 3:  # Show first 3 mismatches
                        print(f"\nMismatch {mismatches}:")
                        print(f"  Participant: {row['participant']}")
                        print(f"  Original fog: {orig_row['fog_name']}")
                        print(f"  Fixed fog:    {row['fog_name']}")
                        print(f"  Original nofog: {orig_row['nofog_name']}")
                        print(f"  Fixed nofog:    {row['nofog_name']}")
        
        if mismatches > 0:
            print(f"\nTotal mismatches found: {mismatches}")
        else:
            print("No mismatches found (but order may differ)")
    
    return fixed_df


def main():
    parser = argparse.ArgumentParser(
        description="Fix pairwise matching in pairwise_summary.csv"
    )
    parser.add_argument(
        "--quality-scores",
        type=Path,
        default=Path("analysis/mesh_quality_batch/quality_scores.csv"),
        help="Path to quality_scores.csv"
    )
    parser.add_argument(
        "--pairwise-summary",
        type=Path,
        default=Path("analysis/mesh_quality_batch/pairwise_summary.csv"),
        help="Path to pairwise_summary.csv (for comparison)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/mesh_quality_batch/pairwise_summary_fixed.csv"),
        help="Output path for fixed pairwise summary"
    )
    parser.add_argument(
        "--master-report",
        type=Path,
        default=Path("analysis/master_fog_no_fog_report.csv"),
        help="Path to master_fog_no_fog_report.csv for accurate participant names"
    )
    
    args = parser.parse_args()
    
    fixed_df = fix_pairwise_matching(
        args.quality_scores,
        args.pairwise_summary,
        args.output,
        args.master_report
    )
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(fixed_df.to_string(index=False))


if __name__ == "__main__":
    main()

