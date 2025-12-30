#!/usr/bin/env python3
"""
Generate simple pairwise HTML report focusing on basic geometric metrics.
"""

import pandas as pd
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def extract_participant_pair_id(name: str) -> tuple:
    """Extract participant and pair_id from mesh name."""
    if name.endswith("_fog"):
        base = name[:-4]
    elif name.endswith("_nofog"):
        base = name[:-6]
    else:
        return None, None
    
    parts = base.split("__")
    if len(parts) != 2:
        return None, None
    
    nofog_part = parts[0]
    fog_session = parts[1]
    
    # Extract participant (everything before last underscore in nofog_part)
    last_underscore = nofog_part.rfind("_")
    if last_underscore == -1:
        return None, None
    
    participant = nofog_part[:last_underscore]
    nofog_session = nofog_part[last_underscore + 1:]
    pair_id = f"{nofog_session}__{fog_session}"
    
    return participant, pair_id


def generate_simple_pairwise_report(
    quality_scores_csv: Path,
    output_html: Path,
    master_report_csv: Path = None
):
    """Generate simple pairwise HTML report."""
    
    print(f"Loading quality scores from {quality_scores_csv}...")
    df = pd.read_csv(quality_scores_csv)
    
    # Extract participant and pair_id
    df["participant"] = df["name"].apply(lambda x: extract_participant_pair_id(x)[0])
    df["pair_id"] = df["name"].apply(lambda x: extract_participant_pair_id(x)[1])
    df["condition"] = df["name"].apply(
        lambda x: "Fog" if x.endswith("_fog") else ("NoFog" if x.endswith("_nofog") else None)
    )
    
    # Use master report for correct participant names if available
    if master_report_csv and master_report_csv.exists():
        print(f"Loading master report from {master_report_csv}...")
        master_df = pd.read_csv(master_report_csv)
        # Create mapping from pair_id to participant
        pair_to_participant = dict(zip(master_df["pair_id"], master_df["participant"]))
        df["participant"] = df["pair_id"].map(pair_to_participant).fillna(df["participant"])
    
    # Filter valid rows
    df = df[df["participant"].notna() & df["pair_id"].notna() & df["condition"].notna()].copy()
    
    # Simple metrics to display
    simple_metrics = {
        "num_vertices": "Vertex Count",
        "num_triangles": "Triangle Count",
        "component_count": "Component Count",
        "boundary_edge_ratio": "Boundary Edge Ratio",
        "degenerate_triangles": "Degenerate Triangles",
    }
    
    # Create pairs
    pairs = []
    for (participant, pair_id), group in df.groupby(["participant", "pair_id"]):
        fog_row = group[group["condition"] == "Fog"]
        nofog_row = group[group["condition"] == "NoFog"]
        
        if len(fog_row) == 0 or len(nofog_row) == 0:
            continue
        
        fog_data = fog_row.iloc[0]
        nofog_data = nofog_row.iloc[0]
        
        # Extract just the date ID from the mesh name
        # Pattern: {participant}_{nofog_session}__{fog_session}_{condition}
        # We want: fog_session for fog_name, nofog_session for nofog_name

        # Split fog name: e.g., "Lena Ehrenreich_20251209_142927__20251209_144306_fog"
        fog_base = fog_data["name"][:-4] if fog_data["name"].endswith("_fog") else fog_data["name"]
        fog_parts = fog_base.split("__")
        if len(fog_parts) == 2:
            fog_session = fog_parts[1]  # "20251209_144306"
        else:
            fog_session = "unknown"

        # Split nofog name: e.g., "Lena Ehrenreich_20251209_142927__20251209_144306_nofog"
        nofog_base = nofog_data["name"][:-6] if nofog_data["name"].endswith("_nofog") else nofog_data["name"]
        nofog_parts = nofog_base.split("__")
        if len(nofog_parts) == 2:
            nofog_part = nofog_parts[0]  # "Lena Ehrenreich_20251209_142927"
            # Extract just the date ID (last 15 characters)
            nofog_session = nofog_part[-15:] if len(nofog_part) >= 15 and "_" in nofog_part[-15:] else "unknown"
        else:
            nofog_session = "unknown"

        pair_info = {
            "participant": participant,
            "pair_id": pair_id,
            "fog_name": fog_session,
            "nofog_name": nofog_session,
        }
        
        # Add simple metrics
        for metric_col, display_name in simple_metrics.items():
            if metric_col in df.columns:
                pair_info[f"fog_{metric_col}"] = fog_data[metric_col]
                pair_info[f"nofog_{metric_col}"] = nofog_data[metric_col]
                pair_info[f"delta_{metric_col}"] = nofog_data[metric_col] - fog_data[metric_col]
        
        pairs.append(pair_info)
    
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values("participant").reset_index(drop=True)
    
    print(f"Found {len(pairs_df)} pairs")
    
    # Generate plots for each metric
    pngs = {}
    
    for metric_col, display_name in simple_metrics.items():
        if f"fog_{metric_col}" not in pairs_df.columns:
            continue
        
        fog_vals = pairs_df[f"fog_{metric_col}"].values
        nofog_vals = pairs_df[f"nofog_{metric_col}"].values
        deltas = pairs_df[f"delta_{metric_col}"].values
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(pairs_df))
        ax.bar([i - 0.2 for i in x], fog_vals, width=0.4, label="Fog", alpha=0.7)
        ax.bar([i + 0.2 for i in x], nofog_vals, width=0.4, label="NoFog", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs_df["participant"], rotation=45, ha="right")
        ax.set_ylabel(display_name)
        ax.set_title(f"{display_name} per pair")
        ax.legend()
        pngs[f"{display_name}_bars"] = _fig_to_base64(fig)
        
        # Delta plot
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ["green" if d >= 0 else "red" for d in deltas]
        ax.bar(x, deltas, color=colors, alpha=0.7)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs_df["participant"], rotation=45, ha="right")
        ax.set_ylabel(f"Delta (NoFog - Fog)")
        ax.set_title(f"{display_name} Delta per pair")
        pngs[f"{display_name}_delta"] = _fig_to_base64(fig)
    
    # Generate HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>Simple Mesh Metrics - Pairwise Comparison</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        "h1, h2 { color: #333; }",
        ".metric-section { margin: 30px 0; }",
        "</style>",
        "</head><body>",
        "<h1>Simple Mesh Metrics - Pairwise Comparison</h1>",
        "<h2>Fog vs NoFog</h2>",
    ]
    
    # Summary table
    html_parts.append("<h2>Summary Table</h2>")
    html_parts.append("<table>")
    html_parts.append("<tr><th>Participant</th><th>Pair ID</th>")
    for metric_col, display_name in simple_metrics.items():
        if f"fog_{metric_col}" in pairs_df.columns:
            html_parts.append(f"<th>Fog {display_name}</th>")
            html_parts.append(f"<th>NoFog {display_name}</th>")
            html_parts.append(f"<th>Delta</th>")
    html_parts.append("</tr>")
    
    for _, row in pairs_df.iterrows():
        html_parts.append(f"<tr><td>{row['participant']}</td><td>{row['pair_id']}</td>")
        for metric_col, display_name in simple_metrics.items():
            if f"fog_{metric_col}" in pairs_df.columns:
                fog_val = row[f"fog_{metric_col}"]
                nofog_val = row[f"nofog_{metric_col}"]
                delta = row[f"delta_{metric_col}"]
                html_parts.append(f"<td>{fog_val:.2f}</td><td>{nofog_val:.2f}</td><td>{delta:+.2f}</td>")
        html_parts.append("</tr>")
    
    html_parts.append("</table>")
    
    # Add plots for each metric
    for metric_col, display_name in simple_metrics.items():
        if f"{display_name}_bars" in pngs:
            html_parts.append(f"<div class='metric-section'>")
            html_parts.append(f"<h2>{display_name}</h2>")
            html_parts.append(f"<h3>Per-pair comparison</h3>")
            html_parts.append(f"<img src='data:image/png;base64,{pngs[f'{display_name}_bars']}' style='max-width:100%;'/>")
            html_parts.append(f"<h3>Delta (NoFog - Fog)</h3>")
            html_parts.append(f"<img src='data:image/png;base64,{pngs[f'{display_name}_delta']}' style='max-width:100%;'/>")
            html_parts.append("</div>")
    
    html_parts.append("</body></html>")
    
    # Write HTML
    output_html.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"HTML report saved to: {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate simple pairwise HTML report"
    )
    parser.add_argument(
        "--quality-scores",
        type=Path,
        default=Path("analysis/mesh_quality_batch/quality_scores.csv"),
        help="Path to quality_scores.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/mesh_quality_batch/simple_pairwise_report.html"),
        help="Output HTML path"
    )
    parser.add_argument(
        "--master-report",
        type=Path,
        default=Path("analysis/master_fog_no_fog_report.csv"),
        help="Path to master_fog_no_fog_report.csv for accurate participant names"
    )
    
    args = parser.parse_args()
    
    generate_simple_pairwise_report(
        args.quality_scores,
        args.output,
        args.master_report
    )


if __name__ == "__main__":
    main()

