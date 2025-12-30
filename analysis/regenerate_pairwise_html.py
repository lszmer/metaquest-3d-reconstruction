#!/usr/bin/env python3
"""
Regenerate pairwise HTML report from fixed pairwise_summary.csv.
"""

import pandas as pd
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
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


def regenerate_html(pairwise_csv: Path, output_html: Path):
    """Regenerate HTML report from pairwise summary CSV."""
    
    print(f"Loading pairwise summary from {pairwise_csv}...")
    df = pd.read_csv(pairwise_csv)
    
    print(f"Found {len(df)} pairs")
    
    # Extract data
    participants = df["participant"].tolist()
    pair_ids = df["pair_id"].tolist()
    fog_names = df["fog_name"].tolist()
    fog_q_norm = df["fog_Q_norm"].astype(float).tolist()
    nofog_names = df["nofog_name"].tolist()
    nofog_q_norm = df["nofog_Q_norm"].astype(float).tolist()
    deltas_nofog_minus_fog = df["delta_nofog_minus_fog"].astype(float).tolist()
    deltas_fog_minus_nofog = df["delta_fog_minus_nofog"].astype(float).tolist()
    
    # Generate plots
    pngs = {}
    
    # Bar chart per pair
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(participants))
    ax.bar([i - 0.2 for i in x], fog_q_norm, width=0.4, label="Fog")
    ax.bar([i + 0.2 for i in x], nofog_q_norm, width=0.4, label="NoFog")
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha="right")
    ax.set_ylabel("Q_norm")
    ax.set_title("Quality scores per pair (normalized)")
    ax.legend()
    pngs["Per-pair scores"] = _fig_to_base64(fig)
    
    # Delta plot
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["green" if d >= 0 else "red" for d in deltas_nofog_minus_fog]
    ax.bar(x, deltas_nofog_minus_fog, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha="right")
    ax.set_ylabel("Delta (NoFog - Fog)")
    ax.set_title("Score delta per pair (Q_norm)")
    pngs["Score delta"] = _fig_to_base64(fig)
    
    # Box plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([fog_q_norm, nofog_q_norm])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Fog", "NoFog"])
    ax.set_ylabel("Q_norm")
    ax.set_title("Score distribution")
    pngs["Distribution"] = _fig_to_base64(fig)
    
    # Generate HTML
    html_parts = [
        "<html><head><title>Fog vs NoFog Mesh Quality</title>",
        "<style>table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:6px;}th{background:#eee;}</style>",
        "</head><body>",
        "<h2>Fog vs NoFog Mesh Quality (normalized scores)</h2>",
        "<table>",
        "<tr><th>Participant</th><th>Pair ID</th><th>Fog</th><th>Fog Q_norm</th><th>NoFog</th><th>NoFog Q_norm</th><th>Delta (NoFog-Fog)</th><th>Delta (Fog-NoFog)</th></tr>",
    ]
    
    for i in range(len(df)):
        html_parts.append(
            f"<tr><td>{participants[i]}</td><td>{pair_ids[i]}</td><td>{fog_names[i]}</td>"
            f"<td>{fog_q_norm[i]:.6f}</td><td>{nofog_names[i]}</td>"
            f"<td>{nofog_q_norm[i]:.6f}</td><td>{deltas_nofog_minus_fog[i]:.6f}</td><td>{deltas_fog_minus_nofog[i]:.6f}</td></tr>"
        )
    
    html_parts.append("</table><br/>")
    for title, b64 in pngs.items():
        html_parts.append(f"<h3>{title}</h3><img src='data:image/png;base64,{b64}' style='max-width:100%;'/>")
    html_parts.append("</body></html>")
    
    # Write HTML
    output_html.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"HTML report saved to: {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate pairwise HTML report from CSV"
    )
    parser.add_argument(
        "--pairwise-csv",
        type=Path,
        default=Path("analysis/mesh_quality_batch/pairwise_summary.csv"),
        help="Path to pairwise_summary.csv"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/mesh_quality_batch/pairwise_quality_report.html"),
        help="Output HTML path"
    )
    
    args = parser.parse_args()
    
    regenerate_html(args.pairwise_csv, args.output)


if __name__ == "__main__":
    main()

