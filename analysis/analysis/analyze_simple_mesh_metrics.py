#!/usr/bin/env python3
"""
Simple geometric analysis of 3D mesh properties comparing Fog vs NoFog conditions.

This script performs statistical analysis of basic geometric properties of reconstructed
3D meshes, focusing on fundamental mesh characteristics rather than complex quality
metrics. It compares these basic properties between fog and no-fog experimental conditions.

Key features:
- Analyzes basic geometric metrics: vertex/triangle counts, component count, boundary edges
- Checks mesh integrity: watertight status, degenerate triangles
- Performs paired statistical tests comparing fog vs no-fog conditions
- Generates simple geometric comparison reports and visualizations
- Useful for understanding fundamental differences in mesh structure

Console Usage Examples:
    # Basic geometric mesh analysis with default paths
    python analysis/analysis/analyze_simple_mesh_metrics.py

    # Specify custom quality scores file and output directory
    python analysis/analysis/analyze_simple_mesh_metrics.py \
        --quality-scores analysis/mesh_quality_batch/quality_scores.csv \
        --output-dir analysis/simple_mesh_analysis_results

    # Analyze geometric properties from different quality scores
    python analysis/analysis/analyze_simple_mesh_metrics.py \
        --quality-scores /data/experiment2/simple_mesh_metrics.csv \
        --output-dir /results/experiment2/geometric_analysis/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def extract_participant_pair_id(name: str) -> Optional[str]:
    """Extract participant and pair identifier from mesh name."""
    if name.endswith("_fog"):
        base = name[:-4]
    elif name.endswith("_nofog"):
        base = name[:-6]
    else:
        return None
    return base


def load_and_pair_data(csv_path: Path) -> pd.DataFrame:
    """Load quality scores CSV and create paired dataset."""
    df = pd.read_csv(csv_path)
    
    # Extract participant_pair_id
    df["participant_pair_id"] = df["name"].apply(extract_participant_pair_id)
    
    # Extract condition from name
    df["condition"] = df["name"].apply(
        lambda x: "Fog" if x.endswith("_fog") else ("NoFog" if x.endswith("_nofog") else None)
    )
    
    # Filter out rows where we couldn't extract pair info
    df = df[df["participant_pair_id"].notna() & df["condition"].notna()].copy()
    
    return df


def check_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """Check normality of data using Shapiro-Wilk test."""
    if len(data) < 3:
        return False, 1.0
    
    if len(data) <= 5000:
        _, p_value = stats.shapiro(data)
        is_normal = p_value > alpha
    else:
        result = stats.anderson(data)
        is_normal = result.statistic < 0.787
        p_value = 0.05 if is_normal else 0.01
    
    return is_normal, p_value


def perform_paired_test(
    fog_values: np.ndarray,
    nofog_values: np.ndarray,
    metric_name: str
) -> Dict:
    """Perform paired statistical test comparing fog vs nofog."""
    if len(fog_values) != len(nofog_values):
        raise ValueError("Fog and nofog values must have same length")
    
    if len(fog_values) < 2:
        return {
            "n_pairs": len(fog_values),
            "fog_mean": np.mean(fog_values) if len(fog_values) > 0 else np.nan,
            "nofog_mean": np.mean(nofog_values) if len(nofog_values) > 0 else np.nan,
            "mean_difference": np.nan,
            "test_name": "Insufficient data",
            "statistic": np.nan,
            "p_value": np.nan,
            "is_normal": False,
            "normality_p": np.nan,
            "cohens_d": np.nan,
            "significant": False
        }
    
    differences = fog_values - nofog_values
    
    # Handle zero variance
    if np.all(differences == 0) or np.std(differences) == 0:
        return {
            "n_pairs": len(fog_values),
            "fog_mean": fog_values.mean(),
            "nofog_mean": nofog_values.mean(),
            "fog_std": fog_values.std(),
            "nofog_std": nofog_values.std(),
            "mean_difference": 0.0,
            "std_difference": 0.0,
            "test_name": "No variance in differences",
            "statistic": np.nan,
            "p_value": 1.0,
            "is_normal": True,
            "normality_p": 1.0,
            "cohens_d": 0.0,
            "significant": False
        }
    
    # Check normality of differences
    is_normal, normality_p = check_normality(differences)
    
    # Perform appropriate test (two-tailed for now)
    if is_normal:
        stat, p_value = stats.ttest_rel(fog_values, nofog_values)
        test_name = "Paired t-test (two-tailed)"
    else:
        stat, p_value = stats.wilcoxon(fog_values, nofog_values, alternative="two-sided")
        test_name = "Wilcoxon signed-rank (two-tailed)"
    
    # Compute effect size (Cohen's d for paired samples)
    diff_mean = differences.mean()
    diff_std = differences.std()
    cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0
    
    # Significance at alpha = 0.05
    significant = p_value < 0.05
    
    return {
        "n_pairs": len(fog_values),
        "fog_mean": fog_values.mean(),
        "nofog_mean": nofog_values.mean(),
        "fog_std": fog_values.std(),
        "nofog_std": nofog_values.std(),
        "mean_difference": diff_mean,
        "std_difference": diff_std,
        "test_name": test_name,
        "statistic": stat,
        "p_value": p_value,
        "is_normal": is_normal,
        "normality_p": normality_p,
        "cohens_d": cohens_d,
        "significant": significant
    }


def analyze_simple_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical analysis on simple geometric metrics."""
    
    # Simple metrics to analyze
    simple_metrics = {
        "num_vertices": "Vertex Count",
        "num_triangles": "Triangle Count",
        "component_count": "Component Count",
        "boundary_edge_ratio": "Boundary Edge Ratio",
        "degenerate_triangles": "Degenerate Triangles",
    }
    
    results = []
    
    for metric_col, display_name in simple_metrics.items():
        if metric_col not in df.columns:
            print(f"Warning: Metric '{metric_col}' not found in data, skipping...")
            continue
        
        # Get paired data
        paired_df = df[["participant_pair_id", "condition", metric_col]].dropna()
        
        fog_data = paired_df[paired_df["condition"] == "Fog"].set_index("participant_pair_id")[metric_col]
        nofog_data = paired_df[paired_df["condition"] == "NoFog"].set_index("participant_pair_id")[metric_col]
        
        # Get common pairs
        common_pairs = fog_data.index.intersection(nofog_data.index)
        
        if len(common_pairs) < 2:
            print(f"Warning: Insufficient pairs for metric '{metric_col}' ({len(common_pairs)} pairs), skipping...")
            continue
        
        fog_values = fog_data[common_pairs].values
        nofog_values = nofog_data[common_pairs].values
        
        # Perform test
        result = perform_paired_test(fog_values, nofog_values, metric_col)
        result["metric"] = metric_col
        result["display_name"] = display_name
        results.append(result)
    
    return pd.DataFrame(results)


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size."""
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def generate_text_report(results_df: pd.DataFrame, output_path: Path):
    """Generate a comprehensive text report."""
    
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SIMPLE MESH METRICS STATISTICAL ANALYSIS REPORT\n")
        f.write("Comparing Fog vs NoFog conditions\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("1. Paired design: Each participant has both Fog and NoFog meshes\n")
        f.write("2. Normality check: Shapiro-Wilk test on differences (α = 0.05)\n")
        f.write("3. Statistical test:\n")
        f.write("   - If differences are normal: Paired t-test (two-tailed)\n")
        f.write("   - If differences are non-normal: Wilcoxon signed-rank test (two-tailed)\n")
        f.write("4. Effect size: Cohen's d for paired samples\n")
        f.write("5. Significance level: α = 0.05\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        for _, row in results_df.iterrows():
            f.write(f"{row['display_name']} ({row['metric']}):\n")
            f.write(f"  N pairs: {row['n_pairs']}\n")
            f.write(f"  Fog:      M = {row['fog_mean']:.2f}, SD = {row['fog_std']:.2f}\n")
            f.write(f"  NoFog:    M = {row['nofog_mean']:.2f}, SD = {row['nofog_std']:.2f}\n")
            f.write(f"  Difference: M = {row['mean_difference']:.2f}, SD = {row['std_difference']:.2f}\n")
            f.write(f"  Normality: {'Normal' if row['is_normal'] else 'Non-normal'} (p = {row['normality_p']:.4f})\n")
            f.write(f"  Test: {row['test_name']}\n")
            p_val = row['p_value']
            if pd.isna(p_val):
                f.write(f"  p-value: N/A (no variance)\n")
            else:
                f.write(f"  p-value: {p_val:.4f}\n")
            f.write(f"  Effect size (Cohen's d): {row['cohens_d']:.4f} ({interpret_effect_size(row['cohens_d'])})\n")
            f.write(f"  Significant: {'Yes' if row['significant'] else 'No'} (α = 0.05)\n")
            f.write("\n")
        
        # Summary table
        f.write("\nSUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")
        cohens_d_label = "Cohen's d"
        f.write(f"{'Metric':<30} {'N':<5} {'Fog M':<12} {'NoFog M':<12} {'Diff':<12} {'p-value':<12} {cohens_d_label:<12} {'Sig':<5}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in results_df.iterrows():
            sig_marker = "***" if row['significant'] else ""
            p_val = row['p_value']
            p_val_str = "N/A" if pd.isna(p_val) else f"{p_val:.4f}"
            f.write(f"{row['display_name']:<30} {row['n_pairs']:<5} "
                   f"{row['fog_mean']:<12.2f} {row['nofog_mean']:<12.2f} "
                   f"{row['mean_difference']:<12.2f} {p_val_str:<12} "
                   f"{row['cohens_d']:<12.4f} {sig_marker:<5}\n")
        
        f.write("\n*** = p < 0.05\n")


def generate_html_report(results_df: pd.DataFrame, output_path: Path):
    """Generate an HTML report with tables and visualizations."""
    
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>Simple Mesh Metrics Analysis</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 20px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
        "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
        "th { background-color: #4CAF50; color: white; }",
        "tr:nth-child(even) { background-color: #f2f2f2; }",
        ".significant { background-color: #ffeb3b; }",
        "h1, h2 { color: #333; }",
        ".summary { margin: 20px 0; padding: 15px; background-color: #f9f9f9; border-left: 4px solid #4CAF50; }",
        "</style>",
        "</head><body>",
        "<h1>Simple Mesh Metrics Statistical Analysis</h1>",
        "<h2>Fog vs NoFog Comparison</h2>",
        "<div class='summary'>",
        "<p><strong>Methodology:</strong> Paired design with normality checks. ",
        "Paired t-test (normal differences) or Wilcoxon signed-rank test (non-normal differences). ",
        "Effect size: Cohen's d. Significance level: α = 0.05</p>",
        "</div>",
        "<h2>Results Summary</h2>",
        "<table>",
        "<tr><th>Metric</th><th>N</th><th>Fog Mean</th><th>NoFog Mean</th><th>Difference</th><th>p-value</th><th>Cohen's d</th><th>Significant</th></tr>",
    ]
    
    for _, row in results_df.iterrows():
        sig_class = "significant" if row['significant'] else ""
        p_val = row['p_value']
        p_val_str = "N/A" if pd.isna(p_val) else f"{p_val:.4f}"
        sig_marker = "***" if row['significant'] else ""
        
        html_parts.append(
            f"<tr class='{sig_class}'>"
            f"<td>{row['display_name']}</td>"
            f"<td>{row['n_pairs']}</td>"
            f"<td>{row['fog_mean']:.2f}</td>"
            f"<td>{row['nofog_mean']:.2f}</td>"
            f"<td>{row['mean_difference']:.2f}</td>"
            f"<td>{p_val_str}</td>"
            f"<td>{row['cohens_d']:.4f}</td>"
            f"<td>{sig_marker}</td>"
            f"</tr>"
        )
    
    html_parts.extend([
        "</table>",
        "<p><strong>***</strong> = p < 0.05</p>",
        "<h2>Detailed Results</h2>",
    ])
    
    for _, row in results_df.iterrows():
        html_parts.append(f"<h3>{row['display_name']}</h3>")
        html_parts.append("<table>")
        html_parts.append(f"<tr><th>Property</th><th>Value</th></tr>")
        html_parts.append(f"<tr><td>N pairs</td><td>{row['n_pairs']}</td></tr>")
        html_parts.append(f"<tr><td>Fog Mean</td><td>{row['fog_mean']:.2f}</td></tr>")
        html_parts.append(f"<tr><td>Fog SD</td><td>{row['fog_std']:.2f}</td></tr>")
        html_parts.append(f"<tr><td>NoFog Mean</td><td>{row['nofog_mean']:.2f}</td></tr>")
        html_parts.append(f"<tr><td>NoFog SD</td><td>{row['nofog_std']:.2f}</td></tr>")
        html_parts.append(f"<tr><td>Mean Difference</td><td>{row['mean_difference']:.2f}</td></tr>")
        html_parts.append(f"<tr><td>Normality</td><td>{'Normal' if row['is_normal'] else 'Non-normal'} (p = {row['normality_p']:.4f})</td></tr>")
        html_parts.append(f"<tr><td>Test</td><td>{row['test_name']}</td></tr>")
        p_val = row['p_value']
        p_val_str = "N/A (no variance)" if pd.isna(p_val) else f"{p_val:.4f}"
        html_parts.append(f"<tr><td>p-value</td><td>{p_val_str}</td></tr>")
        html_parts.append(f"<tr><td>Effect size (Cohen's d)</td><td>{row['cohens_d']:.4f} ({interpret_effect_size(row['cohens_d'])})</td></tr>")
        html_parts.append(f"<tr><td>Significant</td><td>{'Yes' if row['significant'] else 'No'}</td></tr>")
        html_parts.append("</table><br/>")
    
    html_parts.append("</body></html>")
    
    output_path.write_text("\n".join(html_parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Simple mesh metrics statistical analysis (Fog vs NoFog)"
    )
    parser.add_argument(
        "--quality-scores",
        type=Path,
        default=Path("analysis/mesh_quality_batch/quality_scores.csv"),
        help="Path to quality_scores.csv file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/mesh_quality_batch"),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Load and prepare data
    print("Loading quality scores data...")
    df = load_and_pair_data(args.quality_scores)
    print(f"Loaded {len(df)} mesh records")
    print(f"Found {df['participant_pair_id'].nunique()} unique pairs")
    
    # Perform statistical analysis
    print("\nPerforming statistical tests on simple metrics...")
    results_df = analyze_simple_metrics(df)
    
    # Save results CSV
    results_csv = args.output_dir / "simple_metrics_statistical_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    # Generate text report
    report_path = args.output_dir / "simple_metrics_statistical_report.txt"
    generate_text_report(results_df, report_path)
    print(f"Text report saved to: {report_path}")
    
    # Generate HTML report
    html_path = args.output_dir / "simple_metrics_statistical_report.html"
    generate_html_report(results_df, html_path)
    print(f"HTML report saved to: {html_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTotal metrics analyzed: {len(results_df)}")
    print(f"Significant results (p < 0.05): {results_df['significant'].sum()}")
    print("\nSignificant metrics:")
    for _, row in results_df[results_df['significant']].iterrows():
        p_val = row['p_value']
        p_str = f"{p_val:.4f}" if pd.notna(p_val) else "N/A"
        print(f"  {row['display_name']}: p = {p_str}, d = {row['cohens_d']:.4f}")


if __name__ == "__main__":
    main()

