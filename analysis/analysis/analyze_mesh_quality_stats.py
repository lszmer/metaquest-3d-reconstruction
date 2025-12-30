#!/usr/bin/env python3
"""
Statistical analysis of 3D mesh reconstruction quality scores comparing Fog vs NoFog conditions.

This script analyzes the quality of reconstructed 3D meshes from the 3D reconstruction pipeline,
comparing quality metrics between fog and no-fog experimental conditions. It performs
paired statistical tests to determine if fog conditions lead to improved mesh quality.

Key features:
- Analyzes multiple quality metrics: geometry, smoothness, completeness, color accuracy
- Performs paired statistical tests (t-tests, Wilcoxon) comparing fog vs no-fog conditions
- Tests directional hypothesis that fog improves mesh quality (one-tailed tests)
- Generates quality comparison reports and visualizations
- Handles participant pairing for within-subject analysis

Console Usage Examples:
    # Basic mesh quality analysis with default paths
    python analysis/analysis/analyze_mesh_quality_stats.py

    # Specify custom quality scores file and output directory
    python analysis/analysis/analyze_mesh_quality_stats.py \
        --quality-scores analysis/mesh_quality_batch/quality_scores.csv \
        --output-dir analysis/mesh_quality_analysis_results

    # Analyze quality scores from a different experiment
    python analysis/analysis/analyze_mesh_quality_stats.py \
        --quality-scores /data/experiment2/mesh_quality.csv \
        --output-dir /results/experiment2/quality_analysis/
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
    """
    Extract participant and pair identifier from mesh name.
    
    Names follow pattern: {participant}_{nofog_session}__{fog_session}_{condition}
    Returns: participant_pair_id (e.g., "Lena Ehrenreich_20251209_142927__20251209_144306")
    """
    # Remove the condition suffix (_fog or _nofog)
    if name.endswith("_fog"):
        base = name[:-4]
    elif name.endswith("_nofog"):
        base = name[:-6]
    else:
        return None
    return base


def load_and_pair_quality_data(csv_path: Path) -> pd.DataFrame:
    """
    Load quality scores CSV and create paired dataset.
    
    Returns DataFrame with columns: participant_pair_id, condition, and all quality metrics.
    """
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
    """
    Check normality of data using Shapiro-Wilk test.
    
    Returns: (is_normal, p_value)
    """
    if len(data) < 3:
        return False, 1.0
    
    # Shapiro-Wilk is reliable for n <= 5000
    if len(data) <= 5000:
        _, p_value = stats.shapiro(data)
        is_normal = p_value > alpha
    else:
        # For larger samples, use Anderson-Darling
        result = stats.anderson(data)
        # Critical value at alpha=0.05 is typically around 0.787
        is_normal = result.statistic < 0.787
        p_value = 0.05 if is_normal else 0.01
    
    return is_normal, p_value


def perform_paired_test(
    fog_values: np.ndarray,
    nofog_values: np.ndarray,
    metric_name: str
) -> Dict:
    """
    Perform paired statistical test comparing fog vs nofog.
    
    Tests hypothesis: fog > nofog (one-tailed)
    
    Returns dictionary with test results.
    """
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
    
    # Handle case where all differences are zero (no variance)
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
            "p_value": 1.0,  # Cannot reject null hypothesis
            "is_normal": True,
            "normality_p": 1.0,
            "cohens_d": 0.0,
            "significant": False
        }
    
    # Check normality of differences
    is_normal, normality_p = check_normality(differences)
    
    # Perform appropriate test
    if is_normal:
        # Paired t-test (one-tailed: fog > nofog)
        stat, p_value_two_tailed = stats.ttest_rel(fog_values, nofog_values)
        # Convert to one-tailed: if stat > 0 (fog > nofog), p = p_two_tailed / 2
        # If stat < 0 (fog < nofog), p = 1 - (p_two_tailed / 2)
        if stat > 0:
            p_value = p_value_two_tailed / 2.0
        else:
            p_value = 1.0 - (p_value_two_tailed / 2.0)
        test_name = "Paired t-test (one-tailed: fog > nofog)"
    else:
        # Wilcoxon signed-rank test (one-tailed: fog > nofog)
        stat, p_value = stats.wilcoxon(fog_values, nofog_values, alternative="greater")
        test_name = "Wilcoxon signed-rank (one-tailed: fog > nofog)"
    
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


def analyze_quality_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform statistical analysis on all quality metrics.
    
    Returns DataFrame with test results for each metric.
    """
    # Quality score metrics
    quality_metrics = ["Q_raw", "Q_norm"]
    
    # Sub-score metrics
    sub_score_metrics = ["S_geom", "S_smooth", "S_complete", "S_color", "S_shape", "S_topology", "S_bonuses"]
    
    all_metrics = quality_metrics + sub_score_metrics
    
    results = []
    
    for metric in all_metrics:
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in data, skipping...")
            continue
        
        # Get paired data
        paired_df = df[["participant_pair_id", "condition", metric]].dropna()
        
        fog_data = paired_df[paired_df["condition"] == "Fog"].set_index("participant_pair_id")[metric]
        nofog_data = paired_df[paired_df["condition"] == "NoFog"].set_index("participant_pair_id")[metric]
        
        # Get common pairs
        common_pairs = fog_data.index.intersection(nofog_data.index)

        if len(common_pairs) < 2:
            print(f"Warning: Insufficient pairs for metric '{metric}' ({len(common_pairs)} pairs), skipping...")
            continue

        fog_values = fog_data[common_pairs].values
        nofog_values = nofog_data[common_pairs].values

        # Store pair identifiers for reference
        pair_ids = list(common_pairs)
        
        # Perform test
        result = perform_paired_test(fog_values, nofog_values, metric)
        result["metric"] = metric
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


def generate_report(results_df: pd.DataFrame, output_path: Path):
    """Generate a comprehensive statistical report."""
    
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MESH QUALITY STATISTICAL ANALYSIS REPORT\n")
        f.write("Hypothesis: Fog conditions produce higher quality scores than NoFog conditions\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("1. Paired design: Each participant has both Fog and NoFog meshes\n")
        f.write("2. Normality check: Shapiro-Wilk test on differences (α = 0.05)\n")
        f.write("3. Statistical test:\n")
        f.write("   - If differences are normal: Paired t-test (one-tailed: fog > nofog)\n")
        f.write("   - If differences are non-normal: Wilcoxon signed-rank test (one-tailed: fog > nofog)\n")
        f.write("4. Effect size: Cohen's d for paired samples\n")
        f.write("5. Significance level: α = 0.05\n\n")
        
        f.write("RESULTS\n")
        f.write("-" * 80 + "\n\n")
        
        # Overall quality scores
        f.write("OVERALL QUALITY SCORES\n")
        f.write("=" * 80 + "\n\n")
        
        quality_metrics = ["Q_raw", "Q_norm"]
        for metric in quality_metrics:
            if metric not in results_df["metric"].values:
                continue
            
            row = results_df[results_df["metric"] == metric].iloc[0]
            
            f.write(f"{metric}:\n")
            f.write(f"  N pairs: {row['n_pairs']}\n")
            f.write(f"  Fog:      M = {row['fog_mean']:.4f}, SD = {row['fog_std']:.4f}\n")
            f.write(f"  NoFog:    M = {row['nofog_mean']:.4f}, SD = {row['nofog_std']:.4f}\n")
            f.write(f"  Difference: M = {row['mean_difference']:.4f}, SD = {row['std_difference']:.4f}\n")
            f.write(f"  Normality check: {'Normal' if row['is_normal'] else 'Non-normal'} (p = {row['normality_p']:.4f})\n")
            f.write(f"  Test: {row['test_name']}\n")
            stat_val = row['statistic']
            if pd.notna(stat_val):
                f.write(f"  Statistic: {stat_val:.4f}\n")
            p_val = row['p_value']
            if pd.isna(p_val):
                f.write(f"  p-value: N/A (no variance)\n")
            else:
                f.write(f"  p-value: {p_val:.4f}\n")
            f.write(f"  Effect size (Cohen's d): {row['cohens_d']:.4f} ({interpret_effect_size(row['cohens_d'])})\n")
            f.write(f"  Significant: {'Yes' if row['significant'] else 'No'} (α = 0.05)\n")
            f.write("\n")
        
        # Sub-scores
        f.write("\nSUB-SCORE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        sub_score_metrics = ["S_geom", "S_smooth", "S_complete", "S_color", "S_shape", "S_topology", "S_bonuses"]
        
        for metric in sub_score_metrics:
            if metric not in results_df["metric"].values:
                continue
            
            row = results_df[results_df["metric"] == metric].iloc[0]
            
            f.write(f"{metric}:\n")
            f.write(f"  N pairs: {row['n_pairs']}\n")
            f.write(f"  Fog:      M = {row['fog_mean']:.4f}, SD = {row['fog_std']:.4f}\n")
            f.write(f"  NoFog:    M = {row['nofog_mean']:.4f}, SD = {row['nofog_std']:.4f}\n")
            f.write(f"  Difference: M = {row['mean_difference']:.4f}, SD = {row['std_difference']:.4f}\n")
            f.write(f"  Normality: {'Normal' if row['is_normal'] else 'Non-normal'} (p = {row['normality_p']:.4f})\n")
            f.write(f"  Test: {row['test_name']}\n")
            p_val = row['p_value']
            if pd.isna(p_val):
                f.write(f"  p-value: N/A (no variance)\n")
            else:
                f.write(f"  p-value: {p_val:.4f}\n")
            f.write(f"  Effect size: {row['cohens_d']:.4f} ({interpret_effect_size(row['cohens_d'])})\n")
            f.write(f"  Significant: {'Yes' if row['significant'] else 'No'}\n")
            f.write("\n")
        
        # Summary table
        f.write("\nSUMMARY TABLE\n")
        f.write("=" * 80 + "\n\n")
        cohens_d_label = "Cohen's d"
        f.write(f"{'Metric':<20} {'N':<5} {'Fog M':<10} {'NoFog M':<10} {'Diff':<10} {'p-value':<10} {cohens_d_label:<12} {'Sig':<5}\n")
        f.write("-" * 80 + "\n")
        
        for _, row in results_df.iterrows():
            sig_marker = "***" if row['significant'] else ""
            p_val = row['p_value']
            p_val_str = "N/A" if pd.isna(p_val) else f"{p_val:.4f}"
            f.write(f"{row['metric']:<20} {row['n_pairs']:<5} "
                   f"{row['fog_mean']:<10.4f} {row['nofog_mean']:<10.4f} "
                   f"{row['mean_difference']:<10.4f} {p_val_str:<10} "
                   f"{row['cohens_d']:<12.4f} {sig_marker:<5}\n")
        
        f.write("\n*** = p < 0.05\n")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of mesh quality scores (Fog vs NoFog)"
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
    df = load_and_pair_quality_data(args.quality_scores)
    print(f"Loaded {len(df)} mesh records")
    print(f"Found {df['participant_pair_id'].nunique()} unique pairs")
    
    # Perform statistical analysis
    print("\nPerforming statistical tests...")
    results_df = analyze_quality_metrics(df)
    
    # Save results CSV
    results_csv = args.output_dir / "quality_statistical_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\nResults saved to: {results_csv}")
    
    # Generate report
    report_path = args.output_dir / "quality_statistical_report.txt"
    generate_report(results_df, report_path)
    print(f"Report saved to: {report_path}")
    
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
        print(f"  {row['metric']}: p = {p_str}, d = {row['cohens_d']:.4f}")


if __name__ == "__main__":
    main()

