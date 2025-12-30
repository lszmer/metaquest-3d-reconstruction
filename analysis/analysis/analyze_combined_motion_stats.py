#!/usr/bin/env python3
"""
Combined HMD and Controller motion analysis for robust activity metrics.

This script integrates head-mounted display (HMD) body movement data with controller
(hand tracking) data to create comprehensive metrics for analyzing user activity and
motion patterns across fog vs no-fog experimental conditions.

Key features:
- Combines body movement (HMD) with hand movement (controllers) for holistic activity analysis
- Creates robust distance and activity metrics by cross-validating multiple data sources
- Generates combined statistical comparisons between experimental conditions
- Produces integrated visualizations showing relationships between different motion types
- Particularly useful for understanding overall user engagement and exploration behavior

Console Usage Examples:
    # Basic combined motion analysis with default paths
    python analysis/analysis/analyze_combined_motion_stats.py

    # Specify custom input files and output directory
    python analysis/analysis/analyze_combined_motion_stats.py \
        --hmd_csv analysis/data/hmd_analysis.csv \
        --controller_csv analysis/data/controller_analysis.csv \
        --output_dir analysis/combined_analysis_results

    # Analyze specific datasets from different locations
    python analysis/analysis/analyze_combined_motion_stats.py \
        --hmd_csv /data/experiment1/hmd_stats.csv \
        --controller_csv /data/experiment1/controller_stats.csv \
        --output_dir /results/experiment1/combined/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Set publication-quality style
sns.set_style("whitegrid")
sns.set_palette("colorblind")
plt.rcParams.update({
    "font.size": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def load_and_merge_data(hmd_csv: Path, controller_csv: Path) -> pd.DataFrame:
    """Load HMD and controller data and merge them by session."""
    hmd_df = pd.read_csv(hmd_csv)
    controller_df = pd.read_csv(controller_csv)
    
    # Ensure condition column exists
    if "condition" not in hmd_df.columns:
        hmd_df["condition"] = hmd_df["capture_path"].apply(
            lambda x: "Fog" if "/Fog/" in x else "NoFog" if "/NoFog/" in x else "Unknown"
        )
    
    # Aggregate controller data per session (average left/right hands)
    controller_hand_df = controller_df[controller_df["hand"].notna()].copy()
    filtered_df = controller_df[controller_df["avg_inter_hand_distance_m"].notna()]
    # Remove duplicates by grouping and taking first occurrence
    controller_interhand_df = filtered_df.groupby(["capture_name", "capture_path"], as_index=False).first().copy()
    
    # Aggregate hand metrics (average of left and right)
    hand_agg = controller_hand_df.groupby(["capture_name", "capture_path", "participant", "condition"]).agg({
        "total_distance_m": "mean",
        "net_displacement_m": "mean",
        "avg_speed_kmh": "mean",
        "peak_speed_kmh": "max",  # Use max for peak
        "avg_acceleration_ms2": "mean",
        "peak_acceleration_ms2": "max",
        "cumulative_rotation_rad": "mean",
        "avg_angular_speed_rad_s": "mean",
        "peak_angular_speed_rad_s": "max",
        "workspace_volume_m3": "sum",  # Sum workspace volumes
        "jitter_stddev_m": "mean",
    }).reset_index()
    
    # Merge with inter-hand metrics
    controller_merged = pd.merge(
        hand_agg,
        controller_interhand_df[["capture_name", "capture_path", "avg_inter_hand_distance_m",
                                  "movement_correlation", "synchronization_score"]],
        on=["capture_name", "capture_path"],
        how="left"
    )
    
    # Merge HMD and controller data
    merged_df = pd.merge(
        hmd_df,
        controller_merged,
        on=["capture_name", "capture_path", "participant", "condition"],
        how="inner",  # Only keep sessions with both HMD and controller data
        suffixes=("_hmd", "_controller")
    )
    
    return merged_df


def compute_combined_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute combined/robust metrics from HMD and controller data."""
    df = df.copy()
    
    # Combined distance metrics (more robust by using multiple sources)
    df["combined_total_distance_m"] = (
        df["body_distance_m"] + 
        df["total_distance_m"].fillna(0)  # Controller distance
    )
    
    # Combined speed metrics
    df["combined_avg_speed_kmh"] = (
        (df["body_avg_speed_kmh"] + df["avg_speed_kmh"].fillna(0)) / 2
    )
    df["combined_peak_speed_kmh"] = df[
        ["body_peak_speed_kmh", "peak_speed_kmh"]
    ].max(axis=1, skipna=True)
    
    # Activity score (normalized combination of body and hand movement)
    # Normalize each component to 0-1 range, then combine
    body_dist_norm = (df["body_distance_m"] - df["body_distance_m"].min()) / (
        df["body_distance_m"].max() - df["body_distance_m"].min() + 1e-10
    )
    hand_dist_norm = (df["total_distance_m"] - df["total_distance_m"].min()) / (
        df["total_distance_m"].max() - df["total_distance_m"].min() + 1e-10
    )
    df["activity_score"] = (body_dist_norm + hand_dist_norm.fillna(0)) / 2
    
    # Motion complexity (combination of body movement and hand coordination)
    body_speed_norm = (df["body_avg_speed_kmh"] - df["body_avg_speed_kmh"].min()) / (
        df["body_avg_speed_kmh"].max() - df["body_avg_speed_kmh"].min() + 1e-10
    )
    hand_speed_norm = (df["avg_speed_kmh"] - df["avg_speed_kmh"].min()) / (
        df["avg_speed_kmh"].max() - df["avg_speed_kmh"].min() + 1e-10
    )
    sync_score = df["synchronization_score"].fillna(0.5)  # Default to neutral
    df["motion_complexity"] = (
        body_speed_norm * 0.4 + 
        hand_speed_norm.fillna(0) * 0.4 + 
        sync_score * 0.2
    )
    
    # Engagement metric (combination of head rotation and hand movement)
    head_rot_norm = (df["head_cumulative_radians"] - df["head_cumulative_radians"].min()) / (
        df["head_cumulative_radians"].max() - df["head_cumulative_radians"].min() + 1e-10
    )
    hand_rot_norm = (df["cumulative_rotation_rad"] - df["cumulative_rotation_rad"].min()) / (
        df["cumulative_rotation_rad"].max() - df["cumulative_rotation_rad"].min() + 1e-10
    )
    df["engagement_score"] = (
        head_rot_norm * 0.5 + 
        hand_rot_norm.fillna(0) * 0.5
    )
    
    # Workspace utilization (combination of body displacement and hand workspace)
    body_disp_norm = (df["body_net_displacement_m"] - df["body_net_displacement_m"].min()) / (
        df["body_net_displacement_m"].max() - df["body_net_displacement_m"].min() + 1e-10
    )
    hand_workspace_norm = (df["workspace_volume_m3"] - df["workspace_volume_m3"].min()) / (
        df["workspace_volume_m3"].max() - df["workspace_volume_m3"].min() + 1e-10
    )
    df["workspace_utilization"] = (
        body_disp_norm * 0.6 + 
        hand_workspace_norm.fillna(0) * 0.4
    )
    
    return df


def perform_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical tests on combined metrics."""
    results = []
    
    # Define combined metrics to test
    combined_metrics = {
        "combined_total_distance_m": ("Combined Total Distance", "m"),
        "combined_avg_speed_kmh": ("Combined Average Speed", "km/h"),
        "combined_peak_speed_kmh": ("Combined Peak Speed", "km/h"),
        "activity_score": ("Activity Score", ""),
        "motion_complexity": ("Motion Complexity", ""),
        "engagement_score": ("Engagement Score", ""),
        "workspace_utilization": ("Workspace Utilization", ""),
    }
    
    has_participants = "participant" in df.columns and df["participant"].notna().any()
    
    fog_data = df[df["condition"] == "Fog"]
    nofog_data = df[df["condition"] == "NoFog"]
    
    for metric_col, (display_name, unit) in combined_metrics.items():
        if metric_col not in df.columns:
            continue
            
        fog_values = fog_data[metric_col].dropna()
        nofog_values = nofog_data[metric_col].dropna()
        
        if len(fog_values) < 2 or len(nofog_values) < 2:
            continue
        
        # Descriptive statistics
        fog_mean = fog_values.mean()
        fog_std = fog_values.std()
        fog_median = fog_values.median()
        
        nofog_mean = nofog_values.mean()
        nofog_std = nofog_values.std()
        nofog_median = nofog_values.median()
        
        # Paired analysis if participants available
        if has_participants:
            paired_df = df[["participant", "condition", metric_col]].dropna()
            fog_paired = paired_df[paired_df["condition"] == "Fog"].set_index("participant")[metric_col]
            nofog_paired = paired_df[paired_df["condition"] == "NoFog"].set_index("participant")[metric_col]
            
            common_participants = fog_paired.index.intersection(nofog_paired.index)
            if len(common_participants) >= 2:
                fog_paired_vals = fog_paired[common_participants].values
                nofog_paired_vals = nofog_paired[common_participants].values
                differences = fog_paired_vals - nofog_paired_vals
                
                _, diff_normal = stats.shapiro(differences) if len(differences) <= 5000 else (None, 0.05)
                
                if diff_normal > 0.05:
                    stat, p_value = stats.ttest_rel(fog_paired_vals, nofog_paired_vals)
                    test_name = "Paired t-test"
                else:
                    stat, p_value = stats.wilcoxon(fog_paired_vals, nofog_paired_vals, alternative="two-sided")
                    test_name = "Wilcoxon signed-rank"
                
                diff_mean = differences.mean()
                diff_std = differences.std()
                cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0
                n_pairs = len(common_participants)
            else:
                has_participants = False
        
        # Independent samples analysis
        if not has_participants or len(common_participants) < 2:
            _, fog_normal = stats.shapiro(fog_values) if len(fog_values) <= 5000 else (None, 0.05)
            _, nofog_normal = stats.shapiro(nofog_values) if len(nofog_values) <= 5000 else (None, 0.05)
            
            if fog_normal > 0.05 and nofog_normal > 0.05:
                stat, p_value = stats.ttest_ind(fog_values, nofog_values)
                test_name = "Independent samples t-test"
            else:
                stat, p_value = stats.mannwhitneyu(fog_values, nofog_values, alternative="two-sided")
                test_name = "Mann-Whitney U"
            
            pooled_std = np.sqrt(((len(fog_values) - 1) * fog_values.std()**2 + 
                                  (len(nofog_values) - 1) * nofog_values.std()**2) / 
                                 (len(fog_values) + len(nofog_values) - 2))
            cohens_d = (fog_mean - nofog_mean) / pooled_std if pooled_std > 0 else 0.0
            n_pairs = None
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        results.append({
            "metric": display_name,
            "unit": unit,
            "fog_n": len(fog_values),
            "fog_mean": fog_mean,
            "fog_std": fog_std,
            "fog_median": fog_median,
            "nofog_n": len(nofog_values),
            "nofog_mean": nofog_mean,
            "nofog_std": nofog_std,
            "nofog_median": nofog_median,
            "test": test_name,
            "n_pairs": n_pairs,
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "effect_size": effect_size,
        })
    
    return pd.DataFrame(results)


def create_visualizations(df: pd.DataFrame, stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Create visualizations for combined metrics."""
    combined_metrics = [
        "combined_total_distance_m",
        "combined_avg_speed_kmh",
        "activity_score",
        "motion_complexity",
        "engagement_score",
        "workspace_utilization",
    ]
    
    # Filter to metrics that exist
    available_metrics = [m for m in combined_metrics if m in df.columns]
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    metric_names = {
        "combined_total_distance_m": "Combined Total Distance (m)",
        "combined_avg_speed_kmh": "Combined Average Speed (km/h)",
        "activity_score": "Activity Score",
        "motion_complexity": "Motion Complexity",
        "engagement_score": "Engagement Score",
        "workspace_utilization": "Workspace Utilization",
    }
    
    for idx, metric_col in enumerate(available_metrics):
        ax = axes[idx]
        temp_df = df[[metric_col, "condition"]].dropna()
        plot_df = pd.DataFrame({"condition": temp_df["condition"], metric_col: temp_df[metric_col]})
        
        sns.boxplot(
            data=plot_df,
            x="condition",
            y=metric_col,
            ax=ax,
            palette="colorblind",
            showmeans=True,
        )
        
        ax.set_ylabel(metric_names.get(metric_col, metric_col))
        ax.set_xlabel("")
        ax.set_title(metric_names.get(metric_col, metric_col))
        
        # Add significance indicator if available
        if len(stats_df) > 0:
            metric_display = metric_names.get(metric_col, metric_col).split("(")[0].strip()
            stat_row = stats_df[stats_df["metric"] == metric_display]
            if len(stat_row) > 0 and stat_row.iloc[0]["significant"]:
                p_val = stat_row.iloc[0]["p_value"]
                p_text = f"p={p_val:.3f}" if p_val >= 0.001 else "p<0.001"
                ax.text(0.5, ax.get_ylim()[1] * 0.95, p_text, ha="center", fontsize=9, 
                       bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5))
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "combined_metrics_boxplots.png")
    plt.close()


def generate_report(stats_df: pd.DataFrame, df: pd.DataFrame, output_dir: Path) -> None:
    """Generate comprehensive report."""
    report_path = output_dir / "combined_analysis_report.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("COMBINED HMD-CONTROLLER MOTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("This analysis combines HMD body movement and controller hand movement\n")
        f.write("data to create more robust metrics for analyzing user activity.\n\n")
        
        fog_n = len(df[df["condition"] == "Fog"])
        nofog_n = len(df[df["condition"] == "NoFog"])
        
        f.write(f"Sample Sizes:\n")
        f.write(f"  Fog condition: {fog_n} sessions\n")
        f.write(f"  NoFog condition: {nofog_n} sessions\n")
        f.write(f"  Total: {fog_n + nofog_n} sessions\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("COMBINED METRICS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. Combined Total Distance: Sum of body and hand movement distances\n")
        f.write("2. Combined Average Speed: Average of body and hand speeds\n")
        f.write("3. Combined Peak Speed: Maximum of body and hand peak speeds\n")
        f.write("4. Activity Score: Normalized combination of body and hand movement\n")
        f.write("5. Motion Complexity: Combination of speeds and hand coordination\n")
        f.write("6. Engagement Score: Combination of head and hand rotation\n")
        f.write("7. Workspace Utilization: Combination of body displacement and hand workspace\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        significant = stats_df[stats_df["significant"]].sort_values("p_value")
        if len(significant) > 0:
            f.write("SIGNIFICANT DIFFERENCES (p < 0.05):\n")
            f.write("-" * 80 + "\n")
            for _, row in significant.iterrows():
                f.write(f"\n{row['metric']} ({row['unit']}):\n")
                f.write(f"  Fog: M={row['fog_mean']:.3f}, SD={row['fog_std']:.3f}\n")
                f.write(f"  NoFog: M={row['nofog_mean']:.3f}, SD={row['nofog_std']:.3f}\n")
                f.write(f"  p={row['p_value']:.4f}, Cohen's d={row['cohens_d']:.3f} ({row['effect_size']})\n")
        
        non_significant = stats_df[~stats_df["significant"]]
        if len(non_significant) > 0:
            f.write("\nNON-SIGNIFICANT DIFFERENCES:\n")
            f.write("-" * 80 + "\n")
            for _, row in non_significant.iterrows():
                f.write(f"{row['metric']}: p={row['p_value']:.4f}, d={row['cohens_d']:.3f}\n")
    
    print(f"[info] Report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Combined HMD and controller motion analysis"
    )
    parser.add_argument(
        "--hmd_csv",
        type=Path,
        default=Path(__file__).parent / "hmd_analysis.csv",
        help="Path to HMD motion analysis CSV",
    )
    parser.add_argument(
        "--controller_csv",
        type=Path,
        default=Path(__file__).parent / "controller_analysis.csv",
        help="Path to controller motion analysis CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "combined_motion_analysis",
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and merge data
    print(f"[info] Loading HMD data from: {args.hmd_csv}")
    print(f"[info] Loading controller data from: {args.controller_csv}")
    merged_df = load_and_merge_data(args.hmd_csv, args.controller_csv)
    print(f"[info] Merged {len(merged_df)} sessions with both HMD and controller data")
    
    # Compute combined metrics
    print("[info] Computing combined metrics...")
    merged_df = compute_combined_metrics(merged_df)
    
    # Save merged data
    merged_df.to_csv(args.output_dir / "combined_data.csv", index=False)
    print(f"[info] Saved combined data to: {args.output_dir / 'combined_data.csv'}")
    
    # Perform statistical tests
    print("[info] Performing statistical tests...")
    stats_df = perform_statistical_tests(merged_df)
    stats_df.to_csv(args.output_dir / "statistical_results.csv", index=False)
    print(f"[info] Statistical results saved")
    
    # Create visualizations
    print("[info] Creating visualizations...")
    create_visualizations(merged_df, stats_df, args.output_dir)
    print("[info] Created visualizations")
    
    # Generate report
    print("[info] Generating report...")
    generate_report(stats_df, merged_df, args.output_dir)
    
    print(f"\n[info] Analysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

