#!/usr/bin/env python3
"""
Statistical analysis of controller motion data comparing Fog vs NoFog conditions.

Generates publication-quality visualizations and a comprehensive statistical report.

Usage:
    python analysis/analyze_controller_motion_stats.py \
        --input_csv analysis/controller_analysis.csv \
        --output_dir analysis/controller_motion_analysis
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


# Define metrics to analyze (per-hand metrics)
HAND_METRICS = {
    "total_distance_m": ("Total Distance Traveled", "m"),
    "net_displacement_m": ("Net Displacement", "m"),
    "avg_speed_kmh": ("Average Speed", "km/h"),
    "peak_speed_kmh": ("Peak Speed", "km/h"),
    "avg_acceleration_ms2": ("Average Acceleration", "m/s²"),
    "peak_acceleration_ms2": ("Peak Acceleration", "m/s²"),
    "cumulative_rotation_rad": ("Cumulative Rotation", "rad"),
    "avg_angular_speed_rad_s": ("Average Angular Speed", "rad/s"),
    "peak_angular_speed_rad_s": ("Peak Angular Speed", "rad/s"),
    "workspace_volume_m3": ("Workspace Volume", "m³"),
    "jitter_stddev_m": ("Tracking Jitter", "m"),
}

# Inter-hand coordination metrics
INTER_HAND_METRICS = {
    "avg_inter_hand_distance_m": ("Average Inter-Hand Distance", "m"),
    "min_inter_hand_distance_m": ("Minimum Inter-Hand Distance", "m"),
    "max_inter_hand_distance_m": ("Maximum Inter-Hand Distance", "m"),
    "inter_hand_distance_stddev_m": ("Inter-Hand Distance StdDev", "m"),
    "avg_relative_speed_kmh": ("Average Relative Speed", "km/h"),
    "peak_relative_speed_kmh": ("Peak Relative Speed", "km/h"),
    "movement_correlation": ("Movement Correlation", ""),
    "synchronization_score": ("Synchronization Score", ""),
}


def load_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and preprocess controller motion data.
    
    Returns:
        hand_df: DataFrame with per-hand metrics (one row per hand)
        interhand_df: DataFrame with inter-hand metrics (one row per session)
    """
    df = pd.read_csv(csv_path)
    
    # Use condition from CSV if available, otherwise infer from path
    if "condition" not in df.columns:
        df["condition"] = df["capture_path"].apply(
            lambda x: "Fog" if "/Fog/" in x else "NoFog" if "/NoFog/" in x else "Unknown"
        )
    
    # Separate hand-level and inter-hand metrics
    hand_df = df[df["hand"].notna()].copy()  # Rows with hand specified
    interhand_df = df[df["hand"].isna() | (df["avg_inter_hand_distance_m"].notna())].copy()
    
    # For inter-hand, keep only one row per session (they're duplicated)
    if len(interhand_df) > 0:
        interhand_df = interhand_df.drop_duplicates(subset=["capture_name", "capture_path"], keep="first")
    
    return hand_df, interhand_df


def perform_statistical_tests_hand(hand_df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical tests comparing Fog vs NoFog for hand-level metrics."""
    results = []
    
    has_participants = "participant" in hand_df.columns and hand_df["participant"].notna().any()
    
    if has_participants:
        print("[info] Participant information detected - using paired statistical tests")
    
    fog_data = hand_df[hand_df["condition"] == "Fog"]
    nofog_data = hand_df[hand_df["condition"] == "NoFog"]
    
    for metric_col, (display_name, unit) in HAND_METRICS.items():
        if metric_col not in hand_df.columns:
            continue
            
        fog_values = fog_data[metric_col].dropna()
        nofog_values = nofog_data[metric_col].dropna()
        
        if len(fog_values) < 2 or len(nofog_values) < 2:
            continue
        
        # Descriptive statistics
        fog_mean = fog_values.mean()
        fog_std = fog_values.std()
        fog_median = fog_values.median()
        fog_q25 = fog_values.quantile(0.25)
        fog_q75 = fog_values.quantile(0.75)
        
        nofog_mean = nofog_values.mean()
        nofog_std = nofog_values.std()
        nofog_median = nofog_values.median()
        nofog_q25 = nofog_values.quantile(0.25)
        nofog_q75 = nofog_values.quantile(0.75)
        
        # Paired analysis if participants available
        if has_participants:
            paired_df = hand_df[["participant", "condition", "hand", metric_col]].dropna()
            fog_paired = paired_df[paired_df["condition"] == "Fog"].set_index(["participant", "hand"])[metric_col]
            nofog_paired = paired_df[paired_df["condition"] == "NoFog"].set_index(["participant", "hand"])[metric_col]
            
            common_keys = fog_paired.index.intersection(nofog_paired.index)
            if len(common_keys) >= 2:
                fog_paired_vals = fog_paired[common_keys].values
                nofog_paired_vals = nofog_paired[common_keys].values
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
                n_pairs = len(common_keys)
            else:
                has_participants = False
        
        # Independent samples analysis
        if not has_participants or len(common_keys) < 2:
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
            "fog_q25": fog_q25,
            "fog_q75": fog_q75,
            "nofog_n": len(nofog_values),
            "nofog_mean": nofog_mean,
            "nofog_std": nofog_std,
            "nofog_median": nofog_median,
            "nofog_q25": nofog_q25,
            "nofog_q75": nofog_q75,
            "test": test_name,
            "n_pairs": n_pairs,
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "effect_size": effect_size,
        })
    
    return pd.DataFrame(results)


def perform_statistical_tests_interhand(interhand_df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical tests comparing Fog vs NoFog for inter-hand metrics."""
    results = []
    
    has_participants = "participant" in interhand_df.columns and interhand_df["participant"].notna().any()
    
    fog_data = interhand_df[interhand_df["condition"] == "Fog"]
    nofog_data = interhand_df[interhand_df["condition"] == "NoFog"]
    
    for metric_col, (display_name, unit) in INTER_HAND_METRICS.items():
        if metric_col not in interhand_df.columns:
            continue
            
        fog_values = fog_data[metric_col].dropna()
        nofog_values = nofog_data[metric_col].dropna()
        
        if len(fog_values) < 2 or len(nofog_values) < 2:
            continue
        
        # Descriptive statistics
        fog_mean = fog_values.mean()
        fog_std = fog_values.std()
        fog_median = fog_values.median()
        fog_q25 = fog_values.quantile(0.25)
        fog_q75 = fog_values.quantile(0.75)
        
        nofog_mean = nofog_values.mean()
        nofog_std = nofog_values.std()
        nofog_median = nofog_values.median()
        nofog_q25 = nofog_values.quantile(0.25)
        nofog_q75 = nofog_values.quantile(0.75)
        
        # Paired analysis if participants available
        if has_participants:
            paired_df = interhand_df[["participant", "condition", metric_col]].dropna()
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
            "fog_q25": fog_q25,
            "fog_q75": fog_q75,
            "nofog_n": len(nofog_values),
            "nofog_mean": nofog_mean,
            "nofog_std": nofog_std,
            "nofog_median": nofog_median,
            "nofog_q25": nofog_q25,
            "nofog_q75": nofog_q75,
            "test": test_name,
            "n_pairs": n_pairs,
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "effect_size": effect_size,
        })
    
    return pd.DataFrame(results)


def create_box_plots(hand_df: pd.DataFrame, interhand_df: pd.DataFrame, output_dir: Path) -> None:
    """Create box plots comparing Fog vs NoFog."""
    # Hand-level metrics
    n_hand_metrics = len([m for m in HAND_METRICS.keys() if m in hand_df.columns])
    n_interhand_metrics = len([m for m in INTER_HAND_METRICS.keys() if m in interhand_df.columns])
    
    if n_hand_metrics > 0:
        n_cols = 3
        n_rows = (n_hand_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_hand_metrics > 1 else [axes]
        
        idx = 0
        for metric_col, (display_name, unit) in HAND_METRICS.items():
            if metric_col not in hand_df.columns:
                continue
            ax = axes[idx]
            plot_df = hand_df[[metric_col, "condition", "hand"]].dropna()
            
            sns.boxplot(
                data=plot_df,
                x="condition",
                y=metric_col,
                hue="hand",
                ax=ax,
                palette="colorblind",
                showmeans=True,
            )
            
            ax.set_ylabel(f"{display_name} ({unit})")
            ax.set_xlabel("")
            ax.set_title(display_name)
            idx += 1
        
        for i in range(idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "boxplots_hand_metrics.png")
        plt.close()
    
    # Inter-hand metrics
    if n_interhand_metrics > 0:
        n_cols = 3
        n_rows = (n_interhand_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_interhand_metrics > 1 else [axes]
        
        idx = 0
        for metric_col, (display_name, unit) in INTER_HAND_METRICS.items():
            if metric_col not in interhand_df.columns:
                continue
            ax = axes[idx]
            plot_df = interhand_df[[metric_col, "condition"]].dropna()
            
            sns.boxplot(
                data=plot_df,
                x="condition",
                y=metric_col,
                ax=ax,
                palette="colorblind",
                showmeans=True,
            )
            
            ax.set_ylabel(f"{display_name} ({unit})")
            ax.set_xlabel("")
            ax.set_title(display_name)
            idx += 1
        
        for i in range(idx, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_dir / "boxplots_interhand_metrics.png")
        plt.close()


def generate_report(
    hand_stats_df: pd.DataFrame,
    interhand_stats_df: pd.DataFrame,
    hand_df: pd.DataFrame,
    interhand_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Generate comprehensive text report."""
    report_path = output_dir / "statistical_report.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CONTROLLER MOTION ANALYSIS: FOG vs NOFOG COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        fog_n_hand = len(hand_df[hand_df["condition"] == "Fog"])
        nofog_n_hand = len(hand_df[hand_df["condition"] == "NoFog"])
        fog_n_interhand = len(interhand_df[interhand_df["condition"] == "Fog"])
        nofog_n_interhand = len(interhand_df[interhand_df["condition"] == "NoFog"])
        
        f.write(f"Sample Sizes:\n")
        f.write(f"  Hand-level metrics: Fog={fog_n_hand}, NoFog={nofog_n_hand}\n")
        f.write(f"  Inter-hand metrics: Fog={fog_n_interhand}, NoFog={nofog_n_interhand}\n")
        
        if "participant" in hand_df.columns and hand_df["participant"].notna().any():
            n_participants = hand_df["participant"].nunique()
            f.write(f"  Participants: {n_participants}\n")
        f.write("\n")
        
        # Hand-level significant results
        if len(hand_stats_df) > 0:
            f.write("=" * 80 + "\n")
            f.write("HAND-LEVEL METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            significant = hand_stats_df[hand_stats_df["significant"]].sort_values("p_value")
            if len(significant) > 0:
                f.write("SIGNIFICANT DIFFERENCES:\n")
                for _, row in significant.iterrows():
                    f.write(f"\n{row['metric']} ({row['unit']}):\n")
                    f.write(f"  Fog: M={row['fog_mean']:.3f}, SD={row['fog_std']:.3f}\n")
                    f.write(f"  NoFog: M={row['nofog_mean']:.3f}, SD={row['nofog_std']:.3f}\n")
                    f.write(f"  p={row['p_value']:.4f}, d={row['cohens_d']:.3f} ({row['effect_size']})\n")
        
        # Inter-hand significant results
        if len(interhand_stats_df) > 0:
            f.write("\n" + "=" * 80 + "\n")
            f.write("INTER-HAND COORDINATION METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            significant = interhand_stats_df[interhand_stats_df["significant"]].sort_values("p_value")
            if len(significant) > 0:
                f.write("SIGNIFICANT DIFFERENCES:\n")
                for _, row in significant.iterrows():
                    f.write(f"\n{row['metric']} ({row['unit']}):\n")
                    f.write(f"  Fog: M={row['fog_mean']:.3f}, SD={row['fog_std']:.3f}\n")
                    f.write(f"  NoFog: M={row['nofog_mean']:.3f}, SD={row['nofog_std']:.3f}\n")
                    f.write(f"  p={row['p_value']:.4f}, d={row['cohens_d']:.3f} ({row['effect_size']})\n")
    
    print(f"[info] Report written to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of controller motion data"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(__file__).parent / "controller_analysis.csv",
        help="Path to input CSV file with controller motion data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "controller_motion_analysis",
        help="Output directory for results",
    )
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[info] Loading data from: {args.input_csv}")
    hand_df, interhand_df = load_data(args.input_csv)
    print(f"[info] Loaded {len(hand_df)} hand-level records, {len(interhand_df)} inter-hand records")
    
    # Perform statistical tests
    print("[info] Performing statistical tests...")
    hand_stats_df = perform_statistical_tests_hand(hand_df)
    interhand_stats_df = perform_statistical_tests_interhand(interhand_df)
    
    # Save statistics tables
    hand_stats_df.to_csv(args.output_dir / "statistical_results_hand.csv", index=False)
    interhand_stats_df.to_csv(args.output_dir / "statistical_results_interhand.csv", index=False)
    print(f"[info] Statistical results saved")
    
    # Create visualizations
    print("[info] Creating visualizations...")
    create_box_plots(hand_df, interhand_df, args.output_dir)
    print("[info] Created box plots")
    
    # Generate report
    print("[info] Generating statistical report...")
    generate_report(hand_stats_df, interhand_stats_df, hand_df, interhand_df, args.output_dir)
    
    print(f"\n[info] Analysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

