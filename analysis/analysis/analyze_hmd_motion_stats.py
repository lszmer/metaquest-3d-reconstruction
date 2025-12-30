#!/usr/bin/env python3
"""
Statistical analysis of HMD motion data comparing Fog vs NoFog conditions.

This script performs comprehensive statistical analysis of head-mounted display (HMD) motion
data, comparing user behavior between fog and no-fog experimental conditions. It generates
publication-quality visualizations, statistical tests, and detailed reports.

Key features:
- Paired statistical tests (t-tests, Wilcoxon) for comparing fog vs no-fog conditions
- Analysis of body movement, head rotation, and viewing sphere coverage metrics
- Automatic detection of paired participants for within-subject analysis
- Generation of box plots, violin plots, bar charts, and improvement analysis
- One-tailed tests for metrics expected to improve with fog (head movement, coverage)

Console Usage Examples:
    # Basic analysis with default paths
    python analysis/analysis/analyze_hmd_motion_stats.py

    # Specify custom input/output paths
    python analysis/analysis/analyze_hmd_motion_stats.py \
        --input_csv analysis/data/hmd_analysis.csv \
        --output_dir analysis/hmd_motion_analysis_custom

    # Exclude specific participants from analysis
    python analysis/analysis/analyze_hmd_motion_stats.py \
        --exclude-participant "Maria" \
        --exclude-participant "John Doe"

    # Merge results into master report
    python analysis/analysis/analyze_hmd_motion_stats.py \
        --merge-to-master \
        --master-report analysis/data/master_fog_no_fog_report.csv
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


# Define metrics to analyze with their display names and units
METRICS = {
    "body_distance_m": ("Total Body Distance", "m"),
    "body_net_displacement_m": ("Net Body Displacement", "m"),
    "body_avg_speed_kmh": ("Average Body Speed", "km/h"),
    "body_peak_speed_kmh": ("Peak Body Speed", "km/h"),
    "head_cumulative_radians": ("Cumulative Head Rotation", "rad"),
    "head_avg_angular_speed_rad_s": ("Average Head Angular Speed", "rad/s"),
    "head_peak_angular_speed_rad_s": ("Peak Head Angular Speed", "rad/s"),
    "yaw_range_rad": ("Yaw Range", "rad"),
    "pitch_range_rad": ("Pitch Range", "rad"),
    "roll_range_rad": ("Roll Range", "rad"),
    "cumulative_vertical_rotation_rad": ("Cumulative Vertical Rotation (Pitch)", "rad"),
    "cumulative_horizontal_rotation_rad": ("Cumulative Horizontal Rotation (Yaw)", "rad"),
    "viewing_sphere_coverage_percent": ("Viewing Sphere Coverage", "%"),
    "viewing_sphere_coverage_with_fov_percent": ("Viewing Sphere Coverage (with FOV)", "%"),
}


def load_data(csv_path: Path) -> pd.DataFrame:
    """Load and preprocess HMD motion data."""
    df = pd.read_csv(csv_path)
    
    # Use condition from CSV if available, otherwise infer from path
    if "condition" not in df.columns:
        df["condition"] = df["capture_path"].apply(
            lambda x: "Fog" if "/Fog/" in x else "NoFog" if "/NoFog/" in x else "Unknown"
        )
    
    return df


def perform_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Perform statistical tests comparing Fog vs NoFog for each metric.
    
    Uses paired tests if participant information is available, otherwise uses independent samples tests.
    For metrics with directional hypotheses (head movement, rotation, coverage), uses one-tailed tests
    expecting fog > nofog.
    """
    results = []
    
    # Check if we have participant information for paired analysis
    has_participants = bool("participant" in df.columns and df["participant"].notna().any())
    
    if has_participants:
        print("[info] Participant information detected - using paired statistical tests")
    
    fog_data = df[df["condition"] == "Fog"]
    nofog_data = df[df["condition"] == "NoFog"]
    
    # Metrics where we expect fog > nofog (one-tailed tests)
    # These represent metrics where fog condition should show better/more activity
    improvement_metrics = [
        "head_avg_angular_speed_rad_s",  # Average head angular speed
        "head_cumulative_radians",  # Cumulative head rotation
        "cumulative_vertical_rotation_rad",  # Cumulative vertical rotation (pitch)
        "cumulative_horizontal_rotation_rad",  # Cumulative horizontal rotation (yaw)
        "viewing_sphere_coverage_percent",  # Viewing sphere coverage
        "viewing_sphere_coverage_with_fov_percent",  # Viewing sphere coverage with FOV
    ]
    
    for metric_col, (display_name, unit) in METRICS.items():
        # Skip if metric column doesn't exist in the data (e.g., old CSV files)
        if metric_col not in df.columns:
            print(f"[warn] Skipping metric '{metric_col}' - not found in data (may need to recompute HMD stats)")
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
        common_participants = []
        if has_participants:
            # Create paired dataset
            paired_df = df[["participant", "condition", metric_col]].dropna()
            fog_paired = paired_df[paired_df["condition"] == "Fog"].set_index("participant")[metric_col]
            nofog_paired = paired_df[paired_df["condition"] == "NoFog"].set_index("participant")[metric_col]
            
            # Get common participants
            common_participants = fog_paired.index.intersection(nofog_paired.index)
            if len(common_participants) >= 2:
                fog_paired_vals = fog_paired[common_participants].values
                nofog_paired_vals = nofog_paired[common_participants].values
                differences = fog_paired_vals - nofog_paired_vals
                
                # Test normality of differences
                _, diff_normal = stats.shapiro(differences) if len(differences) <= 5000 else (None, 0.05)
                
                # Determine if one-tailed test is appropriate (for improvement metrics)
                is_improvement_metric = metric_col in improvement_metrics
                alternative = "greater" if is_improvement_metric else "two-sided"
                
                if diff_normal > 0.05:
                    # Paired t-test
                    if is_improvement_metric:
                        # One-tailed: test if fog > nofog
                        stat, p_value_two_tailed = stats.ttest_rel(fog_paired_vals, nofog_paired_vals)
                        # Convert to one-tailed p-value
                        if stat > 0:  # Fog is greater
                            p_value = p_value_two_tailed / 2.0
                        else:  # Fog is not greater
                            p_value = 1.0 - (p_value_two_tailed / 2.0)
                        test_name = "Paired t-test (one-tailed: fog > nofog)"
                    else:
                        stat, p_value = stats.ttest_rel(fog_paired_vals, nofog_paired_vals)
                        test_name = "Paired t-test"
                else:
                    # Wilcoxon signed-rank test
                    stat, p_value = stats.wilcoxon(fog_paired_vals, nofog_paired_vals, alternative=alternative)
                    test_name = f"Wilcoxon signed-rank ({alternative})" if is_improvement_metric else "Wilcoxon signed-rank"
                
                # Effect size for paired data (Cohen's d for paired samples)
                diff_mean = differences.mean()
                diff_std = differences.std()
                cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0
                n_pairs = len(common_participants)
            else:
                # Fall back to independent samples
                has_participants = False
        
        # Independent samples analysis (fallback or if no participants)
        if not has_participants or len(common_participants) < 2:
            # Test normality (Shapiro-Wilk)
            _, fog_normal = stats.shapiro(fog_values) if len(fog_values) <= 5000 else (None, 0.05)
            _, nofog_normal = stats.shapiro(nofog_values) if len(nofog_values) <= 5000 else (None, 0.05)
            
            # Choose appropriate test
            if fog_normal > 0.05 and nofog_normal > 0.05:
                # Both normal: use t-test
                stat, p_value = stats.ttest_ind(fog_values, nofog_values)
                test_name = "Independent samples t-test"
            else:
                # Non-normal: use Mann-Whitney U
                stat, p_value = stats.mannwhitneyu(fog_values, nofog_values, alternative="two-sided")
                test_name = "Mann-Whitney U"
            
            # Effect size (Cohen's d for independent samples)
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


def create_box_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create box plots comparing Fog vs NoFog for each metric using seaborn."""
    # Count available metrics first
    available_metrics = [(col, name) for col, name in METRICS.items() if col in df.columns]
    
    if len(available_metrics) == 0:
        print("[warn] No metrics available for box plots")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_col, (display_name, unit)) in enumerate(available_metrics):
        ax = axes[idx]
        
        temp_df = df[[metric_col, "condition"]].dropna()
        plot_df = pd.DataFrame({"condition": temp_df["condition"], metric_col: temp_df[metric_col]})
        
        # Use seaborn boxplot for better styling
        sns.boxplot(
            data=plot_df,
            x="condition",
            y=metric_col,
            hue="condition",        # avoid seaborn palette warning
            dodge=False,
            ax=ax,
            palette="colorblind",
            showmeans=True,
            legend=False,
        )
        
        ax.set_ylabel(f"{display_name} ({unit})")
        ax.set_xlabel("")
        ax.set_title(display_name)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "boxplots_comparison.png")
    plt.close()


def create_violin_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create violin plots comparing Fog vs NoFog for each metric using seaborn."""
    # Count available metrics first
    available_metrics = [(col, name) for col, name in METRICS.items() if col in df.columns]
    
    if len(available_metrics) == 0:
        print("[warn] No metrics available for violin plots")
        return
    
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for idx, (metric_col, (display_name, unit)) in enumerate(available_metrics):
        ax = axes[idx]
        
        temp_df = df[[metric_col, "condition"]].dropna()
        plot_df = pd.DataFrame({"condition": temp_df["condition"], metric_col: temp_df[metric_col]})
        
        # Use seaborn violinplot for better styling
        sns.violinplot(
            data=plot_df,
            x="condition",
            y=metric_col,
            hue="condition",        # avoid seaborn palette warning
            dodge=False,
            ax=ax,
            palette="colorblind",
            inner="quart",  # Show quartiles inside
            legend=False,
        )
        
        ax.set_ylabel(f"{display_name} ({unit})")
        ax.set_xlabel("")
        ax.set_title(display_name)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "violinplots_comparison.png")
    plt.close()


def create_paired_plots(df: pd.DataFrame, output_dir: Path) -> None:
    """Create paired plots showing individual participant changes between Fog and NoFog."""
    if "participant" not in df.columns or bool(df["participant"].isna().all()):
        print("[warn] No participant information available - skipping paired plots")
        return
    
    # Select key metrics for paired visualization (filter to those that exist)
    key_metrics = [
        "body_distance_m",
        "body_avg_speed_kmh",
        "head_cumulative_radians",
        "head_avg_angular_speed_rad_s",
        "cumulative_vertical_rotation_rad",
        "cumulative_horizontal_rotation_rad",
        "viewing_sphere_coverage_with_fov_percent",
    ]
    
    # Filter to metrics that exist in the data
    available_key_metrics = [m for m in key_metrics if m in df.columns]
    
    if len(available_key_metrics) == 0:
        print("[warn] No key metrics available for paired plots")
        return
    
    n_metrics = len(available_key_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric_col in enumerate(available_key_metrics):
        ax = axes[idx]
        metric_name = METRICS[metric_col][0]
        unit = METRICS[metric_col][1]
        
        # Create paired dataset
        paired_df = df[["participant", "condition", metric_col]].dropna()
        fog_data = paired_df[paired_df["condition"] == "Fog"].set_index("participant")[metric_col]
        nofog_data = paired_df[paired_df["condition"] == "NoFog"].set_index("participant")[metric_col]
        
        # Get common participants
        common_participants = fog_data.index.intersection(nofog_data.index)
        if len(common_participants) == 0:
            continue
        
        fog_vals = fog_data[common_participants].values
        nofog_vals = nofog_data[common_participants].values
        
        # Create positions for plotting (NoFog on left, Fog on right)
        x_pos = np.arange(len(common_participants))
        x_nofog = x_pos - 0.15
        x_fog = x_pos + 0.15
        
        # Plot points (NoFog first, then Fog)
        ax.scatter(x_nofog, nofog_vals, color=sns.color_palette("colorblind")[1], 
                  s=50, alpha=0.7, label="NoFog", zorder=3)
        ax.scatter(x_fog, fog_vals, color=sns.color_palette("colorblind")[0], 
                  s=50, alpha=0.7, label="Fog", zorder=3)
        
        # Draw lines connecting paired points (from NoFog to Fog)
        for i in range(len(common_participants)):
            ax.plot([x_nofog[i], x_fog[i]], [nofog_vals[i], fog_vals[i]], 
                   'k-', alpha=0.3, linewidth=0.5, zorder=1)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels([p[:10] + "..." if len(p) > 10 else p for p in common_participants], 
                          rotation=45, ha="right")
        ax.set_ylabel(f"{metric_name} ({unit})")
        ax.set_title(f"{metric_name}\n(Paired by Participant)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "paired_participant_plots.png")
    plt.close()


def analyze_improvements(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """
    Analyze improvements (fog - nofog) for each participant.
    Focuses on coverage metrics where fog is expected to be better.
    """
    if "participant" not in df.columns or bool(df["participant"].isna().all()):
        print("[warn] No participant information - skipping improvement analysis")
        return pd.DataFrame()
    
    # Metrics where we expect fog > nofog (one-tailed tests)
    improvement_metrics = [
        "head_avg_angular_speed_rad_s",  # Average head angular speed
        "head_cumulative_radians",  # Cumulative head rotation
        "cumulative_vertical_rotation_rad",  # Cumulative vertical rotation (pitch)
        "cumulative_horizontal_rotation_rad",  # Cumulative horizontal rotation (yaw)
        "viewing_sphere_coverage_percent",  # Viewing sphere coverage
        "viewing_sphere_coverage_with_fov_percent",  # Viewing sphere coverage with FOV
    ]
    
    # Filter to metrics that exist
    available_metrics = [m for m in improvement_metrics if m in df.columns]
    
    if len(available_metrics) == 0:
        print("[warn] No improvement metrics available")
        return pd.DataFrame()
    
    improvements = []
    
    for metric_col in available_metrics:
        display_name = METRICS[metric_col][0]
        unit = METRICS[metric_col][1]
        
        # Create paired dataset
        paired_df = df[["participant", "condition", metric_col]].dropna()
        fog_data = paired_df[paired_df["condition"] == "Fog"].set_index("participant")[metric_col]
        nofog_data = paired_df[paired_df["condition"] == "NoFog"].set_index("participant")[metric_col]
        
        # Get common participants
        common_participants = fog_data.index.intersection(nofog_data.index)
        if len(common_participants) < 2:
            continue
        
        fog_vals = fog_data[common_participants].values
        nofog_vals = nofog_data[common_participants].values
        differences = fog_vals - nofog_vals
        
        # Statistical test: is mean improvement significantly > 0?
        _, diff_normal = stats.shapiro(differences) if len(differences) <= 5000 else (None, 0.05)
        
        if diff_normal > 0.05:
            # One-sample t-test: test if mean difference > 0
            stat, p_value_two_tailed = stats.ttest_1samp(differences, 0.0)
            if stat > 0:  # Mean is positive
                p_value = p_value_two_tailed / 2.0
            else:
                p_value = 1.0 - (p_value_two_tailed / 2.0)
            test_name = "One-sample t-test (one-tailed: improvement > 0)"
        else:
            # Wilcoxon signed-rank test: test if median > 0
            stat, p_value = stats.wilcoxon(differences, alternative="greater")
            test_name = "Wilcoxon signed-rank (one-tailed: improvement > 0)"
        
        # Effect size (Cohen's d for one-sample)
        diff_mean = differences.mean()
        diff_std = differences.std()
        cohens_d = diff_mean / diff_std if diff_std > 0 else 0.0
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            effect_size = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_size = "small"
        elif abs(cohens_d) < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        improvements.append({
            "metric": display_name,
            "unit": unit,
            "n_participants": len(common_participants),
            "mean_improvement": diff_mean,
            "std_improvement": diff_std,
            "median_improvement": float(np.median(differences)),
            "min_improvement": float(np.min(differences)),
            "max_improvement": float(np.max(differences)),
            "improvement_percent": (diff_mean / abs(nofog_vals.mean()) * 100) if abs(nofog_vals.mean()) > 1e-10 else 0.0,
            "test": test_name,
            "statistic": stat,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "cohens_d": cohens_d,
            "effect_size": effect_size,
        })
        
        # Create improvement plot for this metric
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(common_participants))
        colors = ['green' if d > 0 else 'red' for d in differences]
        
        bars = ax.barh(x_pos, differences, color=colors, alpha=0.7, edgecolor='black')
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        
        # Add mean improvement line
        ax.axvline(x=diff_mean, color='blue', linestyle='-', linewidth=2, 
                  label=f'Mean improvement: {diff_mean:.2f} {unit}')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels([p[:15] + "..." if len(p) > 15 else p for p in common_participants])
        ax.set_xlabel(f"Improvement ({unit})\n(Fog - NoFog)")
        ax.set_title(f"{display_name}\nIndividual Participant Improvements\n"
                    f"Mean: {diff_mean:.2f} {unit}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, differences)):
            ax.text(val + (0.01 * max(differences)) if val >= 0 else val - (0.01 * max(differences)),
                   i, f'{val:.2f}', va='center',
                   ha='left' if val >= 0 else 'right', fontsize=9)
        
        plt.tight_layout()
        safe_name = display_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        plt.savefig(output_dir / f"improvements_{safe_name}.png")
        plt.close()
    
    # Create summary improvement plot if we have multiple metrics
    if len(improvements) > 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = [row["metric"] for row in improvements]
        mean_improvements = [row["mean_improvement"] for row in improvements]
        std_improvements = [row["std_improvement"] for row in improvements]
        p_values = [row["p_value"] for row in improvements]
        
        x_pos = np.arange(len(metrics))
        colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'gray' for p in p_values]
        
        bars = ax.barh(x_pos, mean_improvements, xerr=std_improvements, 
                      color=colors, alpha=0.7, edgecolor='black', capsize=5)
        
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(x_pos)
        ax.set_yticklabels(metrics)
        ax.set_xlabel("Mean Improvement (Fog - NoFog)")
        ax.set_title("Summary of Improvements Across Coverage Metrics")
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add significance indicators
        if mean_improvements:  # Check if list is not empty
            max_improvement = max(mean_improvements) if mean_improvements else 1.0
            for i, (bar, p_val, mean_imp) in enumerate(zip(bars, p_values, mean_improvements)):
                sig_text = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                ax.text(mean_imp + std_improvements[i] + 0.01 * max_improvement,
                       i, sig_text, va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "improvements_summary.png")
        plt.close()
    
    return pd.DataFrame(improvements)


def create_summary_bar_chart(stats_df: pd.DataFrame, output_dir: Path) -> None:
    """Create bar chart with error bars showing mean ± SD for key metrics."""
    key_metrics = [
        "body_distance_m",
        "body_avg_speed_kmh",
        "head_cumulative_radians",
        "head_avg_angular_speed_rad_s",
        "cumulative_vertical_rotation_rad",
        "cumulative_horizontal_rotation_rad",
        "viewing_sphere_coverage_with_fov_percent",
    ]
    
    # Filter to metrics that exist in stats_df
    available_metrics = []
    for metric_col in key_metrics:
        metric_name = METRICS[metric_col][0]
        if len(stats_df[stats_df["metric"] == metric_name]) > 0:
            available_metrics.append(metric_col)
    
    if len(available_metrics) == 0:
        print("[warn] No key metrics available for summary bar chart")
        return
    
    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric_col in enumerate(available_metrics):
        ax = axes[idx]
        metric_name = METRICS[metric_col][0]
        unit = METRICS[metric_col][1]
        
        row = stats_df[stats_df["metric"] == metric_name].iloc[0]
        
        x = np.arange(2)
        means = [row["fog_mean"], row["nofog_mean"]]
        stds = [row["fog_std"], row["nofog_std"]]
        # Use seaborn colorblind palette
        palette = sns.color_palette("colorblind", 2)
        colors = [palette[0], palette[1]]
        
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor="black")
        
        # Add significance indicator
        if row["significant"] and means and stds:  # Ensure lists are not empty
            max_val = max(means) + max(stds)
            ax.plot([0, 1], [max_val * 1.1, max_val * 1.1], "k-", linewidth=1)
            ax.plot([0, 0], [max_val * 1.05, max_val * 1.1], "k-", linewidth=1)
            ax.plot([1, 1], [max_val * 1.05, max_val * 1.1], "k-", linewidth=1)
            p_text = f"p={row['p_value']:.3f}" if row['p_value'] >= 0.001 else "p<0.001"
            ax.text(0.5, max_val * 1.15, p_text, ha="center", fontsize=9)
        
        ax.set_xticks(x)
        ax.set_xticklabels(["Fog", "NoFog"])
        ax.set_ylabel(f"{metric_name} ({unit})")
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_dir / "summary_bar_chart.png")
    plt.close()


def generate_report(stats_df: pd.DataFrame, df: pd.DataFrame, improvements_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate a comprehensive text report with statistical interpretation."""
    report_path = output_dir / "statistical_report.txt"
    
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HMD MOTION ANALYSIS: FOG vs NOFOG COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        # Sample sizes
        fog_n = len(df[df["condition"] == "Fog"])
        nofog_n = len(df[df["condition"] == "NoFog"])
        has_participants = bool("participant" in df.columns and df["participant"].notna().any())
        
        f.write(f"Sample Sizes:\n")
        f.write(f"  Fog condition: {fog_n} sessions\n")
        f.write(f"  NoFog condition: {nofog_n} sessions\n")
        f.write(f"  Total: {fog_n + nofog_n} sessions\n")
        
        if has_participants:
            n_participants = df["participant"].nunique()
            f.write(f"  Participants: {n_participants}\n")
            f.write(f"  Design: Paired (each participant has both Fog and NoFog measurements)\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        # Significant results first
        significant = stats_df[stats_df["significant"]].sort_values("p_value")
        if len(significant) > 0:
            f.write("SIGNIFICANT DIFFERENCES (p < 0.05):\n")
            f.write("-" * 80 + "\n")
            for _, row in significant.iterrows():
                f.write(f"\n{row['metric']} ({row['unit']}):\n")
                f.write(f"  Fog:      M={row['fog_mean']:.3f}, SD={row['fog_std']:.3f}, "
                       f"Median={row['fog_median']:.3f}, IQR=[{row['fog_q25']:.3f}, {row['fog_q75']:.3f}]\n")
                f.write(f"  NoFog:    M={row['nofog_mean']:.3f}, SD={row['nofog_std']:.3f}, "
                       f"Median={row['nofog_median']:.3f}, IQR=[{row['nofog_q25']:.3f}, {row['nofog_q75']:.3f}]\n")
                test_info = f"{row['test']}, statistic={row['statistic']:.3f}, p={row['p_value']:.4f}"
                if pd.notna(row.get('n_pairs')):
                    test_info += f", n_pairs={int(row['n_pairs'])}"
                f.write(f"  Test:     {test_info}\n")
                f.write(f"  Effect:    Cohen's d={row['cohens_d']:.3f} ({row['effect_size']})\n")
                
                # Interpretation
                direction = "higher" if row['fog_mean'] > row['nofog_mean'] else "lower"
                f.write(f"  Result:    Fog condition shows {direction} {row['metric'].lower()} "
                       f"compared to NoFog condition.\n")
            f.write("\n")
        
        # Non-significant results
        non_significant = stats_df[~stats_df["significant"]].sort_values("metric")
        if len(non_significant) > 0:
            f.write("NON-SIGNIFICANT DIFFERENCES (p >= 0.05):\n")
            f.write("-" * 80 + "\n")
            for _, row in non_significant.iterrows():
                f.write(f"{row['metric']}: p={row['p_value']:.4f}, "
                       f"Cohen's d={row['cohens_d']:.3f} ({row['effect_size']})\n")
            f.write("\n")
        
        # Improvement analysis section
        if len(improvements_df) > 0:
            f.write("=" * 80 + "\n")
            f.write("IMPROVEMENT ANALYSIS (Fog - NoFog)\n")
            f.write("=" * 80 + "\n\n")
            f.write("This section tests directional hypotheses that Fog > NoFog for:\n")
            f.write("  • Average head angular speed (more head movement)\n")
            f.write("  • Cumulative head rotation (more total rotation)\n")
            f.write("  • Cumulative vertical/horizontal rotation (more structured scanning)\n")
            f.write("  • Viewing sphere coverage (better exploration)\n\n")
            f.write("One-tailed tests are used to test if improvements are significantly > 0.\n\n")
            
            for _, row in improvements_df.iterrows():
                f.write(f"{row['metric']} ({row['unit']}):\n")
                f.write(f"  Mean improvement: {row['mean_improvement']:.3f} {row['unit']}\n")
                f.write(f"  Improvement percentage: {row['improvement_percent']:.1f}% relative to NoFog\n")
                f.write(f"  Range: [{row['min_improvement']:.3f}, {row['max_improvement']:.3f}] {row['unit']}\n")
                f.write(f"  Median: {row['median_improvement']:.3f} {row['unit']}\n")
                f.write(f"  Test: {row['test']}\n")
                f.write(f"  Statistic: {row['statistic']:.3f}, p={row['p_value']:.4f}")
                if row['significant']:
                    f.write(" *** SIGNIFICANT ***\n")
                else:
                    f.write(" (not significant)\n")
                f.write(f"  Effect size: Cohen's d={row['cohens_d']:.3f} ({row['effect_size']})\n")
                
                if row['significant']:
                    f.write(f"  ✓ Fog condition shows significant improvement over NoFog\n")
                    f.write(f"    ({row['n_participants']} participants, mean improvement: {row['mean_improvement']:.2f} {row['unit']})\n")
                else:
                    f.write(f"  ✗ No significant improvement detected\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("INTERPRETATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        n_significant = len(significant)
        n_total = len(stats_df)
        f.write(f"Out of {n_total} metrics analyzed, {n_significant} showed statistically "
               f"significant differences between Fog and NoFog conditions.\n\n")
        
        if n_significant > 0:
            f.write("Key Findings:\n")
            for _, row in significant.head(5).iterrows():  # Top 5 most significant
                direction = "increased" if row['fog_mean'] > row['nofog_mean'] else "decreased"
                f.write(f"  • {row['metric']}: {direction} in Fog condition "
                       f"(p={row['p_value']:.4f}, d={row['cohens_d']:.3f})\n")
        
        f.write("\n")
        f.write("Effect Size Guidelines (Cohen's d):\n")
        f.write("  |d| < 0.2:  Negligible effect\n")
        f.write("  0.2 ≤ |d| < 0.5:  Small effect\n")
        f.write("  0.5 ≤ |d| < 0.8:  Medium effect\n")
        f.write("  |d| ≥ 0.8:  Large effect\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("METHODOLOGY\n")
        f.write("=" * 80 + "\n\n")
        
        if has_participants:
            f.write("PAIRED DESIGN ANALYSIS:\n")
            f.write("  • Each participant completed both Fog and NoFog conditions\n")
            f.write("  • Paired statistical tests account for within-subject variability\n")
            f.write("  • More powerful than independent samples tests for this design\n\n")
        
        f.write("Statistical tests were chosen based on data distribution:\n")
        if has_participants:
            f.write("  • Paired design: Shapiro-Wilk test on differences\n")
            f.write("  • Normal differences: Paired t-test\n")
            f.write("  • Non-normal differences: Wilcoxon signed-rank test\n")
            f.write("  • One-tailed tests (fog > nofog) for metrics with directional hypotheses:\n")
            f.write("    - Average head angular speed\n")
            f.write("    - Cumulative head rotation\n")
            f.write("    - Cumulative vertical/horizontal rotation\n")
            f.write("    - Viewing sphere coverage\n")
        else:
            f.write("  • Independent samples: Shapiro-Wilk test used to assess normality\n")
            f.write("  • Normal distributions: Independent samples t-test\n")
            f.write("  • Non-normal distributions: Mann-Whitney U test\n")
        f.write("  • Effect sizes calculated using Cohen's d\n")
        f.write("  • Significance threshold: α = 0.05\n")
        f.write("  • Improvement analysis uses one-tailed tests to test if improvements > 0\n")
    
    print(f"[info] Report written to: {report_path}")


def merge_hmd_data_to_master_report(hmd_csv_path: Path, master_report_path: Path) -> None:
    """Merge HMD motion data into master_fog_no_fog_report.csv."""
    print(f"[info] Loading HMD data from: {hmd_csv_path}")
    hmd_df = pd.read_csv(hmd_csv_path)

    print(f"[info] Loading master report from: {master_report_path}")
    master_df = pd.read_csv(master_report_path)

    # Create lookup dictionary for HMD data keyed by (session_id, condition)
    hmd_lookup = {}
    for _, row in hmd_df.iterrows():
        session_id = row['capture_name']
        condition = row['condition'].lower()  # fog or nofog
        hmd_lookup[(session_id, condition)] = row

    # Define HMD columns to add
    hmd_columns = [
        'num_samples', 'duration_seconds', 'sampling_hz',
        'body_distance_m', 'body_net_displacement_m', 'body_avg_speed_kmh', 'body_peak_speed_kmh',
        'head_cumulative_radians', 'head_avg_angular_speed_rad_s', 'head_peak_angular_speed_rad_s',
        'yaw_range_rad', 'pitch_range_rad', 'roll_range_rad',
        'cumulative_vertical_rotation_rad', 'cumulative_horizontal_rotation_rad',
        'viewing_sphere_coverage_percent', 'viewing_sphere_coverage_with_fov_percent'
    ]

    # Add HMD columns for both fog and nofog conditions
    for condition in ['fog', 'nofog']:
        for col in hmd_columns:
            new_col = f"{condition}_hmd_{col}"
            master_df[new_col] = None

    # Fill in HMD data
    updated_rows = 0
    for idx, row in master_df.iterrows():
        # Process nofog data
        nofog_session = row.get('nofog_session_id')
        if nofog_session and (nofog_session, 'nofog') in hmd_lookup:
            hmd_data = hmd_lookup[(nofog_session, 'nofog')]
            for col in hmd_columns:
                master_df.at[idx, f"nofog_hmd_{col}"] = hmd_data[col]
            updated_rows += 1

        # Process fog data
        fog_session = row.get('fog_session_id')
        if fog_session and (fog_session, 'fog') in hmd_lookup:
            hmd_data = hmd_lookup[(fog_session, 'fog')]
            for col in hmd_columns:
                master_df.at[idx, f"fog_hmd_{col}"] = hmd_data[col]
            updated_rows += 1

    # Save updated master report
    master_df.to_csv(master_report_path, index=False)
    print(f"[info] Updated {updated_rows} rows in master report with HMD data")
    print(f"[info] Saved updated master report to: {master_report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of HMD motion data"
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=Path(__file__).parent / "hmd_analysis.csv",
        help="Path to input CSV file with HMD motion data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).parent / "hmd_motion_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--exclude-participant",
        action="append",
        default=[],
        help="Participant name to exclude (can be passed multiple times)",
    )
    parser.add_argument(
        "--merge-to-master",
        action="store_true",
        help="Merge HMD data into master_fog_no_fog_report.csv",
    )
    parser.add_argument(
        "--master-report",
        type=Path,
        default=Path(__file__).parent / "master_fog_no_fog_report.csv",
        help="Path to master fog/no-fog report CSV file",
    )
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"[info] Loading data from: {args.input_csv}")
    df = load_data(args.input_csv)
    if args.exclude_participant:
        before = len(df)
        df = df[~df["participant"].isin(args.exclude_participant)]
        after = len(df)
        print(f"[info] Excluded participants {args.exclude_participant}; rows: {before} -> {after}")
    print(f"[info] Loaded {len(df)} sessions")
    
    # Perform statistical tests
    print("[info] Performing statistical tests...")
    stats_df = perform_statistical_tests(df)
    
    # Save statistics table
    stats_csv_path = args.output_dir / "statistical_results.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[info] Statistical results saved to: {stats_csv_path}")
    
    # Create visualizations
    print("[info] Creating visualizations...")
    create_box_plots(df, args.output_dir)
    print("[info] Created box plots")
    
    create_violin_plots(df, args.output_dir)
    print("[info] Created violin plots")
    
    create_summary_bar_chart(stats_df, args.output_dir)
    print("[info] Created summary bar chart")
    
    # Create paired participant plots if participant info available
    create_paired_plots(df, args.output_dir)
    print("[info] Created paired participant plots")
    
    # Analyze improvements (fog - nofog)
    print("[info] Analyzing improvements (Fog - NoFog)...")
    improvements_df = analyze_improvements(df, args.output_dir)
    if len(improvements_df) > 0:
        improvements_csv_path = args.output_dir / "improvement_analysis.csv"
        improvements_df.to_csv(improvements_csv_path, index=False)
        print(f"[info] Improvement analysis saved to: {improvements_csv_path}")
        print("[info] Created improvement visualizations")
    else:
        print("[warn] No improvement analysis performed (missing participant info or metrics)")
    
    # Generate report
    print("[info] Generating statistical report...")
    generate_report(stats_df, df, improvements_df, args.output_dir)
    
    # Optionally merge HMD data into master report
    if args.merge_to_master:
        print("[info] Merging HMD data into master report...")
        merge_hmd_data_to_master_report(args.input_csv, args.master_report)

    print(f"\n[info] Analysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

