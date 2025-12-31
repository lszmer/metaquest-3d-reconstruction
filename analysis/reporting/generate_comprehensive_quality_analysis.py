#!/usr/bin/env python3
"""
Generate comprehensive HTML analysis report for mesh quality scores.

This script creates an interactive HTML report analyzing all quality metrics
from the quality_scores.csv file, comparing fog vs no-fog conditions on a
per-participant basis. It includes:

- Charts comparing fog vs nofog for all quality score metrics
- Per-participant analysis
- Statistical comparisons (means, standard deviations, t-tests)
- Additional insights and trends
- Interactive visualizations

Console Usage Examples:
    # Generate report using default paths
    python analysis/reporting/generate_comprehensive_quality_analysis.py

    # Specify custom quality scores and output location
    python analysis/reporting/generate_comprehensive_quality_analysis.py \
        --quality-scores analysis/mesh_quality_batch/quality_scores.csv \
        --output analysis/reports/comprehensive_quality_analysis.html
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from scipy import stats
import base64
import io
from typing import Dict, List, Tuple, Optional
import re


# Set style for better-looking plots
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
if not HAS_SEABORN:
    plt.style.use('default')


def _fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


def extract_participant_and_condition(name: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract participant name and condition (fog/nofog) from mesh name.
    
    Pattern: {Participant}_{nofog_session}__{fog_session}_{condition}
    Example: "Lena Ehrenreich_20251209_142927__20251209_144306_fog"
    """
    if name.endswith("_fog"):
        base = name[:-4]
        condition = "fog"
    elif name.endswith("_nofog"):
        base = name[:-6]
        condition = "nofog"
    else:
        return None, None
    
    # Split by double underscore to separate the two timestamps
    parts = base.split("__")
    if len(parts) != 2:
        return None, None
    
    # Participant name is everything before the last underscore in the first part
    nofog_part = parts[0]
    last_underscore = nofog_part.rfind("_")
    if last_underscore == -1:
        return None, None
    
    participant = nofog_part[:last_underscore]
    return participant, condition


def categorize_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Categorize columns into different metric types."""
    categories = {
        "quality_scores": [
            "Q_raw", "Q_norm", "S_geom", "S_smooth", "S_complete", 
            "S_color", "S_shape", "S_topology", "S_bonuses"
        ],
        "geometry_metrics": [
            "mean_aspect_ratio", "mean_skewness", "degenerate_triangles",
            "non_manifold_edges", "boundary_edge_ratio", "component_count"
        ],
        "smoothness_metrics": [
            "normal_deviation_avg_deg", "dihedral_min_deg", "dihedral_max_deg",
            "dihedral_penalty", "surface_roughness"
        ],
        "completeness_metrics": [
            "is_single_component", "vertex_density_stddev"
        ],
        "color_metrics": [
            "has_color", "uncolored_vertex_ratio", "color_gradient_stddev"
        ],
        "topology_metrics": [
            "is_manifold", "is_watertight"
        ],
        "size_metrics": [
            "num_vertices", "num_triangles", "total_edges"
        ]
    }
    
    # Filter to only include columns that exist in the dataframe
    result = {}
    for category, cols in categories.items():
        existing_cols = [col for col in cols if col in df.columns]
        if existing_cols:
            result[category] = existing_cols
    
    return result


def create_comparison_chart(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str = None
) -> str:
    """Create a grouped bar chart comparing fog vs nofog by participant."""
    if ylabel is None:
        ylabel = metric
    
    # Prepare data
    participants = sorted(df['participant'].unique())
    fog_values = []
    nofog_values = []
    
    for participant in participants:
        participant_data = df[df['participant'] == participant]
        fog_data = participant_data[participant_data['condition'] == 'fog']
        nofog_data = participant_data[participant_data['condition'] == 'nofog']
        
        fog_val = fog_data[metric].iloc[0] if len(fog_data) > 0 else np.nan
        nofog_val = nofog_data[metric].iloc[0] if len(nofog_data) > 0 else np.nan
        
        fog_values.append(fog_val)
        nofog_values.append(nofog_val)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(participants))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fog_values, width, label='Fog', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, nofog_values, width, label='NoFog', alpha=0.8, color='#e74c3c')
    
    ax.set_xlabel('Participant', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(participants, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    return _fig_to_base64(fig)


def create_boxplot_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str = None
) -> str:
    """Create a boxplot comparing fog vs nofog distributions."""
    if ylabel is None:
        ylabel = metric
    
    fog_values = df[df['condition'] == 'fog'][metric].dropna()
    nofog_values = df[df['condition'] == 'nofog'][metric].dropna()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    data_to_plot = [fog_values, nofog_values]
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    
    # Set labels manually to avoid deprecation warning
    ax.set_xticklabels(['Fog', 'NoFog'])
    
    # Color the boxes
    colors = ['#3498db', '#e74c3c']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    return _fig_to_base64(fig)


def create_scatter_comparison(
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str = "Fog",
    ylabel: str = "NoFog"
) -> str:
    """Create a scatter plot comparing fog vs nofog values."""
    # Create pairs
    participants = sorted(df['participant'].unique())
    fog_vals = []
    nofog_vals = []
    labels = []
    
    for participant in participants:
        participant_data = df[df['participant'] == participant]
        fog_data = participant_data[participant_data['condition'] == 'fog']
        nofog_data = participant_data[participant_data['condition'] == 'nofog']
        
        if len(fog_data) > 0 and len(nofog_data) > 0:
            fog_val = fog_data[metric].iloc[0]
            nofog_val = nofog_data[metric].iloc[0]
            if not (np.isnan(fog_val) or np.isnan(nofog_val)):
                fog_vals.append(fog_val)
                nofog_vals.append(nofog_val)
                labels.append(participant)
    
    if len(fog_vals) == 0:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(fog_vals, nofog_vals, alpha=0.6, s=100, color='#2ecc71')
    
    # Add diagonal line (y=x)
    min_val = min(min(fog_vals), min(nofog_vals))
    max_val = max(max(fog_vals), max(nofog_vals))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x')
    
    # Add participant labels
    for i, label in enumerate(labels):
        ax.annotate(label, (fog_vals[i], nofog_vals[i]), 
                   fontsize=8, alpha=0.7, rotation=45)
    
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    return _fig_to_base64(fig)


def is_higher_better_metric(metric: str) -> bool:
    """Determine if higher values are better for a given metric.
    
    Returns True if higher is better, False if lower is better.
    """
    # Quality scores - higher is better
    quality_scores = ['Q_raw', 'Q_norm', 'S_geom', 'S_smooth', 'S_complete', 
                     'S_color', 'S_shape', 'S_topology', 'S_bonuses']
    if metric in quality_scores:
        return True
    
    # Topology/geometry - higher is better for some
    if metric in ['is_manifold', 'is_watertight', 'is_single_component', 'has_color']:
        return True
    
    # "Badness" metrics - lower is better
    badness_metrics = [
        'mean_aspect_ratio', 'mean_skewness', 'degenerate_triangles',
        'non_manifold_edges', 'boundary_edge_ratio', 'component_count',
        'normal_deviation_avg_deg', 'dihedral_penalty', 'surface_roughness',
        'vertex_density_stddev', 'uncolored_vertex_ratio', 'color_gradient_stddev'
    ]
    if metric in badness_metrics:
        return False
    
    # Size metrics - neutral, but typically more is better for completeness
    if metric in ['num_vertices', 'num_triangles', 'total_edges']:
        return True  # More vertices/triangles generally indicates better reconstruction
    
    # Dihedral angles - min/max are descriptive, penalty is badness
    if 'min' in metric.lower():
        return False  # Lower minimum might indicate issues
    if 'max' in metric.lower():
        return True  # Higher maximum is generally fine
    
    # Default: assume higher is better
    return True


def compute_statistics(df: pd.DataFrame, metric: str) -> Dict:
    """Compute comprehensive statistical comparisons between fog and nofog.
    
    Performs:
    - Normality tests (Shapiro-Wilk)
    - Parametric test (paired t-test) if data is normal
    - Non-parametric test (Wilcoxon signed-rank) if data is not normal
    - One-tailed tests with correct direction based on whether higher/lower is better
    - Two-tailed tests for general difference detection
    
    Ground hypothesis: Fog is better/higher quality
    - For "higher is better" metrics: H0: fog >= nofog, H1: nofog > fog
    - For "lower is better" metrics: H0: fog <= nofog, H1: nofog < fog
    """
    # Get paired data (fog and nofog for each participant)
    participants = sorted(df['participant'].unique())
    fog_paired = []
    nofog_paired = []
    differences = []
    
    for participant in participants:
        participant_data = df[df['participant'] == participant]
        fog_data = participant_data[participant_data['condition'] == 'fog'][metric]
        nofog_data = participant_data[participant_data['condition'] == 'nofog'][metric]
        
        if len(fog_data) > 0 and len(nofog_data) > 0:
            fog_val = fog_data.iloc[0]
            nofog_val = nofog_data.iloc[0]
            if not (np.isnan(fog_val) or np.isnan(nofog_val)):
                fog_paired.append(fog_val)
                nofog_paired.append(nofog_val)
                differences.append(nofog_val - fog_val)
    
    if len(fog_paired) < 2:
        return None
    
    fog_array = np.array(fog_paired)
    nofog_array = np.array(nofog_paired)
    diff_array = np.array(differences)
    
    stats_dict = {
        'fog_mean': float(fog_array.mean()),
        'fog_std': float(fog_array.std()),
        'fog_median': float(np.median(fog_array)),
        'nofog_mean': float(nofog_array.mean()),
        'nofog_std': float(nofog_array.std()),
        'nofog_median': float(np.median(nofog_array)),
        'mean_difference': float(diff_array.mean()),
        'percent_change': float((nofog_array.mean() - fog_array.mean()) / fog_array.mean() * 100) if fog_array.mean() != 0 else 0.0,
        'n_pairs': len(fog_paired)
    }
    
    # Determine if higher is better for this metric
    higher_is_better = is_higher_better_metric(metric)
    
    # Test for normality using Shapiro-Wilk test
    # Test on differences (for paired tests) and individual groups
    try:
        # Check if data has variation (not all same values)
        if len(diff_array) <= 5000 and len(diff_array) >= 3 and np.std(diff_array) > 1e-10:
            _, p_diff_norm = stats.shapiro(diff_array)
            # Consider normal if p > 0.05 for differences (most important for paired tests)
            is_normal = p_diff_norm > 0.05 if p_diff_norm is not None else False
            stats_dict['normality_test_p'] = float(p_diff_norm) if p_diff_norm is not None else None
            stats_dict['is_normal'] = is_normal
        else:
            # Data has no variation or too few samples - cannot test normality
            is_normal = False
            stats_dict['normality_test_p'] = None
            stats_dict['is_normal'] = False
    except:
        is_normal = False
        stats_dict['normality_test_p'] = None
        stats_dict['is_normal'] = False
    
    # Perform appropriate statistical test
    test_type = None
    test_statistic = None
    p_value_two_tailed = None
    p_value_one_tailed = None
    
    if is_normal and len(fog_paired) >= 3:
        # Use paired t-test (parametric)
        try:
            t_stat, p_two = stats.ttest_rel(fog_array, nofog_array)
            test_type = 'paired_t_test'
            test_statistic = float(t_stat)
            p_value_two_tailed = float(p_two)
            
            # One-tailed test: H0: fog >= nofog (for higher is better) or fog <= nofog (for lower is better)
            if higher_is_better:
                # H0: fog >= nofog, H1: nofog > fog
                # If t_stat > 0, nofog is higher, so p_one = p_two / 2
                # If t_stat < 0, fog is higher, so p_one = 1 - p_two / 2
                if t_stat > 0:
                    p_value_one_tailed = float(p_two / 2)
                else:
                    p_value_one_tailed = float(1 - p_two / 2)
            else:
                # H0: fog <= nofog, H1: nofog < fog
                # If t_stat < 0, nofog is lower, so p_one = p_two / 2
                # If t_stat > 0, fog is lower, so p_one = 1 - p_two / 2
                if t_stat < 0:
                    p_value_one_tailed = float(p_two / 2)
                else:
                    p_value_one_tailed = float(1 - p_two / 2)
        except Exception as e:
            pass
    else:
        # Use Wilcoxon signed-rank test (non-parametric)
        try:
            # Check if there's variation in differences
            if np.std(diff_array) < 1e-10:
                # All differences are essentially the same - cannot perform test
                pass
            else:
                # For Wilcoxon, we test if the median difference is significantly different from 0
                w_stat, p_two = stats.wilcoxon(fog_array, nofog_array, alternative='two-sided', zero_method='wilcox')
                test_type = 'wilcoxon_signed_rank'
                test_statistic = float(w_stat) if not np.isnan(w_stat) else None
                p_value_two_tailed = float(p_two) if not np.isnan(p_two) else None
                
                # One-tailed test
                if higher_is_better and p_value_two_tailed is not None:
                    # H0: fog >= nofog, H1: nofog > fog
                    try:
                        w_stat_neg, p_neg = stats.wilcoxon(fog_array, nofog_array, alternative='less', zero_method='wilcox')
                        w_stat_pos, p_pos = stats.wilcoxon(fog_array, nofog_array, alternative='greater', zero_method='wilcox')
                        # If median difference > 0, nofog > fog, use p_neg
                        # If median difference < 0, fog > nofog, use p_pos
                        if np.median(diff_array) > 0:
                            p_value_one_tailed = float(p_neg) if not np.isnan(p_neg) else None
                        else:
                            p_value_one_tailed = float(p_pos) if not np.isnan(p_pos) else None
                    except:
                        p_value_one_tailed = None
                elif not higher_is_better and p_value_two_tailed is not None:
                    # H0: fog <= nofog, H1: nofog < fog
                    try:
                        w_stat_pos, p_pos = stats.wilcoxon(fog_array, nofog_array, alternative='greater', zero_method='wilcox')
                        w_stat_neg, p_neg = stats.wilcoxon(fog_array, nofog_array, alternative='less', zero_method='wilcox')
                        # If median difference < 0, nofog < fog, use p_neg
                        # If median difference > 0, fog < nofog, use p_pos
                        if np.median(diff_array) < 0:
                            p_value_one_tailed = float(p_neg) if not np.isnan(p_neg) else None
                        else:
                            p_value_one_tailed = float(p_pos) if not np.isnan(p_pos) else None
                    except:
                        p_value_one_tailed = None
        except Exception as e:
            pass
    
    stats_dict['test_type'] = test_type
    stats_dict['test_statistic'] = test_statistic
    stats_dict['p_value_two_tailed'] = p_value_two_tailed
    stats_dict['p_value_one_tailed'] = p_value_one_tailed
    stats_dict['higher_is_better'] = higher_is_better
    
    # Determine significance
    alpha = 0.05
    stats_dict['significant_two_tailed'] = p_value_two_tailed is not None and p_value_two_tailed < alpha
    stats_dict['significant_one_tailed'] = p_value_one_tailed is not None and p_value_one_tailed < alpha
    
    # Interpretation
    if p_value_one_tailed is not None:
        if higher_is_better:
            # H0: fog >= nofog, H1: nofog > fog
            if p_value_one_tailed < alpha:
                if diff_array.mean() > 0:
                    stats_dict['interpretation'] = 'NoFog significantly better than Fog'
                else:
                    stats_dict['interpretation'] = 'Fog significantly better than NoFog'
            else:
                stats_dict['interpretation'] = 'No significant difference (Fog >= NoFog)'
        else:
            # H0: fog <= nofog, H1: nofog < fog
            if p_value_one_tailed < alpha:
                if diff_array.mean() < 0:
                    stats_dict['interpretation'] = 'NoFog significantly better than Fog'
                else:
                    stats_dict['interpretation'] = 'Fog significantly better than NoFog'
            else:
                stats_dict['interpretation'] = 'No significant difference (Fog <= NoFog)'
    else:
        stats_dict['interpretation'] = 'Test could not be performed'
    
    return stats_dict


def generate_statistics_table_html(stats_data: Dict, metric: str) -> List[str]:
    """Generate HTML table rows for statistics display."""
    html_parts = []
    
    html_parts.append("<table class='stats-table'>")
    html_parts.append("<tr><th>Statistic</th><th>Fog</th><th>NoFog</th><th>Difference</th></tr>")
    html_parts.append(f"<tr><td>Mean</td><td>{stats_data['fog_mean']:.4f}</td><td>{stats_data['nofog_mean']:.4f}</td><td>{stats_data['mean_difference']:+.4f}</td></tr>")
    html_parts.append(f"<tr><td>Median</td><td>{stats_data['fog_median']:.4f}</td><td>{stats_data['nofog_median']:.4f}</td><td>{stats_data['nofog_median'] - stats_data['fog_median']:+.4f}</td></tr>")
    html_parts.append(f"<tr><td>Std Dev</td><td>{stats_data['fog_std']:.4f}</td><td>{stats_data['nofog_std']:.4f}</td><td>-</td></tr>")
    html_parts.append(f"<tr><td>N (pairs)</td><td colspan='3'>{stats_data.get('n_pairs', 'N/A')}</td></tr>")
    
    # Normality test
    if stats_data.get('normality_test_p') is not None:
        norm_p = stats_data['normality_test_p']
        is_norm = stats_data.get('is_normal', False)
        norm_class = "significant" if is_norm else "not-significant"
        html_parts.append(f"<tr><td>Normality (Shapiro-Wilk p)</td><td colspan='3' class='{norm_class}'>{norm_p:.4f} ({'normal' if is_norm else 'non-normal'})</td></tr>")
    
    # Statistical test results
    test_type = stats_data.get('test_type')
    if test_type and test_type != 'N/A':
        html_parts.append(f"<tr><td>Test Type</td><td colspan='3'><strong>{test_type.replace('_', ' ').title()}</strong></td></tr>")
        
        if stats_data.get('test_statistic') is not None:
            html_parts.append(f"<tr><td>Test Statistic</td><td colspan='3'>{stats_data['test_statistic']:.4f}</td></tr>")
        
        # Two-tailed test
        if stats_data.get('p_value_two_tailed') is not None:
            p_two = stats_data['p_value_two_tailed']
            sig_two = stats_data.get('significant_two_tailed', False)
            sig_class = "significant" if sig_two else "not-significant"
            html_parts.append(f"<tr><td>p-value (two-tailed)</td><td colspan='3' class='{sig_class}'>{p_two:.4f} {'(significant)' if sig_two else '(not significant)'}</td></tr>")
        
        # One-tailed test (ground hypothesis: fog is better)
        if stats_data.get('p_value_one_tailed') is not None:
            p_one = stats_data['p_value_one_tailed']
            sig_one = stats_data.get('significant_one_tailed', False)
            sig_class = "significant" if sig_one else "not-significant"
            higher_better = stats_data.get('higher_is_better', True)
            direction = "NoFog > Fog" if higher_better else "NoFog < Fog"
            html_parts.append(f"<tr><td>p-value (one-tailed, H1: {direction})</td><td colspan='3' class='{sig_class}'>{p_one:.4f} {'(significant)' if sig_one else '(not significant)'}</td></tr>")
        
        # Interpretation
        interpretation = stats_data.get('interpretation', 'N/A')
        html_parts.append(f"<tr><td><strong>Interpretation</strong></td><td colspan='3'><strong>{interpretation}</strong></td></tr>")
    
    html_parts.append("</table>")
    
    return html_parts


def generate_comprehensive_analysis(
    quality_scores_csv: Path,
    output_html: Path,
    exclude_participants: List[str] = None
):
    """Generate comprehensive HTML analysis report.
    
    Args:
        quality_scores_csv: Path to the quality scores CSV file
        output_html: Path where the HTML report will be saved
        exclude_participants: List of participant names to exclude from analysis
    """
    if exclude_participants is None:
        exclude_participants = []
    
    print(f"Loading quality scores from {quality_scores_csv}...")
    df = pd.read_csv(quality_scores_csv)
    
    # Extract participant and condition
    df["participant"] = df["name"].apply(lambda x: extract_participant_and_condition(x)[0])
    df["condition"] = df["name"].apply(lambda x: extract_participant_and_condition(x)[1])
    
    # Filter valid rows
    df = df[df["participant"].notna() & df["condition"].notna()].copy()
    
    # Store original count and participants before exclusion
    original_count = len(df)
    original_participants = sorted(df['participant'].unique())
    
    # Exclude specified participants
    if exclude_participants:
        print(f"Excluding participants: {exclude_participants}")
        # Match participants by full name (case-insensitive partial matching)
        excluded_mask = df['participant'].apply(
            lambda p: any(excluded.lower() in str(p).lower() for excluded in exclude_participants)
        )
        excluded_participants = sorted(df[excluded_mask]['participant'].unique())
        df = df[~excluded_mask].copy()
        print(f"Excluded {len(excluded_participants)} participant(s): {excluded_participants}")
    else:
        excluded_participants = []
    
    print(f"Found {len(df)} valid records (after exclusions)")
    print(f"Participants included: {sorted(df['participant'].unique())}")
    
    # Categorize columns
    column_categories = categorize_columns(df)
    
    # Get all numeric columns (excluding name, path, and boolean columns that should be treated specially)
    numeric_cols = []
    boolean_cols = []
    for col in df.columns:
        if col in ['name', 'path', 'participant', 'condition']:
            continue
        if df[col].dtype in ['int64', 'float64']:
            # Check if it's actually boolean (only 0/1 or True/False)
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 2 and all(v in [0, 1, True, False, 0.0, 1.0] for v in unique_vals):
                boolean_cols.append(col)
            else:
                numeric_cols.append(col)
        elif df[col].dtype == 'bool':
            boolean_cols.append(col)
    
    print(f"Analyzing {len(numeric_cols)} numeric metrics and {len(boolean_cols)} boolean metrics")
    
    # Generate charts and statistics for each metric
    charts = {}
    statistics = {}
    
    for metric in numeric_cols:
        print(f"Processing {metric}...")
        
        # Skip if all values are NaN
        if df[metric].isna().all():
            continue
        
        # Create comparison chart
        display_name = metric.replace('_', ' ').title()
        charts[f"{metric}_comparison"] = create_comparison_chart(
            df, metric, f"{display_name} - Fog vs NoFog Comparison", display_name
        )
        
        # Create boxplot
        charts[f"{metric}_boxplot"] = create_boxplot_comparison(
            df, metric, f"{display_name} - Distribution Comparison", display_name
        )
        
        # Create scatter plot
        scatter = create_scatter_comparison(
            df, metric, f"{display_name} - Fog vs NoFog Scatter", "Fog", "NoFog"
        )
        if scatter:
            charts[f"{metric}_scatter"] = scatter
        
        # Compute statistics
        stats_result = compute_statistics(df, metric)
        if stats_result:
            statistics[metric] = stats_result
    
    # Generate HTML
    html_parts = [
        "<!DOCTYPE html>",
        "<html><head>",
        "<title>Comprehensive Mesh Quality Analysis - Fog vs NoFog</title>",
        "<meta charset='utf-8'>",
        "<style>",
        "body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f5f5f5; }",
        ".container { max-width: 1400px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        "h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }",
        "h2 { color: #34495e; margin-top: 40px; border-left: 4px solid #3498db; padding-left: 15px; }",
        "h3 { color: #555; margin-top: 30px; }",
        "table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 14px; }",
        "th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }",
        "th { background-color: #3498db; color: white; font-weight: bold; }",
        "tr:nth-child(even) { background-color: #f8f9fa; }",
        "tr:hover { background-color: #e8f4f8; }",
        ".metric-section { margin: 40px 0; padding: 20px; background-color: #fafafa; border-radius: 5px; border: 1px solid #e0e0e0; }",
        ".chart-container { margin: 20px 0; text-align: center; }",
        ".chart-container img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
        ".stats-table { margin: 20px 0; }",
        ".insight-box { background-color: #e8f4f8; border-left: 4px solid #3498db; padding: 15px; margin: 20px 0; border-radius: 4px; }",
        ".summary-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }",
        ".stat-card { background-color: #fff; padding: 15px; border-radius: 5px; border: 1px solid #ddd; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }",
        ".stat-card h4 { margin: 0 0 10px 0; color: #2c3e50; }",
        ".stat-value { font-size: 24px; font-weight: bold; color: #3498db; }",
        ".nav-menu { background-color: #2c3e50; padding: 15px; border-radius: 5px; margin-bottom: 30px; }",
        ".nav-menu a { color: white; text-decoration: none; margin-right: 20px; padding: 8px 15px; border-radius: 3px; }",
        ".nav-menu a:hover { background-color: #34495e; }",
        ".category-header { background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin-top: 30px; }",
        ".significant { color: #e74c3c; font-weight: bold; }",
        ".not-significant { color: #95a5a6; }",
        "</style>",
        "</head><body>",
        "<div class='container'>",
        "<h1>Comprehensive Mesh Quality Analysis: Fog vs NoFog</h1>",
        "<div class='nav-menu'>",
        "<a href='#summary'>Summary</a>",
        "<a href='#quality-scores'>Quality Scores</a>",
        "<a href='#geometry'>Geometry Metrics</a>",
        "<a href='#smoothness'>Smoothness Metrics</a>",
        "<a href='#completeness'>Completeness Metrics</a>",
        "<a href='#color'>Color Metrics</a>",
        "<a href='#topology'>Topology Metrics</a>",
        "<a href='#size'>Size Metrics</a>",
        "<a href='#statistics'>Statistical Summary</a>",
        "</div>",
    ]
    
    # Summary section
    html_parts.append("<div id='summary'>")
    html_parts.append("<h2>Executive Summary</h2>")
    
    # Overall statistics
    participants = sorted(df['participant'].unique())
    html_parts.append(f"<p><strong>Total Participants (included):</strong> {len(participants)}</p>")
    html_parts.append(f"<p><strong>Total Records:</strong> {len(df)} ({len(df[df['condition']=='fog'])} fog, {len(df[df['condition']=='nofog'])} nofog)</p>")
    
    # Show excluded participants if any
    if excluded_participants:
        html_parts.append("<div class='insight-box' style='background-color: #fff3cd; border-left-color: #ffc107;'>")
        html_parts.append(f"<h3>Excluded Participants</h3>")
        html_parts.append(f"<p><strong>Excluded:</strong> {', '.join(excluded_participants)}</p>")
        html_parts.append(f"<p><strong>Total excluded:</strong> {len(excluded_participants)} participant(s), {original_count - len(df)} record(s)</p>")
        html_parts.append("</div>")
    
    # Statistical testing information
    html_parts.append("<div class='insight-box'>")
    html_parts.append("<h3>Statistical Testing Methodology</h3>")
    html_parts.append("<p><strong>Ground Hypothesis:</strong> Fog condition produces better/higher quality meshes than NoFog condition.</p>")
    html_parts.append("<p><strong>Null Hypothesis (H0):</strong>")
    html_parts.append("<ul>")
    html_parts.append("<li>For metrics where <em>higher is better</em>: H0: Fog ≥ NoFog (Fog is better or equal)</li>")
    html_parts.append("<li>For metrics where <em>lower is better</em>: H0: Fog ≤ NoFog (Fog is better or equal)</li>")
    html_parts.append("</ul>")
    html_parts.append("<p><strong>Alternative Hypothesis (H1):</strong> NoFog is significantly better than Fog.</p>")
    html_parts.append("<p><strong>Testing Approach:</strong>")
    html_parts.append("<ul>")
    html_parts.append("<li>Normality tested using Shapiro-Wilk test on paired differences</li>")
    html_parts.append("<li>If normal: Paired t-test (parametric)</li>")
    html_parts.append("<li>If non-normal: Wilcoxon signed-rank test (non-parametric)</li>")
    html_parts.append("<li>Both one-tailed (directional) and two-tailed (general difference) tests performed</li>")
    html_parts.append("<li>Significance level: α = 0.05</li>")
    html_parts.append("</ul>")
    html_parts.append("</div>")
    
    # Key quality score summary (using Q_raw as main metric)
    if 'Q_raw' in statistics:
        q_stats = statistics['Q_raw']
        html_parts.append("<div class='summary-stats'>")
        html_parts.append(f"<div class='stat-card'><h4>Overall Quality (Q_raw)</h4>")
        html_parts.append(f"<div class='stat-value'>Fog: {q_stats['fog_mean']:.3f}</div>")
        html_parts.append(f"<div class='stat-value'>NoFog: {q_stats['nofog_mean']:.3f}</div>")
        html_parts.append(f"<div>Difference: {q_stats['mean_difference']:+.3f} ({q_stats['percent_change']:+.1f}%)</div>")
        if q_stats.get('p_value_one_tailed') is not None:
            sig_class = "significant" if q_stats.get('significant_one_tailed', False) else "not-significant"
            html_parts.append(f"<div class='{sig_class}'>One-tailed p-value: {q_stats['p_value_one_tailed']:.4f}</div>")
            html_parts.append(f"<div class='{sig_class}'><strong>{q_stats.get('interpretation', 'N/A')}</strong></div>")
        html_parts.append("</div>")
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Quality Scores Section
    html_parts.append("<div id='quality-scores'>")
    html_parts.append("<h2>Quality Scores</h2>")
    html_parts.append("<div class='category-header'><p>These are the main quality score metrics computed from the mesh analysis.</p></div>")
    
    quality_score_cols = column_categories.get('quality_scores', [])
    for metric in quality_score_cols:
        if metric not in numeric_cols:
            continue
        
        html_parts.append(f"<div class='metric-section' id='{metric}'>")
        html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
        
        # Add statistics table
        if metric in statistics:
            stats_data = statistics[metric]
            html_parts.extend(generate_statistics_table_html(stats_data, metric))
        
        # Add charts
        if f"{metric}_comparison" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_comparison']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_boxplot" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_boxplot']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_scatter" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_scatter']}'/>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Geometry Metrics Section
    html_parts.append("<div id='geometry'>")
    html_parts.append("<h2>Geometry Metrics</h2>")
    html_parts.append("<div class='category-header'><p>Geometric properties of the meshes including aspect ratios, skewness, and topology issues.</p></div>")
    
    geometry_cols = column_categories.get('geometry_metrics', [])
    for metric in geometry_cols:
        if metric not in numeric_cols:
            continue
        
        html_parts.append(f"<div class='metric-section' id='{metric}'>")
        html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
        
        if metric in statistics:
            stats_data = statistics[metric]
            html_parts.append("<table class='stats-table'>")
            html_parts.append("<tr><th>Statistic</th><th>Fog</th><th>NoFog</th><th>Difference</th></tr>")
            html_parts.append(f"<tr><td>Mean</td><td>{stats_data['fog_mean']:.4f}</td><td>{stats_data['nofog_mean']:.4f}</td><td>{stats_data['mean_difference']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Median</td><td>{stats_data['fog_median']:.4f}</td><td>{stats_data['nofog_median']:.4f}</td><td>{stats_data['nofog_median'] - stats_data['fog_median']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std Dev</td><td>{stats_data['fog_std']:.4f}</td><td>{stats_data['nofog_std']:.4f}</td><td>-</td></tr>")
            if 'p_value' in stats_data:
                sig_class = "significant" if stats_data['significant'] else "not-significant"
                html_parts.append(f"<tr><td>p-value</td><td colspan='3' class='{sig_class}'>{stats_data['p_value']:.4f} {'(significant)' if stats_data['significant'] else '(not significant)'}</td></tr>")
            html_parts.append("</table>")
        
        if f"{metric}_comparison" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_comparison']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_boxplot" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_boxplot']}'/>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Smoothness Metrics Section
    html_parts.append("<div id='smoothness'>")
    html_parts.append("<h2>Smoothness Metrics</h2>")
    html_parts.append("<div class='category-header'><p>Measures of surface smoothness including normal deviations and dihedral angles.</p></div>")
    
    smoothness_cols = column_categories.get('smoothness_metrics', [])
    for metric in smoothness_cols:
        if metric not in numeric_cols:
            continue
        
        html_parts.append(f"<div class='metric-section' id='{metric}'>")
        html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
        
        if metric in statistics:
            stats_data = statistics[metric]
            html_parts.append("<table class='stats-table'>")
            html_parts.append("<tr><th>Statistic</th><th>Fog</th><th>NoFog</th><th>Difference</th></tr>")
            html_parts.append(f"<tr><td>Mean</td><td>{stats_data['fog_mean']:.4f}</td><td>{stats_data['nofog_mean']:.4f}</td><td>{stats_data['mean_difference']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Median</td><td>{stats_data['fog_median']:.4f}</td><td>{stats_data['nofog_median']:.4f}</td><td>{stats_data['nofog_median'] - stats_data['fog_median']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std Dev</td><td>{stats_data['fog_std']:.4f}</td><td>{stats_data['nofog_std']:.4f}</td><td>-</td></tr>")
            if 'p_value' in stats_data:
                sig_class = "significant" if stats_data['significant'] else "not-significant"
                html_parts.append(f"<tr><td>p-value</td><td colspan='3' class='{sig_class}'>{stats_data['p_value']:.4f} {'(significant)' if stats_data['significant'] else '(not significant)'}</td></tr>")
            html_parts.append("</table>")
        
        if f"{metric}_comparison" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_comparison']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_boxplot" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_boxplot']}'/>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Completeness Metrics Section
    html_parts.append("<div id='completeness'>")
    html_parts.append("<h2>Completeness Metrics</h2>")
    html_parts.append("<div class='category-header'><p>Measures of mesh completeness including component connectivity and vertex density.</p></div>")
    
    completeness_cols = column_categories.get('completeness_metrics', [])
    for metric in completeness_cols:
        if metric not in numeric_cols:
            continue
        
        html_parts.append(f"<div class='metric-section' id='{metric}'>")
        html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
        
        if metric in statistics:
            stats_data = statistics[metric]
            html_parts.append("<table class='stats-table'>")
            html_parts.append("<tr><th>Statistic</th><th>Fog</th><th>NoFog</th><th>Difference</th></tr>")
            html_parts.append(f"<tr><td>Mean</td><td>{stats_data['fog_mean']:.4f}</td><td>{stats_data['nofog_mean']:.4f}</td><td>{stats_data['mean_difference']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Median</td><td>{stats_data['fog_median']:.4f}</td><td>{stats_data['nofog_median']:.4f}</td><td>{stats_data['nofog_median'] - stats_data['fog_median']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std Dev</td><td>{stats_data['fog_std']:.4f}</td><td>{stats_data['nofog_std']:.4f}</td><td>-</td></tr>")
            if 'p_value' in stats_data:
                sig_class = "significant" if stats_data['significant'] else "not-significant"
                html_parts.append(f"<tr><td>p-value</td><td colspan='3' class='{sig_class}'>{stats_data['p_value']:.4f} {'(significant)' if stats_data['significant'] else '(not significant)'}</td></tr>")
            html_parts.append("</table>")
        
        if f"{metric}_comparison" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_comparison']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_boxplot" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_boxplot']}'/>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Color Metrics Section
    html_parts.append("<div id='color'>")
    html_parts.append("<h2>Color Metrics</h2>")
    html_parts.append("<div class='category-header'><p>Color-related properties of the meshes.</p></div>")
    
    color_cols = column_categories.get('color_metrics', [])
    for metric in color_cols:
        if metric not in numeric_cols:
            continue
        
        html_parts.append(f"<div class='metric-section' id='{metric}'>")
        html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
        
        if metric in statistics:
            stats_data = statistics[metric]
            html_parts.append("<table class='stats-table'>")
            html_parts.append("<tr><th>Statistic</th><th>Fog</th><th>NoFog</th><th>Difference</th></tr>")
            html_parts.append(f"<tr><td>Mean</td><td>{stats_data['fog_mean']:.4f}</td><td>{stats_data['nofog_mean']:.4f}</td><td>{stats_data['mean_difference']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Median</td><td>{stats_data['fog_median']:.4f}</td><td>{stats_data['nofog_median']:.4f}</td><td>{stats_data['nofog_median'] - stats_data['fog_median']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std Dev</td><td>{stats_data['fog_std']:.4f}</td><td>{stats_data['nofog_std']:.4f}</td><td>-</td></tr>")
            if 'p_value' in stats_data:
                sig_class = "significant" if stats_data['significant'] else "not-significant"
                html_parts.append(f"<tr><td>p-value</td><td colspan='3' class='{sig_class}'>{stats_data['p_value']:.4f} {'(significant)' if stats_data['significant'] else '(not significant)'}</td></tr>")
            html_parts.append("</table>")
        
        if f"{metric}_comparison" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_comparison']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_boxplot" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_boxplot']}'/>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Topology Metrics Section
    html_parts.append("<div id='topology'>")
    html_parts.append("<h2>Topology Metrics</h2>")
    html_parts.append("<div class='category-header'><p>Topological properties including manifold and watertight status.</p></div>")
    
    topology_cols = column_categories.get('topology_metrics', [])
    for metric in topology_cols:
        if metric not in numeric_cols:
            continue
        
        html_parts.append(f"<div class='metric-section' id='{metric}'>")
        html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
        
        if metric in statistics:
            stats_data = statistics[metric]
            html_parts.append("<table class='stats-table'>")
            html_parts.append("<tr><th>Statistic</th><th>Fog</th><th>NoFog</th><th>Difference</th></tr>")
            html_parts.append(f"<tr><td>Mean</td><td>{stats_data['fog_mean']:.4f}</td><td>{stats_data['nofog_mean']:.4f}</td><td>{stats_data['mean_difference']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Median</td><td>{stats_data['fog_median']:.4f}</td><td>{stats_data['nofog_median']:.4f}</td><td>{stats_data['nofog_median'] - stats_data['fog_median']:+.4f}</td></tr>")
            html_parts.append(f"<tr><td>Std Dev</td><td>{stats_data['fog_std']:.4f}</td><td>{stats_data['nofog_std']:.4f}</td><td>-</td></tr>")
            if 'p_value' in stats_data:
                sig_class = "significant" if stats_data['significant'] else "not-significant"
                html_parts.append(f"<tr><td>p-value</td><td colspan='3' class='{sig_class}'>{stats_data['p_value']:.4f} {'(significant)' if stats_data['significant'] else '(not significant)'}</td></tr>")
            html_parts.append("</table>")
        
        if f"{metric}_comparison" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_comparison']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_boxplot" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_boxplot']}'/>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Size Metrics Section
    html_parts.append("<div id='size'>")
    html_parts.append("<h2>Size Metrics</h2>")
    html_parts.append("<div class='category-header'><p>Mesh size properties including vertex and triangle counts.</p></div>")
    
    size_cols = column_categories.get('size_metrics', [])
    for metric in size_cols:
        if metric not in numeric_cols:
            continue
        
        html_parts.append(f"<div class='metric-section' id='{metric}'>")
        html_parts.append(f"<h3>{metric.replace('_', ' ').title()}</h3>")
        
        if metric in statistics:
            stats_data = statistics[metric]
            html_parts.extend(generate_statistics_table_html(stats_data, metric))
        
        if f"{metric}_comparison" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_comparison']}'/>")
            html_parts.append("</div>")
        
        if f"{metric}_boxplot" in charts:
            html_parts.append("<div class='chart-container'>")
            html_parts.append(f"<img src='data:image/png;base64,{charts[f'{metric}_boxplot']}'/>")
            html_parts.append("</div>")
        
        html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Statistical Summary Section
    html_parts.append("<div id='statistics'>")
    html_parts.append("<h2>Statistical Summary</h2>")
    html_parts.append("<div class='category-header'><p>Complete statistical comparison table for all metrics.</p></div>")
    
    html_parts.append("<table class='stats-table'>")
    html_parts.append("<tr><th>Metric</th><th>Fog Mean</th><th>NoFog Mean</th><th>Difference</th><th>% Change</th><th>Test Type</th><th>Normality</th><th>p (two-tailed)</th><th>p (one-tailed)</th><th>Significant</th><th>Interpretation</th></tr>")
    
    for metric in sorted(statistics.keys()):
        stats_data = statistics[metric]
        
        # Determine significance (prefer one-tailed for hypothesis testing)
        sig_one = stats_data.get('significant_one_tailed', False)
        sig_two = stats_data.get('significant_two_tailed', False)
        sig_class = "significant" if (sig_one or sig_two) else "not-significant"
        sig_str = "Yes" if (sig_one or sig_two) else "No"
        
        # Test type
        test_type = stats_data.get('test_type')
        test_type_str = test_type.replace('_', ' ').title() if test_type and test_type != 'N/A' else 'N/A'
        
        # Normality
        is_norm = stats_data.get('is_normal', False)
        norm_str = "Normal" if is_norm else "Non-normal"
        norm_p = stats_data.get('normality_test_p')
        norm_str_full = f"{norm_str}" + (f" (p={norm_p:.3f})" if norm_p is not None else "")
        
        # P-values
        p_two = stats_data.get('p_value_two_tailed')
        p_two_str = f"{p_two:.4f}" if p_two is not None else "N/A"
        
        p_one = stats_data.get('p_value_one_tailed')
        p_one_str = f"{p_one:.4f}" if p_one is not None else "N/A"
        
        # Interpretation
        interpretation = stats_data.get('interpretation', 'N/A')
        
        html_parts.append(f"<tr>")
        html_parts.append(f"<td><strong>{metric}</strong></td>")
        html_parts.append(f"<td>{stats_data['fog_mean']:.4f}</td>")
        html_parts.append(f"<td>{stats_data['nofog_mean']:.4f}</td>")
        html_parts.append(f"<td>{stats_data['mean_difference']:+.4f}</td>")
        html_parts.append(f"<td>{stats_data['percent_change']:+.2f}%</td>")
        html_parts.append(f"<td>{test_type_str}</td>")
        html_parts.append(f"<td>{norm_str_full}</td>")
        html_parts.append(f"<td>{p_two_str}</td>")
        html_parts.append(f"<td class='{sig_class}'>{p_one_str}</td>")
        html_parts.append(f"<td class='{sig_class}'>{sig_str}</td>")
        html_parts.append(f"<td><em>{interpretation}</em></td>")
        html_parts.append(f"</tr>")
    
    html_parts.append("</table>")
    
    # Key Insights
    html_parts.append("<div class='insight-box'>")
    html_parts.append("<h3>Key Insights</h3>")
    html_parts.append("<ul>")
    
    # Find metrics with significant differences (using one-tailed test for hypothesis testing)
    significant_one_tailed = [m for m, s in statistics.items() if s.get('significant_one_tailed', False)]
    significant_two_tailed = [m for m, s in statistics.items() if s.get('significant_two_tailed', False)]
    
    if significant_one_tailed:
        html_parts.append(f"<li><strong>Significant Differences (One-tailed test, α=0.05):</strong> {len(significant_one_tailed)} metrics show statistically significant differences rejecting the null hypothesis (fog is better).</li>")
        for metric in significant_one_tailed[:5]:  # Show top 5
            stats_data = statistics[metric]
            interpretation = stats_data.get('interpretation', 'N/A')
            p_val = stats_data.get('p_value_one_tailed', 'N/A')
            p_val_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
            html_parts.append(f"<li><strong>{metric}:</strong> {interpretation} (p={p_val_str})</li>")
    
    if significant_two_tailed:
        html_parts.append(f"<li><strong>Significant Differences (Two-tailed test, α=0.05):</strong> {len(significant_two_tailed)} metrics show statistically significant differences between fog and nofog conditions.</li>")
    
    # Overall quality insight (using Q_raw as main metric)
    if 'Q_raw' in statistics:
        q_stats = statistics['Q_raw']
        interpretation = q_stats.get('interpretation', 'N/A')
        p_val = q_stats.get('p_value_one_tailed', 'N/A')
        p_val_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
        html_parts.append(f"<li><strong>Overall Quality (Q_raw):</strong> {interpretation} (p={p_val_str}, difference={q_stats['mean_difference']:+.3f})</li>")
    
    # Size insights
    if 'num_vertices' in statistics:
        size_stats = statistics['num_vertices']
        direction = "more" if size_stats['mean_difference'] > 0 else "fewer"
        html_parts.append(f"<li><strong>Mesh Size:</strong> NoFog meshes have {direction} vertices on average ({size_stats['mean_difference']:+.0f} difference).</li>")
    
    html_parts.append("</ul>")
    html_parts.append("</div>")
    
    html_parts.append("</div>")
    
    # Close container and body
    html_parts.append("</div>")
    html_parts.append("</body></html>")
    
    # Write HTML
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"HTML report saved to: {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive HTML analysis report for mesh quality scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate report with default settings
  python generate_comprehensive_quality_analysis.py

  # Exclude specific participants
  python generate_comprehensive_quality_analysis.py \\
      --exclude-participants "Samuel Thompson" "Maria Parisopoulo"

  # Exclude multiple participants and specify custom paths
  python generate_comprehensive_quality_analysis.py \\
      --quality-scores path/to/scores.csv \\
      --output path/to/report.html \\
      --exclude-participants "Participant 1" "Participant 2"
        """
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
        default=Path("analysis/mesh_quality_batch/comprehensive_quality_analysis.html"),
        help="Output HTML path"
    )
    parser.add_argument(
        "--exclude-participants",
        type=str,
        nargs="+",
        default=[],
        help="List of participant names to exclude from analysis (partial name matching, case-insensitive). "
             "Example: --exclude-participants 'Samuel Thompson' 'Maria Parisopoulo'"
    )
    
    args = parser.parse_args()
    
    generate_comprehensive_analysis(
        args.quality_scores,
        args.output,
        exclude_participants=args.exclude_participants
    )


if __name__ == "__main__":
    main()

