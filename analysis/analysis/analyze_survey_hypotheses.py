#!/usr/bin/env python3
"""
Statistical Analysis for Survey Hypotheses (H4-H7)

This script performs statistical analyses comparing Assisted (sphere) vs Unassisted (nosphere) conditions:
- H4: Flow/Engagement (FSS_FlowTotal)
- H5: Perceived Performance (TLX_4)
- H6: Realism Trade-off (IPQ Realism subscale)
- H7: Workload Redistribution (TLX_2 Physical Demand, TLX_6 Frustration)
- Descriptive Statistics: SUS Score

All tests use Wilcoxon Signed-Rank Tests (non-parametric paired t-tests) due to small sample size (N=14).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats


def load_data(excel_path: Path) -> pd.DataFrame:
    """Load survey data from Excel file."""
    print(f"[info] Loading data from: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"[info] Loaded {len(df)} participants")
    return df


def calculate_ipq_realism(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate IPQ Realism subscale for both conditions.
    
    Steps:
    1. Reverse code IPQ_11 (8 - IPQ_11) since it's anchored opposite to items 12-14
    2. Calculate mean of IPQ_11_REV, IPQ_12, IPQ_13, IPQ_14 for each condition
    """
    df = df.copy()
    
    # Step 1: Reverse code IPQ_11
    df['IPQ_11_sphere_REV'] = 8 - df['IPQ_11_sphere']
    df['IPQ_11_nosphere_REV'] = 8 - df['IPQ_11_nosphere']
    
    # Step 2: Calculate Realism subscale mean
    # Assisted (sphere)
    realism_items_sphere = ['IPQ_11_sphere_REV', 'IPQ_12_sphere', 'IPQ_13_sphere', 'IPQ_14_sphere']
    df['IPQ_Realism_sphere'] = df[realism_items_sphere].mean(axis=1)
    
    # Unassisted (nosphere)
    realism_items_nosphere = ['IPQ_11_nosphere_REV', 'IPQ_12_nosphere', 'IPQ_13_nosphere', 'IPQ_14_nosphere']
    df['IPQ_Realism_nosphere'] = df[realism_items_nosphere].mean(axis=1)
    
    print("[info] Calculated IPQ Realism subscale for both conditions")
    return df


def perform_wilcoxon_test(assisted: pd.Series, unassisted: pd.Series, alternative: str, test_name: str) -> dict:
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Args:
        assisted: Series of assisted (sphere) condition values
        unassisted: Series of unassisted (nosphere) condition values
        alternative: 'greater', 'less', or 'two-sided'
        test_name: Descriptive name for the test
    
    Returns:
        Dictionary with test results
    """
    # Remove missing values (pairwise)
    paired_data = pd.DataFrame({
        'assisted': assisted,
        'unassisted': unassisted
    }).dropna()
    
    if len(paired_data) < 2:
        return {
            'test_name': test_name,
            'n': len(paired_data),
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'assisted_mean': np.nan,
            'assisted_std': np.nan,
            'assisted_median': np.nan,
            'unassisted_mean': np.nan,
            'unassisted_std': np.nan,
            'unassisted_median': np.nan,
            'error': 'Insufficient data'
        }
    
    assisted_vals = paired_data['assisted'].values
    unassisted_vals = paired_data['unassisted'].values
    
    # Perform Wilcoxon signed-rank test
    try:
        stat, p_value = stats.wilcoxon(assisted_vals, unassisted_vals, alternative=alternative)
    except Exception as e:
        return {
            'test_name': test_name,
            'n': len(paired_data),
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'assisted_mean': np.nan,
            'assisted_std': np.nan,
            'assisted_median': np.nan,
            'unassisted_mean': np.nan,
            'unassisted_std': np.nan,
            'unassisted_median': np.nan,
            'error': str(e)
        }
    
    # Descriptive statistics
    assisted_mean = np.mean(assisted_vals)
    assisted_std = np.std(assisted_vals, ddof=1)
    assisted_median = np.median(assisted_vals)
    
    unassisted_mean = np.mean(unassisted_vals)
    unassisted_std = np.std(unassisted_vals, ddof=1)
    unassisted_median = np.median(unassisted_vals)
    
    return {
        'test_name': test_name,
        'n': len(paired_data),
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'assisted_mean': assisted_mean,
        'assisted_std': assisted_std,
        'assisted_median': assisted_median,
        'unassisted_mean': unassisted_mean,
        'unassisted_std': unassisted_std,
        'unassisted_median': unassisted_median,
        'alternative': alternative
    }


def analyze_hypotheses(df: pd.DataFrame) -> dict:
    """Perform all statistical tests for hypotheses H4-H7."""
    results = {}
    
    # Test 1: H4 - Engagement (Flow)
    print("\n[info] Test 1: H4 - Engagement (Flow Total)")
    print("  Hypothesis: Assisted > Unassisted")
    results['H4_Flow'] = perform_wilcoxon_test(
        df['FSS_FlowTotal_sphere'],
        df['FSS_FlowTotal_nosphere'],
        alternative='greater',
        test_name='H4: Flow Total (Assisted > Unassisted)'
    )
    
    # Test 2: H5 - Perceived Performance
    print("\n[info] Test 2: H5 - Perceived Performance (TLX_4)")
    print("  Hypothesis: Assisted < Unassisted (lower is better)")
    results['H5_Performance'] = perform_wilcoxon_test(
        df['TLX_4_sphere'],
        df['TLX_4_nosphere'],
        alternative='less',
        test_name='H5: Perceived Performance (Assisted < Unassisted)'
    )
    
    # Test 3: H6 - Realism Trade-off
    print("\n[info] Test 3: H6 - Realism Trade-off (IPQ Realism)")
    print("  Hypothesis: Assisted < Unassisted (overlay reduces realism)")
    results['H6_Realism'] = perform_wilcoxon_test(
        df['IPQ_Realism_sphere'],
        df['IPQ_Realism_nosphere'],
        alternative='less',
        test_name='H6: IPQ Realism (Assisted < Unassisted)'
    )
    
    # Test 4: H7 - Workload Redistribution
    print("\n[info] Test 4: H7 - Workload Redistribution")
    
    # Physical Demand
    print("  H7a: Physical Demand (TLX_2)")
    print("    Expectation: Assisted > Unassisted (more physically demanding)")
    results['H7a_Physical'] = perform_wilcoxon_test(
        df['TLX_2_sphere'],
        df['TLX_2_nosphere'],
        alternative='greater',
        test_name='H7a: Physical Demand (Assisted > Unassisted)'
    )
    
    # Frustration
    print("  H7b: Frustration (TLX_6)")
    print("    Expectation: Assisted < Unassisted (less frustrating)")
    results['H7b_Frustration'] = perform_wilcoxon_test(
        df['TLX_6_sphere'],
        df['TLX_6_nosphere'],
        alternative='less',
        test_name='H7b: Frustration (Assisted < Unassisted)'
    )
    
    # Mental Demand (additional analysis)
    print("  Additional: Mental Demand (TLX_1)")
    results['Mental_Demand'] = perform_wilcoxon_test(
        df['TLX_1_sphere'],
        df['TLX_1_nosphere'],
        alternative='two-sided',
        test_name='Mental Demand (two-sided)'
    )
    
    return results


def calculate_sus_descriptives(df: pd.DataFrame) -> dict:
    """Calculate descriptive statistics for SUS Score (Assisted condition only)."""
    sus_scores = df['SUS_Score_sphere'].dropna()
    
    if len(sus_scores) == 0:
        return {
            'n': 0,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan,
            'min': np.nan,
            'max': np.nan,
            'benchmark': 'No data'
        }
    
    mean = sus_scores.mean()
    std = sus_scores.std(ddof=1)
    median = sus_scores.median()
    
    # Benchmark interpretation
    if mean > 80:
        benchmark = 'Excellent'
    elif mean > 68:
        benchmark = 'Above Average'
    else:
        benchmark = 'Below Average'
    
    return {
        'n': len(sus_scores),
        'mean': mean,
        'std': std,
        'median': median,
        'min': sus_scores.min(),
        'max': sus_scores.max(),
        'benchmark': benchmark
    }


def generate_report(results: dict, sus_stats: dict, output_path: Path) -> None:
    """Generate a comprehensive text report of all statistical analyses."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL ANALYSIS REPORT: SURVEY HYPOTHESES (H4-H7)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("METHODOLOGY\n")
        f.write("-" * 80 + "\n")
        f.write("All statistical tests use Wilcoxon Signed-Rank Tests (non-parametric paired t-tests)\n")
        f.write("due to small sample size (N=14).\n\n")
        f.write("One-tailed tests are used when directional hypotheses are specified.\n\n")
        
        # Test 1: H4
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEST 1: H4 - ENGAGEMENT (FLOW)\n")
        f.write("=" * 80 + "\n")
        h4 = results['H4_Flow']
        f.write(f"Hypothesis: Assisted > Unassisted\n")
        f.write(f"Variables: FSS_FlowTotal_sphere vs. FSS_FlowTotal_nosphere\n")
        f.write(f"Test: Paired Wilcoxon Signed-Rank (Alternative: Greater)\n\n")
        f.write(f"Sample Size: N = {h4['n']}\n")
        f.write(f"\nDescriptive Statistics:\n")
        f.write(f"  Assisted (Sphere):\n")
        f.write(f"    Mean = {h4['assisted_mean']:.2f}\n")
        f.write(f"    SD = {h4['assisted_std']:.2f}\n")
        f.write(f"    Median = {h4['assisted_median']:.2f}\n")
        f.write(f"  Unassisted (Nosphere):\n")
        f.write(f"    Mean = {h4['unassisted_mean']:.2f}\n")
        f.write(f"    SD = {h4['unassisted_std']:.2f}\n")
        f.write(f"    Median = {h4['unassisted_median']:.2f}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Statistic = {h4['statistic']:.4f}\n")
        f.write(f"  p-value = {h4['p_value']:.4f}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if h4['significant'] else 'No'}\n")
        if h4['significant']:
            f.write(f"  *** p < 0.05\n")
        
        # Test 2: H5
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEST 2: H5 - PERCEIVED PERFORMANCE\n")
        f.write("=" * 80 + "\n")
        h5 = results['H5_Performance']
        f.write(f"Hypothesis: Assisted < Unassisted (Lower score is better)\n")
        f.write(f"Variables: TLX_4_sphere vs. TLX_4_nosphere\n")
        f.write(f"Test: Paired Wilcoxon Signed-Rank (Alternative: Less)\n")
        f.write(f"Note: Anchor 0 = 'Excellent', 100 = 'Failure'. Lower values = better performance.\n\n")
        f.write(f"Sample Size: N = {h5['n']}\n")
        f.write(f"\nDescriptive Statistics:\n")
        f.write(f"  Assisted (Sphere):\n")
        f.write(f"    Mean = {h5['assisted_mean']:.2f}\n")
        f.write(f"    SD = {h5['assisted_std']:.2f}\n")
        f.write(f"    Median = {h5['assisted_median']:.2f}\n")
        f.write(f"  Unassisted (Nosphere):\n")
        f.write(f"    Mean = {h5['unassisted_mean']:.2f}\n")
        f.write(f"    SD = {h5['unassisted_std']:.2f}\n")
        f.write(f"    Median = {h5['unassisted_median']:.2f}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Statistic = {h5['statistic']:.4f}\n")
        f.write(f"  p-value = {h5['p_value']:.4f}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if h5['significant'] else 'No'}\n")
        if h5['significant']:
            f.write(f"  *** p < 0.05\n")
        
        # Test 3: H6
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEST 3: H6 - REALISM TRADE-OFF\n")
        f.write("=" * 80 + "\n")
        h6 = results['H6_Realism']
        f.write(f"Hypothesis: Assisted < Unassisted (Overlay reduces realism)\n")
        f.write(f"Variables: IPQ Realism Subscale (Items 11-14)\n")
        f.write(f"  - Item 11 was reverse coded (8 - IPQ_11)\n")
        f.write(f"  - Subscale = Mean of IPQ_11_REV, IPQ_12, IPQ_13, IPQ_14\n")
        f.write(f"Test: Paired Wilcoxon Signed-Rank (Alternative: Less)\n\n")
        f.write(f"Sample Size: N = {h6['n']}\n")
        f.write(f"\nDescriptive Statistics:\n")
        f.write(f"  Assisted (Sphere):\n")
        f.write(f"    Mean = {h6['assisted_mean']:.2f}\n")
        f.write(f"    SD = {h6['assisted_std']:.2f}\n")
        f.write(f"    Median = {h6['assisted_median']:.2f}\n")
        f.write(f"  Unassisted (Nosphere):\n")
        f.write(f"    Mean = {h6['unassisted_mean']:.2f}\n")
        f.write(f"    SD = {h6['unassisted_std']:.2f}\n")
        f.write(f"    Median = {h6['unassisted_median']:.2f}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Statistic = {h6['statistic']:.4f}\n")
        f.write(f"  p-value = {h6['p_value']:.4f}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if h6['significant'] else 'No'}\n")
        if h6['significant']:
            f.write(f"  *** p < 0.05\n")
        
        # Test 4: H7
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEST 4: H7 - WORKLOAD REDISTRIBUTION\n")
        f.write("=" * 80 + "\n")
        
        # H7a: Physical Demand
        h7a = results['H7a_Physical']
        f.write(f"\nH7a: Physical Demand (TLX_2)\n")
        f.write(f"Expectation: Assisted > Unassisted (Assisted is more physically demanding)\n")
        f.write(f"Variables: TLX_2_sphere vs. TLX_2_nosphere\n")
        f.write(f"Test: Paired Wilcoxon Signed-Rank (Alternative: Greater)\n\n")
        f.write(f"Sample Size: N = {h7a['n']}\n")
        f.write(f"\nDescriptive Statistics:\n")
        f.write(f"  Assisted (Sphere):\n")
        f.write(f"    Mean = {h7a['assisted_mean']:.2f}\n")
        f.write(f"    SD = {h7a['assisted_std']:.2f}\n")
        f.write(f"    Median = {h7a['assisted_median']:.2f}\n")
        f.write(f"  Unassisted (Nosphere):\n")
        f.write(f"    Mean = {h7a['unassisted_mean']:.2f}\n")
        f.write(f"    SD = {h7a['unassisted_std']:.2f}\n")
        f.write(f"    Median = {h7a['unassisted_median']:.2f}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Statistic = {h7a['statistic']:.4f}\n")
        f.write(f"  p-value = {h7a['p_value']:.4f}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if h7a['significant'] else 'No'}\n")
        if h7a['significant']:
            f.write(f"  *** p < 0.05\n")
        
        # H7b: Frustration
        h7b = results['H7b_Frustration']
        f.write(f"\nH7b: Frustration (TLX_6)\n")
        f.write(f"Expectation: Assisted < Unassisted (Assisted is less frustrating)\n")
        f.write(f"Variables: TLX_6_sphere vs. TLX_6_nosphere\n")
        f.write(f"Test: Paired Wilcoxon Signed-Rank (Alternative: Less)\n\n")
        f.write(f"Sample Size: N = {h7b['n']}\n")
        f.write(f"\nDescriptive Statistics:\n")
        f.write(f"  Assisted (Sphere):\n")
        f.write(f"    Mean = {h7b['assisted_mean']:.2f}\n")
        f.write(f"    SD = {h7b['assisted_std']:.2f}\n")
        f.write(f"    Median = {h7b['assisted_median']:.2f}\n")
        f.write(f"  Unassisted (Nosphere):\n")
        f.write(f"    Mean = {h7b['unassisted_mean']:.2f}\n")
        f.write(f"    SD = {h7b['unassisted_std']:.2f}\n")
        f.write(f"    Median = {h7b['unassisted_median']:.2f}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Statistic = {h7b['statistic']:.4f}\n")
        f.write(f"  p-value = {h7b['p_value']:.4f}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if h7b['significant'] else 'No'}\n")
        if h7b['significant']:
            f.write(f"  *** p < 0.05\n")
        
        # Additional: Mental Demand
        mental = results['Mental_Demand']
        f.write(f"\nAdditional Analysis: Mental Demand (TLX_1)\n")
        f.write(f"Variables: TLX_1_sphere vs. TLX_1_nosphere\n")
        f.write(f"Test: Paired Wilcoxon Signed-Rank (Two-sided)\n\n")
        f.write(f"Sample Size: N = {mental['n']}\n")
        f.write(f"\nDescriptive Statistics:\n")
        f.write(f"  Assisted (Sphere):\n")
        f.write(f"    Mean = {mental['assisted_mean']:.2f}\n")
        f.write(f"    SD = {mental['assisted_std']:.2f}\n")
        f.write(f"    Median = {mental['assisted_median']:.2f}\n")
        f.write(f"  Unassisted (Nosphere):\n")
        f.write(f"    Mean = {mental['unassisted_mean']:.2f}\n")
        f.write(f"    SD = {mental['unassisted_std']:.2f}\n")
        f.write(f"    Median = {mental['unassisted_median']:.2f}\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Statistic = {mental['statistic']:.4f}\n")
        f.write(f"  p-value = {mental['p_value']:.4f}\n")
        f.write(f"  Significant (p < 0.05): {'Yes' if mental['significant'] else 'No'}\n")
        if mental['significant']:
            f.write(f"  *** p < 0.05\n")
        
        # SUS Descriptive Statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("DESCRIPTIVE STATISTICS: SYSTEM USABILITY SCALE (SUS)\n")
        f.write("=" * 80 + "\n")
        f.write(f"Metric: SUS_Score_sphere (Assisted condition only)\n\n")
        f.write(f"Sample Size: N = {sus_stats['n']}\n")
        f.write(f"Mean = {sus_stats['mean']:.2f}\n")
        f.write(f"SD = {sus_stats['std']:.2f}\n")
        f.write(f"Median = {sus_stats['median']:.2f}\n")
        f.write(f"Range: {sus_stats['min']:.2f} - {sus_stats['max']:.2f}\n\n")
        f.write(f"Benchmark Interpretation:\n")
        f.write(f"  Mean = {sus_stats['mean']:.2f}\n")
        f.write(f"  Classification: {sus_stats['benchmark']}\n")
        f.write(f"  (>68: Above Average, >80: Excellent)\n")
        
        # Summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        significant_tests = [k for k, v in results.items() if v.get('significant', False)]
        f.write(f"\nTotal tests performed: {len(results)}\n")
        f.write(f"Significant results (p < 0.05): {len(significant_tests)}\n")
        if significant_tests:
            f.write("\nSignificant tests:\n")
            for test_key in significant_tests:
                test_result = results[test_key]
                f.write(f"  - {test_result['test_name']}: p = {test_result['p_value']:.4f}\n")
        else:
            f.write("\nNo significant results found (p < 0.05).\n")
    
    print(f"[info] Report written to: {output_path}")


def save_results_csv(results: dict, sus_stats: dict, output_path: Path) -> None:
    """Save statistical results to CSV file."""
    rows = []
    
    for key, result in results.items():
        rows.append({
            'Test': result['test_name'],
            'N': result['n'],
            'Assisted_Mean': result['assisted_mean'],
            'Assisted_SD': result['assisted_std'],
            'Assisted_Median': result['assisted_median'],
            'Unassisted_Mean': result['unassisted_mean'],
            'Unassisted_SD': result['unassisted_std'],
            'Unassisted_Median': result['unassisted_median'],
            'Statistic': result['statistic'],
            'P_Value': result['p_value'],
            'Significant': result['significant'],
            'Alternative': result.get('alternative', 'N/A')
        })
    
    # Add SUS descriptive stats
    rows.append({
        'Test': 'SUS Score (Assisted) - Descriptive Only',
        'N': sus_stats['n'],
        'Assisted_Mean': sus_stats['mean'],
        'Assisted_SD': sus_stats['std'],
        'Assisted_Median': sus_stats['median'],
        'Unassisted_Mean': np.nan,
        'Unassisted_SD': np.nan,
        'Unassisted_Median': np.nan,
        'Statistic': np.nan,
        'P_Value': np.nan,
        'Significant': False,
        'Alternative': 'N/A'
    })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_path, index=False)
    print(f"[info] Results CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical analysis of survey hypotheses (H4-H7)"
    )
    parser.add_argument(
        "--input-excel",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "Master_Survey_Data_By_Condition.xlsx",
        help="Path to input Excel file with survey data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "survey_hypotheses_analysis",
        help="Output directory for results",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_data(args.input_excel)
    
    # Calculate IPQ Realism subscale
    df = calculate_ipq_realism(df)
    
    # Perform statistical analyses
    print("\n[info] Performing statistical tests...")
    results = analyze_hypotheses(df)
    
    # Calculate SUS descriptive statistics
    print("\n[info] Calculating SUS descriptive statistics...")
    sus_stats = calculate_sus_descriptives(df)
    
    # Generate report
    report_path = args.output_dir / "statistical_report.txt"
    print("\n[info] Generating report...")
    generate_report(results, sus_stats, report_path)
    
    # Save results CSV
    results_csv_path = args.output_dir / "statistical_results.csv"
    save_results_csv(results, sus_stats, results_csv_path)
    
    print(f"\n[info] Analysis complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

