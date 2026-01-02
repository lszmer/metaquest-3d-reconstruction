#!/usr/bin/env python3
"""
Generate publication-ready participant demographics tables and visualizations.

This script creates ISMAR-ready descriptive statistics tables and visualizations
for participant demographics including:
- Age statistics (mean, SD, range)
- Gender distribution
- Nationality distribution
- Education levels
- VR and gaming experience
- Laterality and glasses-wearer information

Outputs:
- LaTeX table (for paper submission)
- HTML table (for easy viewing)
- CSV table (for data sharing)
- Publication-ready visualizations (PNG/PDF)

Console Usage:
    python analysis/reporting/generate_participant_demographics.py
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
    sns.set_palette("husl")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'


def load_survey_data(csv_path: Path) -> pd.DataFrame:
    """Load and clean survey data."""
    df = pd.read_csv(csv_path)
    
    # Remove rows with missing critical data (like participant 11 who has incomplete data)
    df = df.dropna(subset=['Age', 'Gender'])
    
    return df


def compute_demographic_statistics(df: pd.DataFrame) -> Dict:
    """Compute comprehensive demographic statistics."""
    stats_dict = {}
    
    # Age statistics
    ages = df['Age'].dropna()
    stats_dict['age'] = {
        'mean': ages.mean(),
        'std': ages.std(),
        'median': ages.median(),
        'min': ages.min(),
        'max': ages.max(),
        'n': len(ages)
    }
    
    # Gender distribution
    gender_counts = df['Gender'].value_counts()
    stats_dict['gender'] = {
        'counts': gender_counts.to_dict(),
        'percentages': (gender_counts / len(df) * 100).to_dict(),
        'total': len(df)
    }
    
    # Nationality distribution
    nationality_counts = df['Nationality'].value_counts()
    stats_dict['nationality'] = {
        'counts': nationality_counts.to_dict(),
        'percentages': (nationality_counts / len(df) * 100).to_dict(),
        'total': len(df)
    }
    
    # Education - Highest Obtained
    edu_highest = df['Education - Highest Obtained'].value_counts()
    stats_dict['education_highest'] = {
        'counts': edu_highest.to_dict(),
        'percentages': (edu_highest / len(df) * 100).to_dict(),
        'total': len(df)
    }
    
    # Education - Currently Pursued
    edu_current = df['Education - Currently Pursued'].value_counts()
    stats_dict['education_current'] = {
        'counts': edu_current.to_dict(),
        'percentages': (edu_current / len(df) * 100).to_dict(),
        'total': len(df)
    }
    
    # VR Experience
    vr_exp = df['Experience using VR Tools (yrs)'].dropna()
    stats_dict['vr_experience'] = {
        'mean': vr_exp.mean(),
        'std': vr_exp.std(),
        'median': vr_exp.median(),
        'min': vr_exp.min(),
        'max': vr_exp.max(),
        'n': len(vr_exp)
    }
    
    # Video Game Experience
    game_exp = df['Experience with Video Games (yrs)'].dropna()
    stats_dict['game_experience'] = {
        'mean': game_exp.mean(),
        'std': game_exp.std(),
        'median': game_exp.median(),
        'min': game_exp.min(),
        'max': game_exp.max(),
        'n': len(game_exp)
    }
    
    # Laterality
    laterality_counts = df['Laterality'].value_counts()
    stats_dict['laterality'] = {
        'counts': laterality_counts.to_dict(),
        'percentages': (laterality_counts / len(df) * 100).to_dict(),
        'total': len(df)
    }
    
    # Glasses wearer
    glasses_counts = df['Glass-wearer'].value_counts()
    stats_dict['glasses'] = {
        'counts': glasses_counts.to_dict(),
        'percentages': (glasses_counts / len(df) * 100).to_dict(),
        'total': len(df)
    }
    
    # Wore glasses during experiment
    wore_glasses_counts = df['Wore glasses during experiment'].value_counts()
    stats_dict['wore_glasses'] = {
        'counts': wore_glasses_counts.to_dict(),
        'percentages': (wore_glasses_counts / len(df) * 100).to_dict(),
        'total': len(df)
    }
    
    return stats_dict


def create_demographics_table(stats_dict: Dict, output_dir: Path) -> pd.DataFrame:
    """Create a comprehensive demographics table."""
    rows = []

    # Age
    age_stats = stats_dict['age']
    rows.append({
        'Characteristic': 'Age (years)',
        'Mean (SD)': f"{age_stats['mean']:.1f} ({age_stats['std']:.1f})",
        'Median [Range]': f"{age_stats['median']:.1f} [{age_stats['min']:.0f}--{age_stats['max']:.0f}]",
        'N': age_stats['n']
    })

    # Gender
    gender_stats = stats_dict['gender']
    rows.append({
        'Characteristic': 'Gender',
        'Mean (SD)': '',
        'Median [Range]': '',
        'N': ''
    })
    for gender, count in sorted(gender_stats['counts'].items()):
        pct = gender_stats['percentages'][gender]
        rows.append({
            'Characteristic': f'  {gender}',
            'Mean (SD)': f"{count} ({pct:.1f}%)",
            'Median [Range]': '',
            'N': ''
        })

    # VR Experience
    vr_stats = stats_dict['vr_experience']
    rows.append({
        'Characteristic': 'VR Experience (years)',
        'Mean (SD)': f"{vr_stats['mean']:.2f} ({vr_stats['std']:.2f})",
        'Median [Range]': f"{vr_stats['median']:.2f} [{vr_stats['min']:.2f}--{vr_stats['max']:.2f}]",
        'N': vr_stats['n']
    })

    # Game Experience
    game_stats = stats_dict['game_experience']
    rows.append({
        'Characteristic': 'Gaming Experience (years)',
        'Mean (SD)': f"{game_stats['mean']:.1f} ({game_stats['std']:.1f})",
        'Median [Range]': f"{game_stats['median']:.1f} [{game_stats['min']:.0f}--{game_stats['max']:.0f}]",
        'N': game_stats['n']
    })

    # Laterality
    laterality_stats = stats_dict['laterality']
    rows.append({
        'Characteristic': 'Laterality',
        'Mean (SD)': '',
        'Median [Range]': '',
        'N': ''
    })
    for laterality, count in sorted(laterality_stats['counts'].items()):
        pct = laterality_stats['percentages'][laterality]
        rows.append({
            'Characteristic': f'  {laterality}-handed',
            'Mean (SD)': f"{count} ({pct:.1f}%)",
            'Median [Range]': '',
            'N': ''
        })

    # Glasses wearer
    glasses_stats = stats_dict['glasses']
    rows.append({
        'Characteristic': 'Glasses wearer',
        'Mean (SD)': '',
        'Median [Range]': '',
        'N': ''
    })
    for glasses, count in sorted(glasses_stats['counts'].items()):
        pct = glasses_stats['percentages'][glasses]
        rows.append({
            'Characteristic': f'  {glasses}',
            'Mean (SD)': f"{count} ({pct:.1f}%)",
            'Median [Range]': '',
            'N': ''
        })

    # Education - Highest
    edu_highest_stats = stats_dict['education_highest']
    rows.append({
        'Characteristic': 'Highest Education',
        'Mean (SD)': '',
        'Median [Range]': '',
        'N': ''
    })
    for edu, count in sorted(edu_highest_stats['counts'].items()):
        pct = edu_highest_stats['percentages'][edu]
        rows.append({
            'Characteristic': f'  {edu}',
            'Mean (SD)': f"{count} ({pct:.1f}%)",
            'Median [Range]': '',
            'N': ''
        })

    # Education - Current
    edu_current_stats = stats_dict['education_current']
    rows.append({
        'Characteristic': 'Current Education',
        'Mean (SD)': '',
        'Median [Range]': '',
        'N': ''
    })
    for edu, count in sorted(edu_current_stats['counts'].items()):
        pct = edu_current_stats['percentages'][edu]
        rows.append({
            'Characteristic': f'  {edu}',
            'Mean (SD)': f"{count} ({pct:.1f}%)",
            'Median [Range]': '',
            'N': ''
        })

    # Nationality
    nationality_stats = stats_dict['nationality']
    rows.append({
        'Characteristic': 'Nationality',
        'Mean (SD)': '',
        'Median [Range]': '',
        'N': ''
    })
    for nat, count in sorted(nationality_stats['counts'].items()):
        pct = nationality_stats['percentages'][nat]
        rows.append({
            'Characteristic': f'  {nat}',
            'Mean (SD)': f"{count} ({pct:.1f}%)",
            'Median [Range]': '',
            'N': ''
        })

    df_table = pd.DataFrame(rows)
    return df_table


def create_latex_table(df_table: pd.DataFrame, output_path: Path):
    """Create LaTeX table for paper submission.

    Note: This table requires the following LaTeX packages:
    - booktabs (for \\toprule, \\midrule, \\bottomrule)
    - float (for table* environment, if not using stfloats)
    """
    latex_lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Participant Demographics}",
        "\\label{tab:demographics}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Characteristic & Mean (SD) & Median [Range] & N \\\\",
        "\\midrule"
    ]
    
    for _, row in df_table.iterrows():
        char_raw = str(row['Characteristic'])
        mean_sd_raw = row['Mean (SD)']
        median_range_raw = row['Median [Range]']
        n_val = row['N']
        
        # Handle empty values properly - ensure they're truly empty strings
        # Check for indentation before processing
        is_indented = char_raw.startswith('  ')
        char = char_raw.replace('&', '\\&').replace('%', '\\%').strip()

        # Handle mean_sd - check if value exists and is not empty
        try:
            mean_sd_str = str(mean_sd_raw).strip()
            mean_sd = mean_sd_str.replace('&', '\\&').replace('%', '\\%') if mean_sd_str != '' and mean_sd_str.lower() != 'nan' else ''
        except (ValueError, TypeError):
            mean_sd = ''

        # Handle median_range - check if value exists and is not empty
        try:
            median_range_str = str(median_range_raw).strip()
            median_range = median_range_str.replace('&', '\\&').replace('%', '\\%') if median_range_str != '' and median_range_str.lower() != 'nan' else ''
        except (ValueError, TypeError):
            median_range = ''
        n_str = str(n_val).strip()
        n = n_str if n_str != '' and n_str.lower() != 'nan' else ''
        
        # Indent sub-items
        if is_indented:
            char = '\\quad ' + char
        
        # Ensure exactly 3 & separators (4 columns total)
        # Empty cells should have no content (not even spaces) for proper LaTeX formatting
        # Use join to ensure proper formatting
        row_parts = []
        for part in [char, mean_sd, median_range, n]:
            # Ensure part is a string and has no leading/trailing whitespace
            part_str = str(part).strip() if part else ''
            # Remove 'nan' strings
            if part_str.lower() == 'nan':
                part_str = ''
            row_parts.append(part_str)
        
        row_str = ' & '.join(row_parts) + ' \\\\'
        
        # Validate: should have exactly 3 & separators
        if row_str.count('&') != 3:
            raise ValueError(f"Row has {row_str.count('&')} & separators, expected 3: {row_str}")
        
        latex_lines.append(row_str)
    
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_lines))


def create_html_table(df_table: pd.DataFrame, output_path: Path):
    """Create HTML table for easy viewing."""
    html = df_table.to_html(index=False, classes='demographics-table', 
                            table_id='demographics-table', escape=False)
    
    # Add some styling
    styled_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Participant Demographics</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #333;
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th {{
                background-color: #333;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }}
            td {{
                padding: 10px;
                border-bottom: 1px solid #ddd;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .indented {{
                padding-left: 30px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Participant Demographics</h1>
            {html}
        </div>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(styled_html)


def create_visualizations(df: pd.DataFrame, stats_dict: Dict, output_dir: Path):
    """Create publication-ready visualizations."""
    
    # 1. Age Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ages = df['Age'].dropna()
    age_mean = float(ages.mean())
    age_median = float(ages.median())
    ax.hist(ages, bins=range(int(ages.min()), int(ages.max())+2), 
            edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(age_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {age_mean:.1f}')
    ax.axvline(age_median, color='green', linestyle='--', linewidth=2, label=f'Median: {age_median:.1f}')
    ax.set_xlabel('Age (years)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Age Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'age_distribution.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'age_distribution.pdf', bbox_inches='tight')
    plt.close()
    
    # 2. Gender Distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    gender_counts = stats_dict['gender']['counts']
    labels = list(gender_counts.keys())
    values = list(gender_counts.values())
    colors = ['#ff9999', '#66b3ff'] if len(gender_counts) == 2 else None
    ax.pie(values, labels=labels, autopct='%1.1f%%',
           startangle=90, colors=colors, textprops={'fontsize': 10})
    ax.set_title('Gender Distribution', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'gender_distribution.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'gender_distribution.pdf', bbox_inches='tight')
    plt.close()
    
    # 3. Experience Levels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # VR Experience
    vr_exp = df['Experience using VR Tools (yrs)'].dropna()
    vr_mean = float(vr_exp.mean())
    ax1.hist(vr_exp, bins=15, edgecolor='black', alpha=0.7, color='coral')
    ax1.axvline(vr_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {vr_mean:.2f}')
    ax1.set_xlabel('VR Experience (years)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('VR Experience Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gaming Experience
    game_exp = df['Experience with Video Games (yrs)'].dropna()
    game_mean = float(game_exp.mean())
    ax2.hist(game_exp, bins=15, edgecolor='black', alpha=0.7, color='lightgreen')
    ax2.axvline(game_mean, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {game_mean:.1f}')
    ax2.set_xlabel('Gaming Experience (years)', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Gaming Experience Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'experience_distributions.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'experience_distributions.pdf', bbox_inches='tight')
    plt.close()
    
    # 4. Education Levels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Highest Education
    edu_highest = stats_dict['education_highest']['counts']
    labels_highest = list(edu_highest.keys())
    values_highest = list(edu_highest.values())
    ax1.barh(range(len(edu_highest)), values_highest, color='steelblue', edgecolor='black')
    ax1.set_yticks(range(len(edu_highest)))
    ax1.set_yticklabels(labels_highest)
    ax1.set_xlabel('Count', fontsize=11)
    ax1.set_title('Highest Education Obtained', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(values_highest):
        ax1.text(v + 0.1, i, str(v), va='center', fontsize=10)
    
    # Current Education
    edu_current = stats_dict['education_current']['counts']
    labels_current = list(edu_current.keys())
    values_current = list(edu_current.values())
    ax2.barh(range(len(edu_current)), values_current, color='lightcoral', edgecolor='black')
    ax2.set_yticks(range(len(edu_current)))
    ax2.set_yticklabels(labels_current)
    ax2.set_xlabel('Count', fontsize=11)
    ax2.set_title('Currently Pursued Education', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(values_current):
        ax2.text(v + 0.1, i, str(v), va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'education_levels.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'education_levels.pdf', bbox_inches='tight')
    plt.close()
    
    # 5. Laterality and Glasses
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Laterality
    laterality = stats_dict['laterality']['counts']
    labels_lat = list(laterality.keys())
    values_lat = list(laterality.values())
    ax1.pie(values_lat, labels=labels_lat, autopct='%1.1f%%',
            startangle=90, colors=['#ffcc99', '#99ccff'], textprops={'fontsize': 10})
    ax1.set_title('Handedness Distribution', fontsize=12, fontweight='bold')
    
    # Glasses
    glasses = stats_dict['glasses']['counts']
    labels_glasses = list(glasses.keys())
    values_glasses = list(glasses.values())
    colors_glasses = ['#ff9999', '#66b3ff'] if len(glasses) == 2 else None
    ax2.pie(values_glasses, labels=labels_glasses, autopct='%1.1f%%',
            startangle=90, colors=colors_glasses, textprops={'fontsize': 10})
    ax2.set_title('Glasses Wearer Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'laterality_glasses.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'laterality_glasses.pdf', bbox_inches='tight')
    plt.close()
    
    # 6. Nationality Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    nationality = stats_dict['nationality']['counts']
    labels_nat = list(nationality.keys())
    values_nat = list(nationality.values())
    ax.barh(range(len(nationality)), values_nat, color='mediumpurple', edgecolor='black')
    ax.set_yticks(range(len(nationality)))
    ax.set_yticklabels(labels_nat)
    ax.set_xlabel('Count', fontsize=11)
    ax.set_title('Nationality Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    for i, v in enumerate(values_nat):
        ax.text(v + 0.1, i, str(v), va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'nationality_distribution.png', bbox_inches='tight', dpi=300)
    plt.savefig(output_dir / 'nationality_distribution.pdf', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Generate ISMAR-ready participant demographics tables and visualizations'
    )
    parser.add_argument(
        '--survey-data',
        type=Path,
        default=Path(__file__).parent.parent / 'data' / 'Master_Survey_Data_final.xlsx - Sheet1.csv',
        help='Path to survey data CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'reports' / 'participant_demographics',
        help='Output directory for generated files'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading survey data from {args.survey_data}...")
    df = load_survey_data(args.survey_data)
    print(f"Loaded data for {len(df)} participants")
    
    print("Computing demographic statistics...")
    stats_dict = compute_demographic_statistics(df)
    
    print("Creating demographics table...")
    df_table = create_demographics_table(stats_dict, args.output_dir)
    
    # Save tables in multiple formats
    print("Saving tables...")
    df_table.to_csv(args.output_dir / 'demographics_table.csv', index=False)
    create_latex_table(df_table, args.output_dir / 'demographics_table.tex')
    create_html_table(df_table, args.output_dir / 'demographics_table.html')
    
    print("Creating visualizations...")
    create_visualizations(df, stats_dict, args.output_dir)
    
    print(f"\n✓ All files saved to {args.output_dir}")
    print("\nGenerated files:")
    print("  - demographics_table.csv (CSV format)")
    print("  - demographics_table.tex (LaTeX format for paper)")
    print("  - demographics_table.html (HTML format for viewing)")
    print("  - age_distribution.png/pdf")
    print("  - gender_distribution.png/pdf")
    print("  - experience_distributions.png/pdf")
    print("  - education_levels.png/pdf")
    print("  - laterality_glasses.png/pdf")
    print("  - nationality_distribution.png/pdf")
    print("\nIMPORTANT: The LaTeX table requires \\usepackage{booktabs}")
    print("Add this to your LaTeX document preamble.")


if __name__ == '__main__':
    main()

