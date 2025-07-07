#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe
import os
import glob
import argparse  # Import the argparse module

def analyze_cell_threshold(filename, p_value_cutoff=0.05):
    """
    Analyzes HPO data to suggest a minimum cell count threshold.
    (Same as before, returns DataFrames)
    """
    # Read the file and try to infer column names
    df = pd.read_csv(filename, sep='\t')
    
    # Identify the cell count column - find the column that contains 'n_cells_in_group'
    cell_count_column = None
    for col in df.columns:
        if 'n_cells_in_group' in col:
            cell_count_column = col
            break
    
    # If we can't find the column, try to use the 11th column (index 10)
    if cell_count_column is None:
        if len(df.columns) >= 11:
            cell_count_column = df.columns[10]
        else:
            print(f"Could not find cell count column in {filename}")
            return None, None, None, None, None
    
    # Drop rows with missing cell count values
    df = df.dropna(subset=[cell_count_column])
    
    # Convert to integer, handling any non-numeric values
    try:
        df[cell_count_column] = pd.to_numeric(df[cell_count_column], errors='coerce')
        df = df.dropna(subset=[cell_count_column])
        df[cell_count_column] = df[cell_count_column].astype(int)
    except ValueError:
        print(f"Error converting cell count column to integer in {filename}")
        return None, None, None, None, None
    
    # Find the equivalent KS statistic and p-value columns
    ks_stat_column = None
    ks_pvalue_column = None
    significant_column = None
    
    for col in df.columns:
        if 'ks_statistic' in col.lower():
            ks_stat_column = col
        if 'ks_pvalue' in col.lower() or ('p' in col.lower() and 'value' in col.lower()):
            ks_pvalue_column = col
        if 'significant' in col.lower():
            significant_column = col
    
    # If we couldn't find the columns, try to use the expected positions
    if ks_stat_column is None and len(df.columns) >= 13:
        ks_stat_column = df.columns[12]
    if ks_pvalue_column is None and len(df.columns) >= 14:
        ks_pvalue_column = df.columns[13]
    if significant_column is None and len(df.columns) >= 16:
        significant_column = df.columns[15]
    
    # If we still can't find the required columns, return None
    if ks_stat_column is None or ks_pvalue_column is None or significant_column is None:
        print(f"Could not find required columns in {filename}")
        return None, None, None, None, None
    
    if len(df[cell_count_column].unique()) < 2:
        return None, None, None, None, None  # Added None for raw_data

    cell_counts = sorted(df[cell_count_column].unique())
    median_pvalues = []
    median_ks_stats = []
    significant_ratios = []

    for count in cell_counts:
        subset = df[df[cell_count_column] >= count]
        median_pvalues.append(subset[ks_pvalue_column].median())
        median_ks_stats.append(subset[ks_stat_column].median())
        significant_ratio = subset[significant_column].mean()
        significant_ratios.append(significant_ratio)

    # Create DataFrames for easier aggregation later
    pvalue_df = pd.DataFrame({'cell_count': cell_counts, 'median_pvalue': median_pvalues, 'file': filename})
    ks_df = pd.DataFrame({'cell_count': cell_counts, 'median_ks_stat': median_ks_stats, 'file': filename})
    significant_df = pd.DataFrame({'cell_count': cell_counts, 'significant_ratio': significant_ratios, 'file': filename})

    # Create a simplified dataframe with raw data for the relationship plot
    raw_data = df[[cell_count_column, ks_stat_column, ks_pvalue_column]].copy()
    raw_data.columns = ['cell_count', 'ks_statistic', 'pvalue']
    raw_data['file'] = filename

    knee_point_significant = None
    if len(significant_ratios) > 1:
        diffs = np.diff(significant_ratios)
        knee_point_index = np.argmax(diffs)
        knee_point_significant = cell_counts[knee_point_index]
    return pvalue_df, ks_df, significant_df, knee_point_significant, raw_data


def analyze_multiple_hpo_files(directory, p_value_cutoff=0.05, output_prefix="hpo_analysis"):
    """
    Analyzes multiple HPO files. (Now with directory as an argument)
    """

    all_pvalues = []
    all_ks_stats = []
    all_significant_ratios = []
    all_knee_points = []
    all_raw_data = []

    filenames = glob.glob(os.path.join(directory, "*.tsv"))
    if not filenames:
        print(f"No .tsv files found in directory: {directory}")
        return

    for filename in filenames:
        print(f"Processing {filename}...")
        pvalue_df, ks_df, significant_df, knee_point, raw_data = analyze_cell_threshold(filename, p_value_cutoff)

        if pvalue_df is not None:
            all_pvalues.append(pvalue_df)
            all_ks_stats.append(ks_df)
            all_significant_ratios.append(significant_df)
            all_raw_data.append(raw_data)
            if knee_point is not None:
                all_knee_points.append(knee_point)

    if not all_pvalues:
        print("No files were successfully analyzed.")
        return
    all_pvalues_df = pd.concat(all_pvalues)
    all_ks_stats_df = pd.concat(all_ks_stats)
    all_significant_ratios_df = pd.concat(all_significant_ratios)
    all_raw_data_df = pd.concat(all_raw_data)

    aggregated_pvalues = all_pvalues_df.groupby('cell_count')['median_pvalue'].median().reset_index()
    aggregated_ks_stats = all_ks_stats_df.groupby('cell_count')['median_ks_stat'].median().reset_index()
    aggregated_significant_ratios = all_significant_ratios_df.groupby('cell_count')['significant_ratio'].median().reset_index()

    # Create original 3 plots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    axes[0].plot(aggregated_pvalues['cell_count'], aggregated_pvalues['median_pvalue'], marker='o')
    axes[0].set_xlabel('Minimum Cell Count Threshold')
    axes[0].set_ylabel('Median of Median P-values')
    axes[0].set_title('Aggregated Median P-value vs. Minimum Cell Count')
    axes[0].axhline(y=p_value_cutoff, color='r', linestyle='--', label=f'Significance Threshold ({p_value_cutoff})')
    axes[0].legend()
    axes[0].set_xscale('log')
    axes[0].grid(True)

    axes[1].plot(aggregated_ks_stats['cell_count'], aggregated_ks_stats['median_ks_stat'], marker='o')
    axes[1].set_xlabel('Minimum Cell Count Threshold')
    axes[1].set_ylabel('Median of Median KS-statistics')
    axes[1].set_title('Aggregated Median KS-statistic vs. Minimum Cell Count')
    axes[1].set_xscale('log')
    axes[1].grid(True)

    axes[2].plot(aggregated_significant_ratios['cell_count'], aggregated_significant_ratios['significant_ratio'], marker='o')
    axes[2].set_xlabel('Minimum Cell Count Threshold')
    axes[2].set_ylabel('Median of Significant Result Proportions')
    axes[2].set_title('Aggregated Proportion of Significant Results vs. Minimum Cell Count')
    axes[2].set_xscale('log')
    axes[2].grid(True)
    axes[2].axhline(y=0.95, color='g', linestyle='--', label='95% Significant')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_aggregated_plots.png")
    plt.close()

    # Create new additional plot showing KS Statistic vs Cell Count colored by p-value
    fig_relationship, ax_relationship = plt.subplots(figsize=(12, 8))

    # Use a scatter plot with color based on p-value
    scatter = ax_relationship.scatter(
        all_raw_data_df['cell_count'],
        all_raw_data_df['ks_statistic'],
        c=-np.log10(all_raw_data_df['pvalue'].clip(1e-50, 1)),  # -log10 transform for better visualization
        cmap='viridis',
        alpha=0.6,
        s=30
    )

    # Add a colorbar and label it
    cbar = plt.colorbar(scatter, ax=ax_relationship)
    cbar.set_label('-log10(p-value)')

    # Add trend line (LOWESS or polynomial fit could be better, using simple median here)
    trend_data = all_raw_data_df.groupby('cell_count')['ks_statistic'].median().reset_index()
    ax_relationship.plot(trend_data['cell_count'], trend_data['ks_statistic'],
                       'r-', linewidth=2, label='Median KS Statistic')

    ax_relationship.set_xlabel('Cell Count')
    ax_relationship.set_ylabel('KS Statistic')
    ax_relationship.set_title('Relationship between KS Statistic and Cell Count')
    ax_relationship.set_xscale('log')
    ax_relationship.grid(True, alpha=0.3)
    ax_relationship.legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_ks_cell_relationship.png")
    plt.close()

    # Create hexbin plot for dense data visualization (alternative view)
    fig_hex, ax_hex = plt.subplots(figsize=(12, 8))

    hb = ax_hex.hexbin(all_raw_data_df['cell_count'], all_raw_data_df['ks_statistic'],
                      gridsize=30, cmap='Blues', xscale='log',
                      mincnt=1, bins='log')

    ax_hex.set_xlabel('Cell Count')
    ax_hex.set_ylabel('KS Statistic')
    ax_hex.set_title('Density Plot of KS Statistic vs Cell Count')
    ax_hex.grid(True, alpha=0.3)

    cbar_hex = plt.colorbar(hb, ax=ax_hex)
    cbar_hex.set_label('log10(count)')

    # Add trend line
    ax_hex.plot(trend_data['cell_count'], trend_data['ks_statistic'],
              'r-', linewidth=2, label='Median KS Statistic')
    ax_hex.legend()

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_ks_cell_hexbin.png")
    plt.close()

    # Process and display knee points if available
    if all_knee_points:
      fig_knee, ax_knee = plt.subplots(figsize=(8, 6))
      ax_knee.hist(all_knee_points, bins=20, edgecolor='black', alpha=0.7)
      ax_knee.set_xlabel('Suggested Threshold (Knee Point)')
      ax_knee.set_ylabel('Number of Files')
      ax_knee.set_title('Distribution of Suggested Thresholds Across Files')
      ax_knee.set_xscale('log')
      ax_knee.grid(axis='y', alpha=0.75)

      median_knee_point = np.median(all_knee_points)
      ax_knee.axvline(median_knee_point, color='r', linestyle='--', label=f'Median Threshold: {median_knee_point:.2f}')
      ax_knee.legend()
      plt.savefig(f"{output_prefix}_knee_point_histogram.png")
      plt.close()
      print(f"\nMedian suggested threshold across all files: {median_knee_point}")
    else:
        print("No knee points were determined across all files.")

    # Show all plots if interactive environment
    print(f"All plots have been saved with prefix: {output_prefix}")
    print("Generated plots:")
    print(f"1. {output_prefix}_aggregated_plots.png")
    print(f"2. {output_prefix}_ks_cell_relationship.png")
    print(f"3. {output_prefix}_ks_cell_hexbin.png")
    if all_knee_points:
        print(f"4. {output_prefix}_knee_point_histogram.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze multiple HPO files to determine a cell count threshold.")
    parser.add_argument("directory", help="Path to the directory containing HPO files.")
    parser.add_argument("-p", "--p_value_cutoff", type=float, default=0.05,
                        help="P-value threshold for significance (default: 0.05).")
    parser.add_argument("-o", "--output_prefix", default="hpo_analysis",
                        help="Prefix for output files (default: hpo_analysis).")
    args = parser.parse_args()

    analyze_multiple_hpo_files(args.directory, args.p_value_cutoff, args.output_prefix)
