#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import scipy.cluster.hierarchy as sch
# --- Configuration ---
MAX_LABEL_LENGTH = 75
DEFAULT_TOP_N = 25
DEFAULT_MAX_CELL_GROUPS = 600  # New default for maximum cell groups to plot
MAX_HEATMAP_DIMS = (30000, 30000)

# --- Helper Functions ---

def truncate_label(label, max_len=MAX_LABEL_LENGTH):
    """Truncates long string labels."""
    if not isinstance(label, str):
        label = str(label)
    if len(label) > max_len:
        return label[:max_len-3] + "..."
    return label

def create_full_heatmap(ks_data, hpo_info, hpo_id_col, hpo_name_col, output_prefix, max_cell_groups=DEFAULT_MAX_CELL_GROUPS):
    """
    Create a comprehensive heatmap of HPO and cell group KS statistics.
    
    Parameters:
    - ks_data: DataFrame with KS statistics
    - hpo_info: DataFrame with HPO information
    - hpo_id_col: Column name for HPO ID
    - hpo_name_col: Column name for HPO name
    - output_prefix: Prefix for output filename
    - max_cell_groups: Maximum number of cell groups to include in the plot
    """
    # Ensure data is numeric and replace any non-numeric values with 0
    ks_matrix = ks_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Select top cell groups by maximum KS statistic if needed
    if ks_matrix.shape[1] > max_cell_groups:
        print(f"Total cell groups: {ks_matrix.shape[1]}. Selecting top {max_cell_groups} by max KS statistic.")
        
        # Calculate max KS for each cell group and select top N
        max_ks_per_cg = ks_matrix.max(axis=0)
        top_cell_groups = max_ks_per_cg.nlargest(max_cell_groups).index
        
        # Subset the matrix to these top cell groups
        ks_matrix = ks_matrix[top_cell_groups]
        print(f"Selected cell groups based on max KS statistic. New matrix shape: {ks_matrix.shape}")
    
    # Prepare labels
    row_labels = hpo_info[hpo_name_col].tolist()
    col_labels = ks_matrix.columns.tolist()
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    
    try:
        # Clustering rows (HPO terms)
        row_linkage = sch.linkage(ks_matrix, method='average')
        row_order = sch.leaves_list(row_linkage)
        
        # Clustering columns (cell groups)
        col_linkage = sch.linkage(ks_matrix.T, method='average')
        col_order = sch.leaves_list(col_linkage)
        
        # Reorder matrix and labels based on clustering
        ks_matrix_clustered = ks_matrix.iloc[row_order, col_order]
        row_labels_clustered = [row_labels[i] for i in row_order]
        col_labels_clustered = [col_labels[i] for i in col_order]
    except Exception as e:
        print(f"Clustering failed: {e}. Using original order.")
        ks_matrix_clustered = ks_matrix
        row_labels_clustered = row_labels
        col_labels_clustered = col_labels

    # Determine appropriate plotting strategy based on matrix size
    print(f"Full matrix size: {ks_matrix_clustered.shape}")
    
    # Strategy for handling very large matrices
    if ks_matrix_clustered.shape[0] > 500 or ks_matrix_clustered.shape[1] > 500:
        print("Matrix is very large. Using tiled heatmap approach.")
        create_tiled_heatmap(
            ks_matrix_clustered, 
            row_labels_clustered, 
            col_labels_clustered, 
            output_prefix
        )
    else:
        create_single_heatmap(
            ks_matrix_clustered, 
            row_labels_clustered, 
            col_labels_clustered, 
            output_prefix
        )
def create_single_heatmap(ks_matrix, row_labels, col_labels, output_prefix):
    """
    Create a single heatmap for smaller matrices.
    """
    # Determine figure size dynamically
    base_width = min(max(len(col_labels) * 0.1, 10), 100)  # Cap at 100
    base_height = min(max(len(row_labels) * 0.1, 10), 100)  # Cap at 100
    
    plt.figure(figsize=(base_width, base_height), dpi=300)
    
    # Create heatmap with updated parameters
    sns.heatmap(
        ks_matrix, 
        cmap='viridis',  # Color map that works well for continuous data
        cbar_kws={'label': 'KS Statistic'},
        xticklabels=col_labels,
        yticklabels=row_labels,
        square=False,  # Allow rectangular cells
        linewidths=0.1,  # Thin lines between cells
        linecolor='lightgrey'
    )
    
    plt.title('Full HPO-Cell Group KS Statistic Heatmap (Clustered)', fontsize=16)
    plt.xlabel('Cell Groups', fontsize=12)
    plt.ylabel('HPO Terms', fontsize=12)
    
    # Rotate x-axis labels for readability
    plt.xticks(rotation=90, ha='right', fontsize=min(8, max(2, 300 // len(col_labels))))
    plt.yticks(rotation=0, fontsize=min(8, max(2, 300 // len(row_labels))))
    
    plt.tight_layout()
    
    # Save with careful file naming and error handling
    filename = f"{output_prefix}_full_heatmap.png"
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight', 
                    max_width=MAX_HEATMAP_DIMS[0], 
                    max_height=MAX_HEATMAP_DIMS[1])
        print(f"Saved full heatmap: {filename}")
    except Exception as e:
        print(f"Error saving full heatmap {filename}: {e}")
    
    plt.close()

def create_single_heatmap(ks_matrix, row_labels, col_labels, output_prefix):
    """
    Create a single heatmap for smaller matrices.
    """
    # Determine figure size dynamically
    # Cap the size in inches to prevent excessively large figures before DPI scaling
    base_width = min(max(len(col_labels) * 0.1, 10), 100)
    base_height = min(max(len(row_labels) * 0.1, 10), 100)

    # Set a high DPI for better resolution
    dpi_setting = 300
    
    print(f"Creating single heatmap with figure size (inches): ({base_width:.2f}, {base_height:.2f}) at {dpi_setting} DPI")

    plt.figure(figsize=(base_width, base_height), dpi=dpi_setting)

    # Create heatmap with updated parameters
    sns.heatmap(
        ks_matrix,
        cmap='viridis',  # Color map that works well for continuous data
        cbar_kws={'label': 'KS Statistic'},
        xticklabels=col_labels, # Use provided clustered labels
        yticklabels=row_labels, # Use provided clustered labels
        square=False,  # Allow rectangular cells
        linewidths=0.1,  # Thin lines between cells
        linecolor='lightgrey'
    )

    plt.title('Full HPO-Cell Group KS Statistic Heatmap (Clustered)', fontsize=16)
    plt.xlabel('Cell Groups', fontsize=12)
    plt.ylabel('HPO Terms', fontsize=12)

    # Adjust font size based on number of labels, with min/max caps
    xtick_fontsize = max(2, min(8, 300 // len(col_labels))) if len(col_labels) > 0 else 8
    ytick_fontsize = max(2, min(8, 300 // len(row_labels))) if len(row_labels) > 0 else 8
    
    plt.xticks(rotation=90, ha='right', fontsize=xtick_fontsize)
    plt.yticks(rotation=0, fontsize=ytick_fontsize)

    # Use tight_layout cautiously, it can fail on very large figures
    try:
        plt.tight_layout()
    except ValueError:
        print("Warning: tight_layout failed, plot margins might be suboptimal.")

    # Save with careful file naming and error handling (NO max_width/max_height)
    filename = f"{output_prefix}_full_heatmap.png"
    try:
        # REMOVED max_width and max_height arguments
        plt.savefig(filename, dpi=dpi_setting, bbox_inches='tight')
        print(f"Saved full heatmap: {filename}")
    except Exception as e:
        print(f"Error saving full heatmap {filename}: {e}")

    plt.close()

# Modified plotting function to handle ranked values (not just counts)
def plot_top_ranked(data_series, top_n, title, xlabel, ylabel, filename, is_horizontal=False):
    """Generates and saves a bar plot for top N ranked items based on their values."""
    if data_series.empty:
        print(f"Skipping plot '{title}': No data to plot.")
        return

    # Ensure data is sorted descending before taking top N (though it should be already)
    top_data = data_series.nlargest(min(top_n, len(data_series)))

    if top_data.empty:
        print(f"Skipping plot '{title}': No data after filtering/selection.")
        return

    plt.figure(figsize=(10, max(6, len(top_data) * 0.3)))
    labels = [truncate_label(label) for label in top_data.index]

    if is_horizontal: # Typically for long labels like cell groups
        plt.barh(labels, top_data.values, color=sns.color_palette("viridis", len(top_data)))
        plt.gca().invert_yaxis() # Top item at the top
        plt.grid(axis='x', linestyle='--', alpha=0.6)
    else: # Typically for shorter labels like HPO names
        plt.bar(labels, top_data.values, color=sns.color_palette("viridis", len(top_data)))
        plt.xticks(rotation=90)
        plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved plot: {filename}")
    except Exception as e:
        print(f"Error saving plot {filename}: {e}")
    plt.close()

def plot_comprehensive_scatter(ks_data, hpo_info, hpo_id_col, hpo_name_col, output_prefix):
    """
    Create a comprehensive scatter plot showing all HPOs and cell groups.
    
    Parameters:
    - ks_data: DataFrame with KS statistics
    - hpo_info: DataFrame with HPO information
    - hpo_id_col: Column name for HPO ID
    - hpo_name_col: Column name for HPO name
    - output_prefix: Prefix for output filename
    """
    # Calculate summary statistics for each HPO
    hpo_max_ks = ks_data.max(axis=1)
    hpo_mean_ks = ks_data.mean(axis=1)
    hpo_num_significant = (ks_data > 0).sum(axis=1)

    # Combine with HPO info
    comprehensive_df = pd.DataFrame({
        'HPO_ID': hpo_info[hpo_id_col],
        'HPO_Name': hpo_info[hpo_name_col],
        'Max_KS': hpo_max_ks,
        'Mean_KS': hpo_mean_ks,
        'Num_Significant_Associations': hpo_num_significant
    })

    # Create scatter plot
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        comprehensive_df['Mean_KS'], 
        comprehensive_df['Num_Significant_Associations'], 
        c=comprehensive_df['Max_KS'], 
        cmap='viridis', 
        alpha=0.7,
        s=50,  # marker size
        edgecolors='black',
        linewidth=0.5
    )
    plt.colorbar(scatter, label='Maximum KS Statistic')
    plt.title('Comprehensive HPO Analysis: KS Statistics Overview', fontsize=16)
    plt.xlabel('Mean KS Statistic Across Cell Groups', fontsize=12)
    plt.ylabel('Number of Significant Associations (KS > 0)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Annotate top 10 interesting points
    interesting_points = comprehensive_df.nlargest(10, 'Max_KS')
    for _, row in interesting_points.iterrows():
        plt.annotate(
            truncate_label(row['HPO_Name'], max_len=30), 
            (row['Mean_KS'], row['Num_Significant_Associations']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.7
        )

    # Save the plot
    filename = f"{output_prefix}_comprehensive_hpo_analysis.png"
    plt.tight_layout()
    try:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved comprehensive analysis plot: {filename}")
    except Exception as e:
        print(f"Error saving comprehensive plot {filename}: {e}")
    plt.close()

    # Additional: Generate a text summary of the comprehensive analysis
    summary_filename = f"{output_prefix}_comprehensive_hpo_analysis_summary.txt"
    with open(summary_filename, 'w') as f:
        f.write("Comprehensive HPO Analysis Summary\n")
        f.write("=" * 40 + "\n\n")
        f.write("Top 10 HPOs by Maximum KS Statistic:\n")
        top_10_max_ks = comprehensive_df.nlargest(10, 'Max_KS')
        f.write(top_10_max_ks.to_string(index=False) + "\n\n")
        
        f.write("Summary Statistics:\n")
        f.write(f"Total HPO Terms: {len(comprehensive_df)}\n")
        f.write(f"Mean Max KS Across HPOs: {comprehensive_df['Max_KS'].mean():.4f}\n")
        f.write(f"Median Max KS Across HPOs: {comprehensive_df['Max_KS'].median():.4f}\n")
        f.write(f"Mean Number of Significant Associations: {comprehensive_df['Num_Significant_Associations'].mean():.2f}\n")

    print(f"Saved comprehensive analysis summary: {summary_filename}")

# --- Main Visualization and Analysis Logic ---

def visualize_and_analyze(input_file, output_prefix, top_n=DEFAULT_TOP_N, max_cell_groups=DEFAULT_MAX_CELL_GROUPS):
    """Loads data, performs analyses, and generates visualizations based on KS magnitude."""

    print(f"Loading matrix data from: {input_file}")
    try:
        df = pd.read_csv(input_file, sep='\t', low_memory=False)
        print(f"Matrix loaded successfully. Shape: {df.shape}")
        # --- Data Validation and Setup ---
        if df.shape[1] < 3: raise ValueError("Matrix needs >= 3 columns (ID, Name, CellGroups).")
        hpo_id_col = df.columns[0]; hpo_name_col = df.columns[1]
        print(f"Assuming HPO ID: '{hpo_id_col}', HPO Name: '{hpo_name_col}'")
        hpo_info = df[[hpo_id_col, hpo_name_col]]
        ks_data = df.iloc[:, 2:]
        if ks_data.empty: raise ValueError("No cell group columns found (columns 3 onwards).")
        # Ensure KS data is numeric, coercing errors and filling resulting NaNs with 0
        ks_data = ks_data.apply(pd.to_numeric, errors='coerce').fillna(0)

    except FileNotFoundError: print(f"Error: Input file not found at {input_file}"); return
    except ValueError as ve: print(f"Error during data setup: {ve}"); return
    except Exception as e: print(f"Error loading or processing data: {e}"); return

    # --- Basic Summaries ---
    num_hpos, num_cell_groups = df.shape[0], ks_data.shape[1]
    num_significant_gt_zero = (ks_data > 0).sum().sum() # Keep this for context
    print(f"Found {num_hpos} HPO terms and {num_cell_groups} unique cell groups.")
    print(f"Total associations with KS > 0: {num_significant_gt_zero}")

    # --- Analysis 1: Top HPO-Cell Group Pairs (Overall Highest KS) ---
    print(f"\n--- Analysis 1: Top {top_n} HPO-Cell Group Pairs (Overall Highest KS) ---")
    ks_values_stacked = ks_data[ks_data > 0].stack()
    if ks_values_stacked.empty:
        print("No associations with KS > 0 found.")
        top_pairs_final_df = pd.DataFrame() # Ensure variable exists
    else:
        top_pairs_series = ks_values_stacked.sort_values(ascending=False).head(top_n)
        top_pairs_df = top_pairs_series.reset_index()
        top_pairs_df.columns = ['original_row_index', 'cell_group', 'ks_statistic']
        hpo_info_to_merge = df[[hpo_id_col, hpo_name_col]].reset_index().rename(columns={'index': 'original_row_index'})
        top_pairs_final_df = pd.merge(top_pairs_df, hpo_info_to_merge, on='original_row_index', how='left')
        top_pairs_final_df = top_pairs_final_df[[hpo_id_col, hpo_name_col, 'cell_group', 'ks_statistic']]
        print(f"Top {top_n} pairs:")
        print(top_pairs_final_df.to_string(index=False))
    print("-" * 60)

    # --- Analysis 2: HPOs with Highest Max KS ---
    print(f"\n--- Analysis 2: Top {top_n} HPOs by Highest Maximum KS Value ---")
    max_ks_per_hpo = ks_data.max(axis=1)
    if max_ks_per_hpo.isnull().all() or max_ks_per_hpo.empty:
        print("Cannot calculate max KS per HPO.")
        top_hpos_by_max_ks = pd.DataFrame() # Ensure variable exists
    else:
        hpo_max_ks_df = hpo_info.copy()
        hpo_max_ks_df['max_ks_statistic'] = max_ks_per_hpo.values
        top_hpos_by_max_ks = hpo_max_ks_df.sort_values(by='max_ks_statistic', ascending=False).head(top_n)
        print(f"Top {top_n} HPOs ranked by their highest single association KS value:")
        print(top_hpos_by_max_ks.to_string(index=False))
    print("-" * 60)

    # --- Analysis 3: Cell Groups with Highest Max KS ---
    print(f"\n--- Analysis 3: Top {top_n} Cell Groups by Highest Maximum KS Value ---")
    max_ks_per_cg = ks_data.max(axis=0)
    if max_ks_per_cg.isnull().all() or max_ks_per_cg.empty:
        print("Cannot calculate max KS per Cell Group.")
        top_cgs_by_max_ks_df = pd.DataFrame() # Ensure variable exists
        top_cgs_by_max_ks_series = pd.Series(dtype=float) # Ensure variable exists
    else:
        top_cgs_by_max_ks_series = max_ks_per_cg.sort_values(ascending=False).head(top_n)
        top_cgs_by_max_ks_df = top_cgs_by_max_ks_series.reset_index()
        top_cgs_by_max_ks_df.columns = ['cell_group', 'max_ks_statistic']
        print(f"Top {top_n} Cell Groups ranked by their highest single association KS value:")
        top_cgs_by_max_ks_df['cell_group_trunc'] = top_cgs_by_max_ks_df['cell_group'].apply(truncate_label)
        print(top_cgs_by_max_ks_df[['cell_group_trunc', 'max_ks_statistic']].to_string(index=False))
    print("-" * 60)


    # --- Visualization Plot 1: Top HPOs by Highest Max KS Value ---
    print(f"\nGenerating plot for top {top_n} HPOs by Highest Max KS...")
    if not top_hpos_by_max_ks.empty:
        # Need Series: Index=HPO Name, Value=Max KS
        plot_data_hpo = top_hpos_by_max_ks.set_index(hpo_name_col)['max_ks_statistic']
        plot_top_ranked(
            data_series=plot_data_hpo,
            top_n=top_n, # Already selected top N
            title=f'Top {len(plot_data_hpo)} HPO Terms by Highest Max KS Statistic',
            xlabel='Maximum KS Statistic',
            ylabel='HPO Term',
            filename=f"{output_prefix}_top_{top_n}_hpo_by_max_ks.png",
            is_horizontal=False # Plot vertically
        )
    else:
        print("Skipping plot: No top HPOs by max KS found.")

    # --- Visualization Plot 2: Top Cell Groups by Highest Max KS Value ---
    print(f"\nGenerating plot for top {top_n} Cell Groups by Highest Max KS...")
    if not top_cgs_by_max_ks_series.empty:
        # Need Series: Index=Cell Group, Value=Max KS (already have this)
        plot_top_ranked(
            data_series=top_cgs_by_max_ks_series,
            top_n=top_n, # Already selected top N
            title=f'Top {len(top_cgs_by_max_ks_series)} Cell Groups by Highest Max KS Statistic',
            xlabel='Maximum KS Statistic',
            ylabel='Cell Group Identifier',
            filename=f"{output_prefix}_top_{top_n}_cellgroup_by_max_ks.png",
            is_horizontal=True # Plot horizontally
        )
    else:
        print("Skipping plot: No top Cell Groups by max KS found.")

    # --- Visualization Plot 3: Heatmap of Top HPOs vs Top Cell Groups (Selected by Max KS) ---
    print(f"\nGenerating heatmap for top {top_n} HPOs vs top {top_n} Cell Groups (selected by Max KS)...")
    
    # Get the names/indices for selection
    top_hpo_maxks_names = top_hpos_by_max_ks[hpo_name_col].tolist() if not top_hpos_by_max_ks.empty else []
    top_cg_maxks_names = top_cgs_by_max_ks_series.index.tolist() if not top_cgs_by_max_ks_series.empty else []

    if not top_hpo_maxks_names or not top_cg_maxks_names:
         print("Skipping heatmap: No top HPOs or Cell Groups found based on Max KS.")
    else:
        # Map HPO names back to original DataFrame index to select rows from ks_data
        hpo_original_indices = hpo_info[hpo_info[hpo_name_col].isin(top_hpo_maxks_names)].index
        
        # Ensure indices/columns exist in ks_data before trying to locate
        hpo_original_indices = hpo_original_indices[hpo_original_indices.isin(ks_data.index)]
        top_cg_maxks_names = [cg for cg in top_cg_maxks_names if cg in ks_data.columns]

        if len(hpo_original_indices) == 0 or len(top_cg_maxks_names) == 0:
             print("Skipping heatmap: Mismatch between selected names/indices and data after filtering.")
        else:
            # Create the subset DataFrame
            subset_df = ks_data.loc[hpo_original_indices, top_cg_maxks_names]
            # Set HPO names as index for the heatmap plot
            subset_df.index = hpo_info.loc[hpo_original_indices, hpo_name_col]
            # Reindex columns to match the sorted order of top CGs by max KS
            subset_df = subset_df.reindex(columns=top_cg_maxks_names)
            # Reindex rows to match the sorted order of top HPOs by max KS
            subset_df = subset_df.reindex(index=top_hpo_maxks_names)

            truncated_cg_labels = [truncate_label(label) for label in subset_df.columns]
            heatmap_height = max(8, len(subset_df.index) * 0.35)
            heatmap_width = max(10, len(subset_df.columns) * 0.45)

            plt.figure(figsize=(heatmap_width, heatmap_height))
            sns.heatmap(
                subset_df, cmap="viridis", linewidths=0.5, linecolor='lightgrey',
                xticklabels=truncated_cg_labels, yticklabels=subset_df.index
            )
            plt.title(f'Heatmap (Top {len(subset_df.index)} HPOs x Top {len(subset_df.columns)} CGs by Max KS)', fontsize=14)
            plt.xlabel('Cell Group Identifier (Truncated)', fontsize=12)
            plt.ylabel('HPO Term', fontsize=12)
            plt.xticks(rotation=90, fontsize=8); plt.yticks(fontsize=8)
            plt.tight_layout()
            heatmap_filename = f"{output_prefix}_heatmap_top{top_n}hpo_top{top_n}cg_bymaxks.png"
            try:
                plt.savefig(heatmap_filename, dpi=150, bbox_inches='tight')
                print(f"Saved plot: {heatmap_filename}")
            except Exception as e: print(f"Error saving plot {heatmap_filename}: {e}")
            plt.close()

    plot_comprehensive_scatter(
            ks_data, 
            hpo_info, 
            hpo_id_col, 
            hpo_name_col, 
            output_prefix
        )
    
    create_full_heatmap(
        ks_data, 
        hpo_info, 
        hpo_id_col, 
        hpo_name_col, 
        output_prefix,
        max_cell_groups
    )


    print("\nAnalysis and visualization script finished.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze and Visualize HPO vs Cell Group Matrix (prioritizing high KS).')
    parser.add_argument('input_file', help='Path to the input TSV matrix file.')
    parser.add_argument('-o', '--output_prefix', default='hpo_cg_maxks_analysis',
                        help='Prefix for output plot filenames (default: hpo_cg_maxks_analysis).')
    parser.add_argument('-n', '--top_n', type=int, default=DEFAULT_TOP_N,
                        help=f'Number of top items for rankings and plots (default: {DEFAULT_TOP_N}).')
    parser.add_argument('-m', '--max_cell_groups', type=int, default=DEFAULT_MAX_CELL_GROUPS,
                        help=f'Maximum number of cell groups to include in the plot (default: {DEFAULT_MAX_CELL_GROUPS}).')
    args = parser.parse_args()
    
    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}"); os.makedirs(output_dir)
    
    visualize_and_analyze(
        args.input_file, 
        args.output_prefix, 
        args.top_n,
        args.max_cell_groups
    )
