#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import pronto # For ontology loading
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
import networkx as nx
import sys
import collections

# --- Configuration ---
MAX_LABEL_LENGTH = 75
DEFAULT_TOP_N = 25
DEFAULT_MAX_ROWS_FULL = 500
DEFAULT_MAX_CELL_GROUPS = 600
P_VALUE_EPSILON = 1e-300 # Value to replace p=0 before log10
DEFAULT_MAX_LOG_P = 10 # Default cap for -log10(p) visualization

# --- Helper Functions (Unchanged) ---
def truncate_label(label, max_len=MAX_LABEL_LENGTH):
    """Truncates long string labels."""
    if not isinstance(label, str): label = str(label)
    if len(label) > max_len: return label[:max_len-3] + "..."
    return label

# --- Ontology Processing Functions (Unchanged) ---
term_cache = {}
def get_term_and_ancestors(hpo_id, ontology):
    # (Code as before)
    if hpo_id in term_cache:
        term, ancestor_ids = term_cache[hpo_id]
        ancestor_terms = {ontology[a_id] for a_id in ancestor_ids if a_id in ontology}
        return term, ancestor_terms
    ancestors = set(); term = None
    if hpo_id in ontology:
        term = ontology[hpo_id]; processed_for_ancestors = set(); temp_queue = collections.deque([term])
        while temp_queue:
            current_term = temp_queue.popleft()
            if current_term.id in processed_for_ancestors: continue
            processed_for_ancestors.add(current_term.id)
            try:
                parents = current_term.superclasses(distance=1, with_self=False)
                for parent in parents:
                    if parent.id not in processed_for_ancestors: # Optimization
                        ancestors.add(parent.id)
                        temp_queue.append(parent)
            except Exception as e: print(f"Warning: Could not process parents for {current_term.id}: {e}", file=sys.stderr)
        ancestor_terms = {ontology[a_id] for a_id in ancestors if a_id in ontology}
        term_cache[hpo_id] = (term, ancestors)
        return term, ancestor_terms
    else:
        term_cache[hpo_id] = (None, set())
        return None, set()


def build_ontology_subgraph(ontology, relevant_hpo_ids):
    # (Code as before)
    print("Building relevant ontology subgraph..."); G = nx.DiGraph(); all_nodes_to_add = set(relevant_hpo_ids); missing_in_ont = 0
    print("Finding ancestors...")
    relevant_hpo_ids_in_ont = {hpo for hpo in relevant_hpo_ids if hpo in ontology}
    if len(relevant_hpo_ids_in_ont) < len(relevant_hpo_ids):
        print(f"Warning: {len(relevant_hpo_ids) - len(relevant_hpo_ids_in_ont)} HPO IDs not found in ontology, excluding them from graph.")

    for hpo_id in relevant_hpo_ids_in_ont:
        _, ancestor_ids = get_term_and_ancestors(hpo_id, ontology)
        all_nodes_to_add.update(ancestor_ids)

    valid_nodes = {node_id for node_id in all_nodes_to_add if node_id in ontology}
    G.add_nodes_from(valid_nodes); print(f"Added {len(valid_nodes)} nodes to graph.")
    print("Adding edges..."); edge_count = 0
    for node_id in valid_nodes:
        try:
            term = ontology[node_id]; parents = term.superclasses(distance=1, with_self=False)
            for parent in parents:
                if parent.id in valid_nodes: G.add_edge(parent.id, node_id); edge_count += 1
        except Exception as e: print(f"Warning: Could not process edges for {node_id}: {e}", file=sys.stderr)
    print(f"Added {edge_count} edges."); return G

def calculate_ontology_distances(graph, hpo_ids_for_plot):
    # (Code as before)
    print("Calculating pairwise ontology distances (shortest path)...")
    if not hpo_ids_for_plot: return None
    if graph is None: print("Error: Input graph is None."); return None

    relevant_nodes_in_graph = [node for node in hpo_ids_for_plot if node in graph]
    if len(relevant_nodes_in_graph) < len(hpo_ids_for_plot):
         print(f"Warning: {len(hpo_ids_for_plot) - len(relevant_nodes_in_graph)} HPO IDs for distance calculation not found in the built graph.")
         hpo_ids_for_plot = relevant_nodes_in_graph

    if not hpo_ids_for_plot or len(hpo_ids_for_plot) < 2:
         print("Warning: Too few HPO IDs remain for distance calculation.")
         return None

    try: G_undirected = graph.to_undirected(); print(f"Working with undirected graph ({G_undirected.number_of_nodes()} nodes, {G_undirected.number_of_edges()} edges).")
    except Exception as e: print(f"Error converting graph to undirected: {e}", file=sys.stderr); return None

    n = len(hpo_ids_for_plot); dist_matrix = np.full((n, n), np.inf); np.fill_diagonal(dist_matrix, 0)

    try: components = list(nx.connected_components(G_undirected)); component_map = {node: i for i, comp in enumerate(components) for node in comp}; print(f"Graph has {len(components)} connected components.")
    except Exception as e: print(f"Error finding connected components: {e}", file=sys.stderr); return None

    paths_calculated = 0; no_paths_found = 0
    for i in range(n):
        for j in range(i + 1, n):
            id1 = hpo_ids_for_plot[i]; id2 = hpo_ids_for_plot[j]
            if id1 in G_undirected and id2 in G_undirected and component_map.get(id1) == component_map.get(id2):
                try: length = nx.shortest_path_length(G_undirected, source=id1, target=id2); dist_matrix[i, j] = dist_matrix[j, i] = length; paths_calculated += 1
                except nx.NetworkXNoPath: no_paths_found += 1
                except nx.NodeNotFound: no_paths_found +=1
            else: no_paths_found += 1

    max_finite_dist = np.max(dist_matrix[np.isfinite(dist_matrix)]) if np.any(np.isfinite(dist_matrix)) else 0
    large_distance = float(max(n, max_finite_dist + 1))
    dist_matrix[np.isinf(dist_matrix)] = large_distance

    if no_paths_found > 0: print(f"Warning: Could not find paths or nodes disconnected for {no_paths_found} pairs. Assigned large distance ({large_distance:.1f}).")
    print(f"Calculated {paths_calculated} valid shortest paths.");
    try: condensed_dist = squareform(dist_matrix); return condensed_dist
    except ValueError as e: print(f"Error converting distance matrix to condensed form: {e}. Matrix shape: {dist_matrix.shape}", file=sys.stderr); return None

# --- Heatmap Plotting Function ---
# Now uses standard 'viridis' cmap, expects capped data for p-values
def create_clustered_heatmap(value_matrix_to_cluster, row_labels_to_cluster,
                             row_linkage, col_linkage,
                             output_filename, plot_title,
                             metric_label): # No longer needs data_type for cmap
    """Create a heatmap using precomputed row and column linkages."""
    print(f"Generating heatmap: {plot_title}..."); print(f"Matrix size for plot: {value_matrix_to_cluster.shape}")
    if value_matrix_to_cluster.empty or value_matrix_to_cluster.shape[0] < 2 or value_matrix_to_cluster.shape[1] < 2: print(f"Skipping heatmap '{plot_title}': Matrix too small or empty."); return
    # Add checks for linkage validity more robustly
    if row_linkage is None or not isinstance(row_linkage, np.ndarray) or row_linkage.shape[0] != value_matrix_to_cluster.shape[0] - 1:
        print(f"Skipping heatmap '{plot_title}': Invalid or incompatible row linkage provided. Shape: {row_linkage.shape if row_linkage is not None else 'None'}")
        return
    if col_linkage is None or not isinstance(col_linkage, np.ndarray) or col_linkage.shape[0] != value_matrix_to_cluster.shape[1] - 1:
        print(f"Skipping heatmap '{plot_title}': Invalid or incompatible column linkage provided. Shape: {col_linkage.shape if col_linkage is not None else 'None'}")
        return

    num_rows = value_matrix_to_cluster.shape[0]; num_cols = value_matrix_to_cluster.shape[1]

    base_width = min(max(num_cols * 0.20 + 6, 14), 200)
    base_height = min(max(num_rows * 0.30 + 8, 15), 150)
    print(f"Calculated figure size (inches): ({base_width:.1f}, {base_height:.1f})")

    MAX_ROWS_FOR_LABELS = 150; show_row_labels = num_rows <= MAX_ROWS_FOR_LABELS
    if show_row_labels:
        plot_row_labels = [truncate_label(lbl) for lbl in row_labels_to_cluster]
        ytick_fontsize = max(0.5, min(6, 600 / num_rows)) if num_rows > 0 else 6
    else:
        print(f"INFO: > {MAX_ROWS_FOR_LABELS} rows, hiding Y labels.")
        plot_row_labels = False
        ytick_fontsize = 1

    xtick_fontsize = max(1, min(7, 500 / num_cols)) if num_cols > 0 else 7

    # Use standard viridis; higher transformed/capped p-value is more intense
    cmap = 'viridis'
    cbar_label = metric_label # Label now includes capping info if applied

    try:
        g = sns.clustermap(
            value_matrix_to_cluster, # This data should be capped if p-value
            row_linkage=row_linkage,
            col_linkage=col_linkage,
            cmap=cmap,
            figsize=(base_width, base_height),
            xticklabels=True,
            yticklabels=plot_row_labels,
            linewidths=0.05,
            linecolor='lightgrey',
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            cbar_kws={'label': cbar_label}
            # vmin=0 # Set vmin explicitly to 0 if desired for -log10 scale
        )

        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha='right', fontsize=xtick_fontsize)
        if show_row_labels: plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=ytick_fontsize)

        g.fig.suptitle(plot_title, y=1.01, fontsize=16)

        try:
            for collection in g.ax_row_dendrogram.collections: collection.set_linewidth(0.5)
            for collection in g.ax_col_dendrogram.collections: collection.set_linewidth(0.5)
        except AttributeError: pass

        print(f"Attempting to save heatmap: {output_filename}"); g.savefig(output_filename, dpi=300, bbox_inches='tight'); print(f"Saved heatmap: {output_filename}")
    except MemoryError as me: print(f"MEMORY ERROR generating clustermap: {me}\nConsider reducing --max_rows or --max_cell_groups.", file=sys.stderr)
    except ValueError as ve: print(f"Error generating clustermap (ValueError): {ve}. Check data/linkage compatibility.", file=sys.stderr)
    except Exception as e: print(f"Error generating clustermap: {e}\nMatrix dimensions: {value_matrix_to_cluster.shape}", file=sys.stderr)
    plt.close('all')

# --- Functions for bar plots and scatter ---
# Uses UNcapped -log10(p) or KS values
def plot_top_ranked(data_series, top_n, title, xlabel, ylabel, filename, is_horizontal=False):
    """Plots the top N items from a Pandas Series. Assumes series is pre-sorted (nlargest)."""
    if data_series.empty: print(f"Skipping plot '{title}': No data."); return

    # Data should already be sorted (largest values = most significant)
    top_data = data_series.head(min(top_n, len(data_series)))

    if top_data.empty: print(f"Skipping plot '{title}': No data after filtering."); return

    fig_height = max(6, len(top_data) * 0.35)
    plt.figure(figsize=(10, fig_height))

    labels = [truncate_label(label) for label in top_data.index]
    colors = sns.color_palette("viridis", len(top_data))

    if is_horizontal:
        plt.barh(labels, top_data.values, color=colors)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.6)
    else:
        plt.figure(figsize=(12, 8))
        plt.bar(labels, top_data.values, color=colors)
        plt.xticks(rotation=75, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.title(title, fontsize=14); plt.xlabel(xlabel, fontsize=12); plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout(pad=1.5)

    try: plt.savefig(filename, dpi=150, bbox_inches='tight'); print(f"Saved plot: {filename}")
    except Exception as e: print(f"Error saving plot {filename}: {e}")
    plt.close()

# Uses UNcapped -log10(p) or KS values
def plot_comprehensive_scatter(value_data, hpo_info, hpo_id_col, hpo_name_col, output_prefix,
                               data_type, metric_name_short, metric_name_long): # metric_name_long includes "-log10" if pval
    """Plots a comprehensive scatter plot of HPO metrics."""
    if value_data.empty: print("Skipping scatter plot: No data."); return

    hpo_info_indexed = hpo_info.set_index(hpo_id_col)
    common_index = hpo_info_indexed.index.intersection(value_data.index)
    if common_index.empty: print("Skipping scatter plot: No common HPOs between data and info."); return

    value_data_aligned = value_data.loc[common_index]
    hpo_info_aligned = hpo_info_indexed.loc[common_index].reset_index()

    if value_data_aligned.empty: print("Skipping scatter plot: No aligned data."); return

    # Calculate metrics (Max and Mean of KS or -log10(p))
    hpo_max_value = value_data_aligned.max(axis=1) # Max is always most significant here
    hpo_mean_value = value_data_aligned.mean(axis=1)
    best_value_label = f'Max_{metric_name_short}' # Label uses original short name (KS or P-value)
    best_value_long_label = f'Maximum {metric_name_long}' # Label uses transformed name if pval

    # Define "significant" based on the metric type
    if data_type == 'pvalue':
        # Significance based on original p-value, but check on -log10 scale
        p_threshold = 0.05
        log_p_threshold = -np.log10(p_threshold)
        significant_associations = (value_data_aligned > log_p_threshold)
        significance_label = f'P-value < {p_threshold} (-log10 > {log_p_threshold:.2f})'
        print(f"Scatter Plot: Defining significance as {significance_label}")
    else: # KS
        ks_threshold = 0 # Example KS threshold
        significant_associations = (value_data_aligned > ks_threshold)
        significance_label = f'KS > {ks_threshold}'
        print(f"Scatter Plot: Defining significance as {significance_label}")

    hpo_num_significant = significant_associations.sum(axis=1)

    comprehensive_df = pd.DataFrame({
        'HPO_ID': hpo_info_aligned[hpo_id_col],
        'HPO_Name': hpo_info_aligned[hpo_name_col],
        best_value_label: hpo_max_value.values, # Store max transformed value
        f'Mean_{metric_name_short}': hpo_mean_value.values,
        'Num_Significant_Assoc': hpo_num_significant.values
    })

    # Annotate top points by Max value (highest KS or highest -log10(p))
    annotation_col = best_value_label
    interesting_points = comprehensive_df.nlargest(15, annotation_col)

    # Create Scatter Plot
    plt.figure(figsize=(16, 10))
    scatter = plt.scatter(
        comprehensive_df[f'Mean_{metric_name_short}'],
        comprehensive_df['Num_Significant_Assoc'],
        c=comprehensive_df[best_value_label], # Color by max transformed value
        cmap='viridis',
        alpha=0.7,
        s=50,
        edgecolors='black',
        linewidth=0.5
    )

    plt.colorbar(scatter, label=best_value_long_label) # Label uses transformed name
    plt.title(f'Comprehensive HPO Analysis: {metric_name_short} Overview', fontsize=16)
    plt.xlabel(f'Mean {metric_name_long} Across Cell Groups', fontsize=12)
    plt.ylabel(f'Number of Significant Associations ({significance_label})', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Annotate interesting points
    for _, row in interesting_points.iterrows():
        plt.annotate(
            truncate_label(row['HPO_Name'], max_len=30),
            (row[f'Mean_{metric_name_short}'], row['Num_Significant_Assoc']),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", lw=0.5, alpha=0.6)
        )

    filename = f"{output_prefix}_comprehensive_hpo_analysis_{metric_name_short.lower()}.png"
    plt.tight_layout()
    try: plt.savefig(filename, dpi=300, bbox_inches='tight'); print(f"Saved comprehensive analysis plot: {filename}")
    except Exception as e: print(f"Error saving comprehensive plot {filename}: {e}")
    plt.close()

    # Save summary text file
    summary_filename = f"{output_prefix}_comprehensive_hpo_analysis_summary_{metric_name_short.lower()}.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"Comprehensive HPO Analysis Summary ({metric_name_short})\n")
        f.write("="*50 + "\n\n")
        f.write(f"Top 10 HPO Groups by {best_value_long_label}:\n")
        top_10_best_val = comprehensive_df.nlargest(10, best_value_label) # Always largest now
        f.write(top_10_best_val.to_string(index=False) + "\n\n")
        f.write("Summary Statistics:\n")
        f.write(f"Total HPO Groups Analyzed: {len(comprehensive_df)}\n")
        f.write(f"Mean {best_value_long_label} Across HPO Groups: {comprehensive_df[best_value_label].mean():.4f}\n")
        f.write(f"Median {best_value_long_label} Across HPO Groups: {comprehensive_df[best_value_label].median():.4f}\n")
        f.write(f"Mean Number of Significant Associations: {comprehensive_df['Num_Significant_Assoc'].mean():.2f}\n")
        f.write(f"(Significance defined as {significance_label})\n")
    print(f"Saved comprehensive analysis summary: {summary_filename}")


# --- Main Visualization and Analysis Logic ---
def visualize_and_analyze(args):
    """Loads data, performs analyses, and generates visualizations."""
    global term_cache; term_cache = {}

    # --- Validate Data Type ---
    if args.type_of_data not in ['ks', 'pvalue']:
        print(f"Error: Invalid --type_of_data '{args.type_of_data}'. Must be 'ks' or 'pvalue'.")
        sys.exit(1)
    data_type = args.type_of_data
    print(f"Data type specified: {data_type}")

    # --- Load Matrix Data ---
    print(f"Loading matrix data from: {args.input_file}")
    try: # Data Loading
        df = pd.read_csv(args.input_file, sep='\t', low_memory=False)
        print(f"Matrix loaded successfully. Shape: {df.shape}")
        if df.shape[1] < 3: raise ValueError("Matrix needs >= 3 columns (ID, Name, CellGroups...).")
        hpo_id_col = df.columns[0]; hpo_name_col = df.columns[1]
        print(f"Assuming HPO ID: '{hpo_id_col}', HPO Name: '{hpo_name_col}'")
        hpo_info = df[[hpo_id_col, hpo_name_col]].copy().drop_duplicates(subset=[hpo_id_col]).set_index(hpo_id_col)

        value_data_raw = df.iloc[:, 2:]
        value_data_raw.index = df[hpo_id_col]
        value_data_raw = value_data_raw[~value_data_raw.index.duplicated(keep='first')]

        # --- Process based on data type (Apply -log10 to p-values) ---
        if data_type == 'pvalue':
            metric_name_short = "P-value"
            metric_name_long = "-log10(P-value)" # Label uses transformed name
            print(f"Processing p-value data. Applying {metric_name_long} transformation.")
            print(f"Replacing p-values of 0 with {P_VALUE_EPSILON} before log.")
            value_data_numeric = value_data_raw.apply(pd.to_numeric, errors='coerce')
            # Replace 0s and handle invalid values before log
            value_data_numeric[value_data_numeric == 0] = P_VALUE_EPSILON
            value_data_numeric[(value_data_numeric < 0) | (value_data_numeric > 1)] = np.nan # Set invalid p-values to NaN

            value_data = -np.log10(value_data_numeric)
            # Fill NaNs resulting from transformation or original NaNs with 0 (representing p=1 / non-significant)
            value_data = value_data.fillna(0)
            value_data.replace([np.inf, -np.inf], 0, inplace=True) # Handle potential infinities from epsilon
            print(f"{metric_name_long} transformation complete. Higher values indicate greater significance.")
        else: # KS data
            metric_name_short = "KS"
            metric_name_long = "KS Statistic"
            value_data_numeric = value_data_raw.apply(pd.to_numeric, errors='coerce')
            # Fill NaNs with 0 for KS
            value_data = value_data_numeric.fillna(0)
            print("KS data loaded. Higher values indicate stronger association. Missing values set to 0.")

        hpo_info = hpo_info.loc[value_data.index]

    except FileNotFoundError: print(f"Error: Input file not found at {args.input_file}"); return
    except ValueError as ve: print(f"Error processing data columns: {ve}"); return
    except Exception as e: print(f"Error loading or processing data: {e}"); return

    if value_data.empty: print("Error: No valid numeric data found after processing."); return

    # --- Load Ontology ---

    print(f"\nLoading HPO ontology from: {args.hpo_obo}")
    try: ontology = pronto.Ontology(args.hpo_obo); print("Ontology loaded.")
    except FileNotFoundError: print(f"Error: HPO OBO file not found at {args.hpo_obo}"); return
    except Exception as e: print(f"Error loading ontology: {e}"); return

    # --- Basic Summaries & Top Lists (Use transformed data, nlargest) ---
    print(f"\n--- Analysis 1, 2, 3 & Plots 1, 2 (Based on {metric_name_long}) ---")

    # Analysis 1: Top HPO-Cell Group Pairs
    print(f"\n--- Analysis 1: Top {args.top_n} HPO-Cell Group Pairs by Highest {metric_name_long} ---")
    value_stacked = value_data.stack()
    if value_stacked.empty:
        print("No associations found.")
        top_pairs_final_df = pd.DataFrame()
    else:
        # Always use nlargest now
        top_pairs_series = value_stacked.nlargest(args.top_n)
        print(f"Ranking by largest {metric_name_long} (most significant).")
        top_pairs_df = top_pairs_series.reset_index()
        top_pairs_df.columns = [hpo_id_col, 'cell_group', metric_name_long]
        top_pairs_final_df = pd.merge(top_pairs_df, hpo_info.reset_index()[[hpo_id_col, hpo_name_col]], on=hpo_id_col, how='left')
        top_pairs_final_df = top_pairs_final_df[[hpo_id_col, hpo_name_col, 'cell_group', metric_name_long]]
        print(f"Top {min(args.top_n, len(top_pairs_final_df))} pairs:")
        print(top_pairs_final_df.to_string(index=False, float_format="%.4f"))
    print("-" * 60)

    # Analysis 2 & Plot 1: Top HPOs by Max Value
    print(f"\n--- Analysis 2 & Plot 1: Top {args.top_n} HPOs by Max {metric_name_long} ---")
    max_value_per_hpo = value_data.max(axis=1) # Max is always best now
    best_label = f'max_{metric_name_long}'

    if max_value_per_hpo.isnull().all() or max_value_per_hpo.empty:
        print(f"Cannot calculate max {metric_name_long} per HPO.")
    else:
        top_hpo_series = max_value_per_hpo.nlargest(args.top_n)
        top_hpos_by_max_value_df = hpo_info.loc[top_hpo_series.index].copy()
        top_hpos_by_max_value_df[best_label] = top_hpo_series
        top_hpos_by_max_value_df = top_hpos_by_max_value_df.sort_values(by=best_label, ascending=False) # Always descending

        print(f"Top {min(args.top_n, len(top_hpos_by_max_value_df))} HPOs ranked:")
        print(top_hpos_by_max_value_df[[hpo_name_col, best_label]].to_string(index=False, float_format="%.4f"))

        plot_data_hpo = top_hpos_by_max_value_df.set_index(hpo_name_col)[best_label]
        plot_top_ranked(plot_data_hpo, args.top_n,
                        f'Top {len(plot_data_hpo)} HPO Groups by Highest Max {metric_name_long}',
                        f'Maximum {metric_name_long}', 'HPO Group',
                        f"{args.output_prefix}_top_{args.top_n}_hpo_by_max_{data_type}.png",
                        is_horizontal=False)
    print("-" * 60)

    # Analysis 3 & Plot 2: Top Cell Groups by Max Value
    print(f"\n--- Analysis 3 & Plot 2: Top {args.top_n} Cell Groups by Max {metric_name_long} ---")
    max_value_per_cg = value_data.max(axis=0)
    best_label_cg = f'max_{metric_name_long}'

    if max_value_per_cg.isnull().all() or max_value_per_cg.empty:
        print(f"Cannot calculate max {metric_name_long} per Cell Group.")
    else:
        top_cgs_by_max_value_series = max_value_per_cg.nlargest(args.top_n)
        top_cgs_by_max_value_df = top_cgs_by_max_value_series.reset_index()
        top_cgs_by_max_value_df.columns = ['cell_group', best_label_cg]
        print(f"Top {min(args.top_n, len(top_cgs_by_max_value_df))} Cell Groups ranked:")
        top_cgs_by_max_value_df['cell_group_trunc'] = top_cgs_by_max_value_df['cell_group'].apply(truncate_label)
        print(top_cgs_by_max_value_df[['cell_group_trunc', best_label_cg]].to_string(index=False, float_format="%.4f"))

        plot_top_ranked(top_cgs_by_max_value_series, args.top_n,
                        f'Top {len(top_cgs_by_max_value_series)} Cell Groups by Highest Max {metric_name_long}',
                        f'Maximum {metric_name_long}', 'Cell Group Identifier',
                        f"{args.output_prefix}_top_{args.top_n}_cellgroup_by_max_{data_type}.png",
                        is_horizontal=True)
    print("-" * 60)

    # Comprehensive Scatter Plot (uses untransformed data)
    print(f"\nGenerating comprehensive scatter plot for {metric_name_short}...")
    plot_comprehensive_scatter(value_data, hpo_info.reset_index(), hpo_id_col, hpo_name_col,
                               args.output_prefix, data_type, metric_name_short, metric_name_long)
    print("-" * 60)


    # --- Prepare Linkages and Data for Heatmap 1 (Large / Filtered) ---
    print(f"\nPreparing Heatmap 1: Filtered data (Max Rows: {args.max_rows}, Max Cols: {args.max_cell_groups})")
    value_matrix_large = value_data.copy() # This is transformed data
    hpo_info_large = hpo_info.copy()

    # Filter Rows (HPOs) based on maximum value (always use nlargest on transformed data)
    if value_matrix_large.shape[0] > args.max_rows:
        print(f"Filtering rows: Selecting top {args.max_rows} HPO groups by max {metric_name_long}.")
        hpo_max_scores = value_matrix_large.max(axis=1)
        top_hpo_ids = hpo_max_scores.nlargest(args.max_rows).index
        value_matrix_large = value_matrix_large.loc[top_hpo_ids]
        hpo_info_large = hpo_info_large.loc[top_hpo_ids]

    # Filter Columns (Cell Groups) based on maximum value (always use nlargest on transformed data)
    if value_matrix_large.shape[1] > args.max_cell_groups:
        print(f"Filtering columns: Selecting top {args.max_cell_groups} cell groups by max {metric_name_long}.")
        cg_max_scores = value_matrix_large.max(axis=0)
        top_cg_names = cg_max_scores.nlargest(args.max_cell_groups).index
        value_matrix_large = value_matrix_large[top_cg_names]

    print(f"Final matrix dimensions for Heatmap 1: {value_matrix_large.shape}")

    # --- Capping for Heatmap Visualization ---
    value_matrix_large_heatmap = value_matrix_large.copy()
    heatmap_metric_label = metric_name_long # Base label
    if data_type == 'pvalue' and args.max_log_p is not None:
         print(f"Applying cap for heatmap: {metric_name_long} > {args.max_log_p} will be set to {args.max_log_p}.")
         value_matrix_large_heatmap[value_matrix_large_heatmap > args.max_log_p] = args.max_log_p
         heatmap_metric_label = f"{metric_name_long} [Capped at {args.max_log_p}]"
    # --- End Capping ---

    if not value_matrix_large_heatmap.empty and value_matrix_large_heatmap.shape[0] > 1 and value_matrix_large_heatmap.shape[1] > 1:
        row_ids_large = list(value_matrix_large_heatmap.index)
        ontology_graph_large = build_ontology_subgraph(ontology, row_ids_large)
        ontology_dist_large = calculate_ontology_distances(ontology_graph_large, row_ids_large)
        row_linkage_ont_large = None; col_linkage_data_large = None

        if ontology_dist_large is not None and len(ontology_dist_large) > 0 :
             try: row_linkage_ont_large = sch.linkage(ontology_dist_large, method='average'); print("Calculated ontology row linkage for Heatmap 1.")
             except Exception as e: print(f"Error H1 ontology row linkage: {e}", file=sys.stderr)
        else: print("Skipping H1 ontology row linkage calculation: Invalid/empty distance matrix.")

        try: # Column linkage based on data (use the capped data for clustering cols)
             data_for_linkage_T = value_matrix_large_heatmap.T.fillna(0) # Fill NaNs with 0 (non-sig)
             if data_for_linkage_T.shape[0] > 1 :
                  condensed_col_dist_large = pdist(data_for_linkage_T, metric='euclidean')
                  col_linkage_data_large = sch.linkage(condensed_col_dist_large, method='average')
                  print("Calculated data column linkage for Heatmap 1.")
             else: print("Skipping H1 data column linkage: Not enough columns.")
        except Exception as e: print(f"Error H1 data column linkage: {e}", file=sys.stderr)

        row_labels_large = hpo_info_large.loc[value_matrix_large_heatmap.index, hpo_name_col].tolist()
        plot_title_h1 = f"HPO vs Cell Group {metric_name_short} (Top {value_matrix_large_heatmap.shape[0]} HPO x Top {value_matrix_large_heatmap.shape[1]} CG)"
        create_clustered_heatmap(value_matrix_large_heatmap, # Pass capped data
                                 row_labels_large, row_linkage_ont_large, col_linkage_data_large,
                                 f"{args.output_prefix}_heatmap_large_ontrows_datacols_{data_type}.png",
                                 plot_title_h1,
                                 heatmap_metric_label) # Pass label with capping info
    else: print("Skipping large heatmap: Matrix too small or empty after filtering.")
    print("-" * 60)

    # --- Prepare for Heatmap 2 (Specific Selected HPOs) ---
    if args.select_hpos:
        print(f"\nPreparing Heatmap 2: Selected HPO Groups ({len(args.select_hpos)} requested)")
        selected_ids_in_matrix = [hpo_id for hpo_id in args.select_hpos if hpo_id in value_data.index]
        missing_hpos = set(args.select_hpos) - set(selected_ids_in_matrix)
        if missing_hpos: print(f"Warning: The following selected HPOs were not found in the data: {', '.join(missing_hpos)}")

        if not selected_ids_in_matrix: print("Warning: None specified HPOs found. Skipping selected heatmap.")
        elif len(selected_ids_in_matrix) < 2: print("Warning: Fewer than 2 selected HPOs found. Skipping selected heatmap.")
        else:
            value_matrix_selected = value_data.loc[selected_ids_in_matrix].copy() # Transformed data
            hpo_info_selected = hpo_info.loc[selected_ids_in_matrix].copy()
            print(f"Found {len(selected_ids_in_matrix)} selected HPOs in the matrix.")

            # Filter Columns for selected heatmap based on max value within selection (use nlargest on transformed)
            if value_matrix_selected.shape[1] > args.max_cell_groups:
                print(f"Filtering columns for selection: Selecting top {args.max_cell_groups} cell groups by max {metric_name_long} within selection.")
                cg_max_scores_sel = value_matrix_selected.max(axis=0)
                top_cg_names_sel = cg_max_scores_sel.nlargest(args.max_cell_groups).index
                value_matrix_selected = value_matrix_selected[top_cg_names_sel]

            print(f"Final matrix dimensions for Heatmap 2: {value_matrix_selected.shape}")

             # --- Capping for Heatmap Visualization ---
            value_matrix_selected_heatmap = value_matrix_selected.copy()
            heatmap_metric_label_sel = metric_name_long # Base label
            if data_type == 'pvalue' and args.max_log_p is not None:
                 print(f"Applying cap for selected heatmap: {metric_name_long} > {args.max_log_p} will be set to {args.max_log_p}.")
                 value_matrix_selected_heatmap[value_matrix_selected_heatmap > args.max_log_p] = args.max_log_p
                 heatmap_metric_label_sel = f"{metric_name_long} [Capped at {args.max_log_p}]"
            # --- End Capping ---


            if not value_matrix_selected_heatmap.empty and value_matrix_selected_heatmap.shape[0] > 1 and value_matrix_selected_heatmap.shape[1] > 1:
                 row_ids_selected = list(value_matrix_selected_heatmap.index)
                 ontology_graph_sel = build_ontology_subgraph(ontology, row_ids_selected)
                 ontology_dist_sel = calculate_ontology_distances(ontology_graph_sel, row_ids_selected)
                 row_linkage_ont_sel = None; col_linkage_data_sel = None

                 if ontology_dist_sel is not None and len(ontology_dist_sel) > 0:
                      try: row_linkage_ont_sel = sch.linkage(ontology_dist_sel, method='average'); print("Calculated ontology row linkage for selection.")
                      except Exception as e: print(f"Error H2 ontology row linkage: {e}", file=sys.stderr)
                 else: print("Skipping H2 ontology row linkage calculation: Invalid/empty distance matrix.")

                 try: # Column linkage based on data (use capped data)
                     data_for_linkage_sel_T = value_matrix_selected_heatmap.T.fillna(0)
                     if data_for_linkage_sel_T.shape[0] > 1:
                          condensed_col_dist_sel = pdist(data_for_linkage_sel_T, metric='euclidean')
                          col_linkage_data_sel = sch.linkage(condensed_col_dist_sel, method='average')
                          print("Calculated data column linkage for selection.")
                     else: print("Skipping H2 data column linkage: Not enough columns.")
                 except Exception as e: print(f"Error H2 data column linkage: {e}", file=sys.stderr)

                 row_labels_selected = hpo_info_selected.loc[value_matrix_selected_heatmap.index, hpo_name_col].tolist()
                 plot_title_h2 = f"Selected HPO Groups ({value_matrix_selected_heatmap.shape[0]}) vs Top {value_matrix_selected_heatmap.shape[1]} CG ({metric_name_short})"
                 create_clustered_heatmap(value_matrix_selected_heatmap, # Pass capped data
                                          row_labels_selected, row_linkage_ont_sel, col_linkage_data_sel,
                                          f"{args.output_prefix}_heatmap_selected_ontrows_datacols_{data_type}.png",
                                          plot_title_h2,
                                          heatmap_metric_label_sel) # Pass label with capping info
            else: print("Skipping selected heatmap: Matrix too small or empty after filtering columns/rows.")
    else: print("\nSkipping Heatmap 2: No specific HPO groups selected via --select_hpos.")

    print("\nAnalysis and visualization script finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    try: from scipy.spatial.distance import pdist
    except ImportError: print("Error: scipy.spatial.distance.pdist not found. Please ensure scipy is installed."); sys.exit(1)

    parser = argparse.ArgumentParser(description='Analyze HPO Group vs Cell Group Matrix (KS or P-value). Applies -log10 transformation to P-values and allows capping for heatmap visualization.')
    parser.add_argument('input_file', help='Path to the input TSV matrix file (HPO ID, HPO Name, CellGroup1, CellGroup2, ...).')
    parser.add_argument('--hpo_obo', type=str, required=True, help="Path to the HPO ontology file (hp.obo).")
    parser.add_argument('--type_of_data', type=str, required=True, choices=['ks', 'pvalue'], help='Type of input matrix values ("ks" or "pvalue"). P-values will be -log10 transformed.')
    parser.add_argument('-o', '--output_prefix', default='hpo_cg_ont_analysis', help='Prefix for output plot filenames.')
    parser.add_argument('-n', '--top_n', type=int, default=DEFAULT_TOP_N, help=f'Number of top items for ranking plots (default: {DEFAULT_TOP_N}). Ranks by highest KS or -log10(p).')
    parser.add_argument('--max_rows', type=int, default=DEFAULT_MAX_ROWS_FULL, help=f'Max HPO rows for large heatmap (filtered by max value, default: {DEFAULT_MAX_ROWS_FULL}).')
    parser.add_argument('--max_cell_groups', type=int, default=DEFAULT_MAX_CELL_GROUPS, help=f'Max Cell Group columns for heatmaps (filtered by max value, default: {DEFAULT_MAX_CELL_GROUPS}).')
    parser.add_argument('--select_hpos', type=str, nargs='+', required=False, help='List of specific HPO Group IDs (e.g., HP:000123 HP:000456) for a separate heatmap.')
    parser.add_argument('--max_log_p', type=float, default=DEFAULT_MAX_LOG_P, help=f'Maximum -log10(P-value) to use for heatmap color scaling. Values above this will be capped. (Default: {DEFAULT_MAX_LOG_P})')


    args = parser.parse_args()

    output_dir = os.path.dirname(args.output_prefix)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    visualize_and_analyze(args)
