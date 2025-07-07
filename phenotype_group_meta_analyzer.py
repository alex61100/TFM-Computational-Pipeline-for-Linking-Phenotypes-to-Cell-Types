#!/usr/bin/env python3

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import pronto
import os
import sys
import time
import argparse

# --- Helper Function: Get Relevant Descendants ---

def get_relevant_descendants(hpo_term, matrix_hpos_set):
    """Gets descendants and filters by presence in matrix."""
    try:
        descendants = {
            term.id for term in hpo_term.subclasses(with_self=True, distance=None)
            if term.id in matrix_hpos_set
        }
        return descendants
    except Exception as e:
        print(f"Warning: Error getting descendants for {hpo_term.id}: {e}", file=sys.stderr)
        return set()

# --- Recursive function to find candidate groups ---

def find_groups_recursive(hpo_term, matrix_hpos_set, min_hpos_threshold, ontology, groups_list, analyzed_ids_set):
    """Recursively adds children if parent meets threshold."""
    parent_descendants = get_relevant_descendants(hpo_term, matrix_hpos_set)
    parent_meets_threshold = len(parent_descendants) >= min_hpos_threshold

    try:
        children = list(hpo_term.subclasses(distance=1, with_self=False))
    except Exception as e:
        print(f"Warning: Could not get children for {hpo_term.id}: {e}", file=sys.stderr)
        children = []

    if parent_meets_threshold:
        for child in children:
            if child.id not in analyzed_ids_set:
                child_descendants = get_relevant_descendants(child, matrix_hpos_set)
                child_name = child.name if child.name else child.id
                print(f"  -> Adding child {child.id} ({child_name}) for analysis because parent {hpo_term.id} met threshold.")
                groups_list.append({
                    'id': child.id,
                    'name': child_name,
                    'hpo_set': child_descendants # Store child's own descendants
                })
                analyzed_ids_set.add(child.id)

    for child in children:
        find_groups_recursive(child, matrix_hpos_set, min_hpos_threshold, ontology, groups_list, analyzed_ids_set)


# --- Main Analysis Function ---
def run_analysis(args):
    """Runs the main HPO group analysis workflow."""
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    # --- Setup ---

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory: {args.output_dir}")
    except OSError as e:
        print(f"ERROR: Could not create output directory '{args.output_dir}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading HPO ontology from: {args.hpo_obo} ...")
    try:
        ontology = pronto.Ontology(args.hpo_obo)
        print("HPO ontology loaded successfully.")
    except FileNotFoundError:
        print(f"ERROR: HPO ontology file not found at '{args.hpo_obo}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load ontology: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading pre-calculated KS matrix from: {args.input_ks_matrix} ...")
    try:
        ks_input_matrix = pd.read_csv(args.input_ks_matrix, sep='\t', index_col=0)
        if ks_input_matrix.index.name is None:
            ks_input_matrix.index.name = 'HPO_ID'
        print(f"Matrix loaded: {ks_input_matrix.shape[0]} HPOs, {ks_input_matrix.shape[1]} columns.")
    except FileNotFoundError:
        print(f"ERROR: Input matrix file not found at '{args.input_ks_matrix}'", file=sys.stderr)
        sys.exit(1)
    except pd.errors.EmptyDataError:
         print(f"ERROR: Input matrix file '{args.input_ks_matrix}' is empty.", file=sys.stderr)
         sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load input matrix: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nIdentifying cell type columns with prefix: '{args.column_prefix}' ...")
    all_columns = ks_input_matrix.columns.tolist()
    global_celltype_columns = [col for col in all_columns if col.startswith(args.column_prefix)]

    if not global_celltype_columns:
        print(f"WARNING: No columns found starting with '{args.column_prefix}'.", file=sys.stderr)
        sys.exit("Error: No target columns found for analysis.")
    else:
        print(f"Found {len(global_celltype_columns)} cell type columns to analyze.")

    # --- HPO Group Identification ---

    matrix_hpos = set(ks_input_matrix.index)
    print(f"\nFound {len(matrix_hpos)} unique HPOs in the matrix index.")

    try:
        start_node = ontology[args.start_hpo_id]
        print(f"Start node found: {start_node.id} ({start_node.name})")
    except KeyError:
        print(f"ERROR: Start HPO ID '{args.start_hpo_id}' not found in the ontology.", file=sys.stderr)
        sys.exit(1)

    candidate_groups = []
    analyzed_group_ids = set()
    print(f"\nRecursively searching descendants of {args.start_hpo_id}...")
    print(f"Rule: Children are added for analysis if their PARENT has >= {args.min_hpos} relevant descendants.")
    find_groups_recursive(
        start_node, matrix_hpos, args.min_hpos, ontology, candidate_groups, analyzed_group_ids
    )

    if not candidate_groups:
        print("\nNo candidate groups were added based on the parent threshold rule. Exiting script.")
        sys.exit(0)
    else:
        candidate_groups.sort(key=lambda x: x['id'])
        print(f"\nFound {len(candidate_groups)} candidate groups added for analysis.")

    # --- Secondary KS Calculation & Initial NaN Row Filtering ---

    print("\nInitiating secondary KS analysis for each added candidate group and cell type...")
    results_ks_stat = {}
    results_p_value = {}

    for group_info in candidate_groups:
        group_id = group_info['id']
        group_name = group_info['name']
        hpo_group_set = group_info['hpo_set']

        if not hpo_group_set:
             print(f"  INFO: Descendant set ('hpo_set') for group {group_id} ({group_name}) is empty. Skipping this group.", file=sys.stderr)
             continue

        # print(f"\nProcessing group: {group_id} ({group_name})") # Reduced verbosity
        hpo_other_set = matrix_hpos - hpo_group_set
        group_ks_results = {}
        group_pvalue_results = {}

        if not hpo_other_set:
             print(f"  INFO: The 'other' HPO set is empty for group {group_id}. Skipping KS tests for this group.", file=sys.stderr)
             continue

        nan_found_in_row = False
        for cell_type_col in global_celltype_columns:
            meta_ks_stat, meta_p_value = np.nan, np.nan
            try:
                valid_group_hpos = list(hpo_group_set.intersection(ks_input_matrix.index))
                valid_other_hpos = list(hpo_other_set.intersection(ks_input_matrix.index))

                if valid_group_hpos and valid_other_hpos:
                    ks_values_group = ks_input_matrix.loc[valid_group_hpos, cell_type_col].dropna().values
                    ks_values_other = ks_input_matrix.loc[valid_other_hpos, cell_type_col].dropna().values

                    if len(ks_values_group) >= args.min_obs_ks and len(ks_values_other) >= args.min_obs_ks:
                        meta_ks_stat, meta_p_value = ks_2samp(ks_values_group, ks_values_other, alternative='two-sided')
                        if np.isnan(meta_ks_stat) or np.isnan(meta_p_value):
                             nan_found_in_row = True
                    else:
                         nan_found_in_row = True
                else:
                    nan_found_in_row = True

            except Exception as e:
                print(f"  ERROR processing {cell_type_col} for group {group_id}: {e}. Assigning NaN.", file=sys.stderr)
                meta_ks_stat, meta_p_value = np.nan, np.nan
                nan_found_in_row = True

            group_ks_results[cell_type_col] = meta_ks_stat
            group_pvalue_results[cell_type_col] = meta_p_value

        if nan_found_in_row:
            print(f"  INFO: Group {group_id} ({group_name}) contained NaN/failed checks. Excluding entire row.")
        else:
            results_ks_stat[group_id] = group_ks_results
            results_p_value[group_id] = group_pvalue_results

    # --- Generate Intermediate Output Matrices ---
    print("\nGenerating intermediate result matrices...")

    if not results_ks_stat:
         print("No results were generated (no groups passed initial NaN filters).")
         sys.exit(0)

    meta_ks_matrix = pd.DataFrame.from_dict(results_ks_stat, orient='index')
    meta_pvalue_matrix = pd.DataFrame.from_dict(results_p_value, orient='index')
    meta_ks_matrix.index.name = 'HPO_Group_ID'
    meta_pvalue_matrix.index.name = 'HPO_Group_ID'

    print(f"Generated intermediate matrices with {len(meta_pvalue_matrix)} groups and {len(meta_pvalue_matrix.columns)} data columns.")

    # Store original indices and columns before potential filtering
    original_rows = meta_pvalue_matrix.index
    original_cols = [col for col in global_celltype_columns if col in meta_pvalue_matrix.columns]


    # <<< MODIFIED: Optional Combined Row and Column P-value Filtering >>>
    rows_to_keep_indices = original_rows # Start assuming all rows (post-NaN) are kept
    columns_to_keep = original_cols # Start with all present data columns

    if args.min_pvalue is not None:
        print(f"\n--min_pvalue provided ({args.min_pvalue}). Identifying rows and columns to keep based on threshold...")

        if not 0.0 <= args.min_pvalue <= 1.0:
            print(f"  ERROR: --min_pvalue must be between 0.0 and 1.0. Skipping combined filter step.", file=sys.stderr)
            # Keep rows_to_keep_indices and columns_to_keep as they were (all rows/cols)
        else:
            # Identify rows to keep (based on the *original* set of columns)
            print(f"  Identifying Rows: Checking rows with at least one p-value <= {args.min_pvalue} across original columns...")
            if original_cols: # Check if there are columns to check against
                 row_mask = (meta_pvalue_matrix[original_cols].le(args.min_pvalue)).any(axis=1)
                 rows_to_keep_indices = original_rows[row_mask] # Select indices using the mask
                 n_rows_before = len(original_rows)
                 n_rows_after = len(rows_to_keep_indices)
                 print(f"  Identified {n_rows_after} of {n_rows_before} rows to potentially keep.")
            else:
                 print("  WARNING: No data columns present to perform row identification check.", file=sys.stderr)
                 rows_to_keep_indices = original_rows # Keep all rows if no columns to check

            # Identify columns to keep (based on the *original* set of rows)
            print(f"  Identifying Columns: Checking columns with at least one p-value <= {args.min_pvalue} across original rows...")
            columns_to_keep = [] # Reset
            if not meta_pvalue_matrix.empty: # Check if there are rows to check against
                for col in original_cols:
                    # Check if ANY p-value in this column (within the full matrix) is <= threshold
                    if (meta_pvalue_matrix[col].le(args.min_pvalue)).any():
                        columns_to_keep.append(col)
            else:
                 print("  WARNING: No data rows present to perform column identification check.", file=sys.stderr)


            n_cols_before = len(original_cols)
            n_cols_after = len(columns_to_keep)
            print(f"  Identified {n_cols_after} of {n_cols_before} columns to potentially keep.")

            # Apply the combined filter only if both rows and columns are identified
            print("\nApplying combined row and column selections...")
            if rows_to_keep_indices.empty or not columns_to_keep:
                 print("  Resulting matrix will be empty due to row or column filtering criteria not being met.")
                 # Make matrices empty but preserve column structure for KS
                 meta_ks_matrix = pd.DataFrame(index=[], columns=meta_ks_matrix.columns)
                 meta_pvalue_matrix = pd.DataFrame(index=[], columns=meta_pvalue_matrix.columns)
            else:
                 # Use .loc for simultaneous row and column selection based on identified keepers
                 # We select from the original matrices using the identified indices/columns
                 meta_ks_matrix = meta_ks_matrix.loc[rows_to_keep_indices, columns_to_keep].copy()
                 meta_pvalue_matrix = meta_pvalue_matrix.loc[rows_to_keep_indices, columns_to_keep].copy()
                 print(f"  Filtered matrices shape: {meta_pvalue_matrix.shape}")

    else:
        print("\n--min_pvalue not provided. Skipping combined p-value filtering.")
        # rows_to_keep_indices and columns_to_keep retain their initial values (all post-NaN rows, all present columns)


    # --- Prepare Final Output ---
    print("\nPreparing final output files...")

    # Check if matrices are empty AFTER potential combined filtering
    if meta_pvalue_matrix.empty:
         print("Resulting matrix is empty after filtering steps. No output files generated.")
    # This check might be redundant if columns_to_keep is empty, but safe.
    elif not columns_to_keep:
         print("No data columns remaining after filtering. No output files generated.")
    else:
        # Add HPO group names column (only if matrix isn't empty)
        # Use the index of the potentially filtered matrix
        group_names_map = {g['id']: g['name'] for g in candidate_groups if g['id'] in meta_pvalue_matrix.index}
        meta_ks_matrix.insert(0, 'HPO_Group_Name', meta_ks_matrix.index.map(group_names_map))
        meta_pvalue_matrix.insert(0, 'HPO_Group_Name', meta_pvalue_matrix.index.map(group_names_map))

        # Reset index
        meta_ks_matrix_viz = meta_ks_matrix.reset_index()
        meta_pvalue_matrix_viz = meta_pvalue_matrix.reset_index()

        # Define the final set of columns (ID, Name, and the kept data columns)
        # columns_to_keep has already been determined
        final_ordered_cols = ['HPO_Group_ID', 'HPO_Group_Name'] + sorted(columns_to_keep)

        # Ensure columns exist before selecting (should be guaranteed by logic above, but safe check)
        # Select the final columns from the potentially filtered dataframes
        final_ordered_cols_present_ks = [col for col in final_ordered_cols if col in meta_ks_matrix_viz.columns]
        final_ordered_cols_present_pval = [col for col in final_ordered_cols if col in meta_pvalue_matrix_viz.columns]

        meta_ks_matrix_viz = meta_ks_matrix_viz[final_ordered_cols_present_ks]
        meta_pvalue_matrix_viz = meta_pvalue_matrix_viz[final_ordered_cols_present_pval]

        # Define output paths
        output_ks_path = os.path.join(args.output_dir, 'meta_ks_statistic.tsv')
        output_pvalue_path = os.path.join(args.output_dir, 'meta_pvalue.tsv')

        # Save the resulting matrices
        try:
            meta_ks_matrix_viz.to_csv(output_ks_path, sep='\t', index=False, na_rep='NaN')
            print(f"KS statistics matrix saved ({len(columns_to_keep)} data columns): {output_ks_path}")
            meta_pvalue_matrix_viz.to_csv(output_pvalue_path, sep='\t', index=False, na_rep='NaN')
            print(f"KS p-value matrix saved ({len(columns_to_keep)} data columns): {output_pvalue_path}")
        except Exception as e:
            print(f"ERROR: Failed to save result files: {e}", file=sys.stderr)


    end_time = time.time()
 
    print(f"\nAnalysis finished in {end_time - start_time:.2f} seconds.")
    print(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Recursively analyze HPO groups based on pre-calculated KS statistics against cell types. "
                    "Adds children for analysis if their PARENT meets the --min_hpos threshold. "
                    "Excludes rows with any NaN result. "
                    "Optionally filters output rows AND columns based on --min_pvalue." # <<< MODIFIED >>> description
    )
    parser.add_argument('--hpo_obo', type=str, required=True, help="Path to the HPO ontology file (hp.obo).")
    parser.add_argument('--input_ks_matrix', type=str, required=True, help="Path to the input TSV/CSV matrix (Rows: HPO_ID index, Columns: Cell Types, Values: Pre-calculated KS stats).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output result files.")
    parser.add_argument('--start_hpo_id', type=str, default='HP:0000707', help="The root HPO term ID (default: 'HP:0000707' - Neurodevelopmental abnormality).")
    parser.add_argument('--min_hpos', type=int, default=10, help="If a term has >= this many relevant descendants, its children are added for analysis (default: 10).")
    parser.add_argument('--column_prefix', type=str, default='global_celltype|', help="Prefix for cell type columns to analyze (default: 'global_celltype|').")
    parser.add_argument('--min_obs_ks', type=int, default=5, help="Minimum observations in both group/other sets for secondary KS test (default: 5).")

    parser.add_argument('--min_pvalue', type=float, default=None,
                        help="Activate combined p-value filtering. Provide a threshold (0.0 to 1.0). "
                             "Rows are kept ONLY if flag is provided AND they have at least one p-value <= threshold. "
                             "Columns are kept ONLY if flag is provided AND they have at least one p-value <= threshold. "
                             "If not provided, no p-value filtering occurs.")

    args = parser.parse_args()

    run_analysis(args)
