#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import glob

def create_hpo_cellgroup_matrix(directory_path, output_file="hpo_cellgroup_matrix.tsv", min_cells=0, debug=False):
    """
    Creates a matrix of HPO vs cell groups with KS statistics for significant associations.
    Handles files with column headers included as data.

    Parameters:
    directory_path (str): Path to the directory containing TSV files for each HPO
    output_file (str): Path to save the output matrix TSV file
    min_cells (int): Minimum number of cells required in a cell group to include it
    debug (bool): Whether to print debug information

    Returns:
    pandas.DataFrame: The resulting matrix DataFrame
    """
    # Get absolute path for output file
    output_file_abs = os.path.abspath(output_file)

    # Get all TSV files in the directory
    tsv_files = glob.glob(os.path.join(directory_path, "*.tsv"))

    if not tsv_files:
        print(f"Error: No TSV files found in {directory_path}")
        return None

    print(f"Found {len(tsv_files)} TSV files in {directory_path}")
    print(f"Output will be written to: {output_file_abs}")

    # Initialize empty dataframes to collect data
    all_data = []

    # Define column names based on the file structure described
    columns = ["hpo_id", "hpo_name", "n_related_genes", "grouping_level", "group_name",
               "tissue", "celltype", "supercluster", "cluster", "subcluster",
               "n_cells_in_group", "n_background_cells", "ks_statistic", "ks_pvalue",
               "effect_size", "significant", "dissection", "dissection_celltype",
               "dissection_supercluster"]

    # Track files with issues
    empty_files = []
    no_significant_files = []
    error_files = []
    processed_files = 0
    filtered_rows = 0
    filtered_rows_without_celltype = 0

    # Process each TSV file
    for file_path in tsv_files:
        try:
            # Check if file is empty
            if os.path.getsize(file_path) == 0:
                empty_files.append(file_path)
                continue

            # Read the TSV file without specifying column names first
            raw_df = pd.read_csv(file_path, sep='\t', header=None)

            # Check if first row contains column headers
            first_row = raw_df.iloc[0, 0]
            has_header_row = isinstance(first_row, str) and first_row.startswith('#')

            if has_header_row:
                # Skip the first row (header row)
                df = raw_df.iloc[1:].copy()
                df.columns = columns
            else:
                # Use the columns directly
                df = raw_df.copy()
                df.columns = columns

            # Print sample of the data if in debug mode
            if debug and processed_files < 3:
                print(f"\nSample from file {os.path.basename(file_path)}:")
                print(df.head(2))

            # Convert significant column to numeric if needed
            df['significant'] = pd.to_numeric(df['significant'], errors='coerce')

            # Convert n_cells_in_group to numeric
            df['n_cells_in_group'] = pd.to_numeric(df['n_cells_in_group'], errors='coerce')

            # Filter rows with significant=1
            df_sig = df[df['significant'] == 1].copy()

            if df_sig.empty:
                no_significant_files.append(file_path)
                continue

            # Filter out rows with cell groups below the minimum threshold
            before_filter = len(df_sig)
            df_sig = df_sig[df_sig['n_cells_in_group'] >= min_cells]
            filtered_in_this_file = before_filter - len(df_sig)
            filtered_rows += filtered_in_this_file

            if debug and filtered_in_this_file > 0:
                print(f"Filtered out {filtered_in_this_file} rows with fewer than {min_cells} cells in {os.path.basename(file_path)}")

            # Filter the rows without cell groups
            before_without_celltype= len(df_sig)
            df_sig = df_sig[~df_sig['celltype'].isnull()]
            filtered_in_this_file_celltype = before_without_celltype - len(df_sig)
            filtered_rows_without_celltype += filtered_in_this_file_celltype
            
            if debug and filtered_in_this_file_celltype > 0:
                print(f"Filtered out {filtered_in_this_file_celltype} rows without celltype {os.path.basename(file_path)}")
                
            if df_sig.empty:
                no_significant_files.append(file_path)
                continue

            # Convert numeric columns to float
            for col in ['ks_statistic', 'ks_pvalue', 'effect_size']:
                df_sig[col] = pd.to_numeric(df_sig[col], errors='coerce')

            all_data.append(df_sig)
            processed_files += 1

            if debug and processed_files <= 3:
                print(f"Extracted {len(df_sig)} significant associations from {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            error_files.append(file_path)

    # Combine all data
    if not all_data:
        print("\nDiagnostic information:")
        print(f"Total files found: {len(tsv_files)}")
        print(f"Empty files: {len(empty_files)}")
        print(f"Files with no significant associations: {len(no_significant_files)}")
        print(f"Files with errors: {len(error_files)}")

        raise ValueError("No significant associations found in any file. Check the diagnostic information above.")

    combined_df = pd.concat(all_data, ignore_index=True)

    print(f"\nSuccessfully processed {processed_files} files with significant associations")
    print(f"Found {len(combined_df)} total significant associations after filtering")
    print(f"Filtered out {filtered_rows} associations with fewer than {min_cells} cells")
    print(f"Filtered out {filtered_rows_without_celltype} associations without celltype")

    # Handle missing dissection columns
    for col in ['dissection', 'dissection_celltype', 'dissection_supercluster']:
        if col not in combined_df.columns:
            combined_df[col] = 'NA'
    # Create a unique cell group identifier, handling missing values gracefully for all parts
    # Using f-string concatenation and inline checks for missing values (NaN, None)
    combined_df['cell_group'] = combined_df.apply(
        lambda row: (
            # For each component, check if it's NA. If so, use '', otherwise convert the value to string.
            # row.get(col, '') safely handles potentially missing columns or None values before pd.isna/str.
            f"{'' if pd.isna(row.get('grouping_level')) else str(row.get('grouping_level', ''))}|"
            f"{'' if pd.isna(row.get('group_name')) else str(row.get('group_name', ''))}|"
            f"{'' if pd.isna(row.get('tissue')) else str(row.get('tissue', ''))}|"
            f"{'' if pd.isna(row.get('celltype')) else str(row.get('celltype', ''))}|"
            f"{'' if pd.isna(row.get('supercluster')) else str(row.get('supercluster', ''))}|"
            f"{'' if pd.isna(row.get('cluster')) else str(row.get('cluster', ''))}|"
            f"{'' if pd.isna(row.get('subcluster')) else str(row.get('subcluster', ''))}|"
            f"{'' if pd.isna(row.get('dissection')) else str(row.get('dissection', ''))}|"
            f"{'' if pd.isna(row.get('dissection_celltype')) else str(row.get('dissection_celltype', ''))}|"
            f"{'' if pd.isna(row.get('dissection_supercluster')) else str(row.get('dissection_supercluster', ''))}"
        ),
        axis=1
    )
    # --- Modified Section End ---
    # Create the matrix: rows are HPOs, columns are cell groups, values are KS statistics
    matrix_df = combined_df.pivot_table(
        index=['hpo_id', 'hpo_name'],
        columns='cell_group',
        values='ks_statistic',
        aggfunc='max',  # Take the maximum KS statistic if there are duplicates
        fill_value=0    # Fill NaN values with 0 (indicating no significant association)
    )

    # Reset the index to make hpo_id and hpo_name regular columns
    matrix_df = matrix_df.reset_index()

    # Save the matrix to a TSV file
    matrix_df.to_csv(output_file, sep='\t', index=False)

    print(f"\nOutput matrix successfully saved to: {output_file_abs}")
    print(f"Matrix dimensions: {matrix_df.shape[0]} HPOs × {matrix_df.shape[1]-2} cell groups")

    # Print some statistics about the matrix
    significant_associations = (matrix_df.iloc[:, 2:] > 0).sum().sum()
    print(f"Total number of significant associations in matrix: {significant_associations}")

    return matrix_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Create HPO-cell group matrix from TSV files')
    parser.add_argument('directory', help='Directory containing HPO TSV files')
    parser.add_argument('--output', '-o', default='hpo_cellgroup_matrix.tsv',
                        help='Output file path (default: hpo_cellgroup_matrix.tsv)')
    parser.add_argument('--min-cells', '-m', type=int, default=0,
                        help='Minimum number of cells in a group to include (default: 0)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Print debug information')

    args = parser.parse_args()

    try:
        print(f"\n=== HPO Cell Group Matrix Generator ===")
        print(f"Input directory: {os.path.abspath(args.directory)}")
        print(f"Output file: {os.path.abspath(args.output)}")
        print(f"Minimum cell threshold: {args.min_cells}")
        print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
        print(f"=======================================\n")

        # Create the matrix with cell group filtering
        matrix = create_hpo_cellgroup_matrix(
            args.directory,
            args.output,
            min_cells=args.min_cells,
            debug=args.debug
        )

        print(f"\n=== Processing Complete ===")
        print(f"Matrix saved to: {os.path.abspath(args.output)}")
        print(f"===========================\n")

    except Exception as e:
        print(f"\nError: {e}")
        print("\nTry running with the --debug flag for more information:")
        print(f"python {os.path.basename(__file__)} {args.directory} -o {args.output} -m {args.min_cells} --debug")
