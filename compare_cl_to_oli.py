#!/usr/bin/env python3

import pandas as pd
from typing import List, Dict, Any
import os
import argparse # For command-line arguments
import logging # For logging

from thefuzz import fuzz # For fuzzy string matching

# --- Configuration ---
# OLI file columns (Target for matching)
OLI_HPO_ID = "#1-hpo_id"
OLI_HPO_NAME = "#2-hpo_name"
OLI_N_RELATED_GENES = "#3-n_related_genes"
OLI_GROUPING_LEVEL = "#4-grouping_level"
OLI_GROUP_NAME = "#5-group_name"
OLI_TISSUE = "#6-tissue"
OLI_CELLTYPE = "#7-celltype"
OLI_SUPERCLUSTER = "#8-supercluster"
OLI_CLUSTER = "#9-cluster"
OLI_SUBCLUSTER = "#10-subcluster"
OLI_N_CELLS_IN_GROUP = "#11-n_cells_in_group"
OLI_N_BACKGROUND_CELLS = "#12-n_background_cells"
OLI_KS_STATISTIC = "#13-ks_statistic"
OLI_KS_PVALUE = "#14-ks_pvalue"
OLI_EFFECT_SIZE = "#15-effect_size"
OLI_SIGNIFICANT = "#16-significant"
OLI_DISSECTION = "#17-dissection"
OLI_DISSECTION_CELLTYPE = "#18-dissection_celltype"
OLI_DISSECTION_SUPERCLUSTER = "#19-dissection_supercluster"

OLI_ALL_COLUMNS = [
    OLI_HPO_ID, OLI_HPO_NAME, OLI_N_RELATED_GENES, OLI_GROUPING_LEVEL,
    OLI_GROUP_NAME, OLI_TISSUE, OLI_CELLTYPE, OLI_SUPERCLUSTER, OLI_CLUSTER,
    OLI_SUBCLUSTER, OLI_N_CELLS_IN_GROUP, OLI_N_BACKGROUND_CELLS,
    OLI_KS_STATISTIC, OLI_KS_PVALUE, OLI_EFFECT_SIZE, OLI_SIGNIFICANT,
    OLI_DISSECTION, OLI_DISSECTION_CELLTYPE, OLI_DISSECTION_SUPERCLUSTER
]
# Fields in OLI file to search for matches
OLI_FIELDS_TO_SEARCH_IN = [OLI_TISSUE, OLI_CELLTYPE, OLI_SUPERCLUSTER, OLI_DISSECTION]

# CL/UBERON file columns (Source of terms to match)
CL_HPO_ID = "HPO_ID" # Used for filtering by OLI phenotype
CL_HPO_NAME = "HPO_Name"
CL_ASSOC_TERM_ID = "Associated_Term_ID"
CL_ASSOC_TERM_NAME = "Associated_Term_Name"
CL_TERM_TYPE = "Term_Type"
CL_P_VALUE = "P_Value"

CL_UBERON_ALL_COLUMNS = [
    CL_HPO_ID, CL_HPO_NAME, CL_ASSOC_TERM_ID, CL_ASSOC_TERM_NAME, CL_TERM_TYPE, CL_P_VALUE
]

# --- Helper Functions ---

def load_oli_file(file_path: str) -> pd.DataFrame:
    logging.info(f"Attempting to load OLI file: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t', comment=None, low_memory=False)
        logging.debug(f"Successfully read OLI file into DataFrame. Shape: {df.shape}")

        if df.empty:
            logging.warning("Loaded OLI file is empty.")
            return df

        if OLI_HPO_ID not in df.columns or OLI_HPO_NAME not in df.columns:
            logging.error(f"OLI file must contain '{OLI_HPO_ID}' and '{OLI_HPO_NAME}' columns for phenotype identification.")
            return pd.DataFrame()

        df[OLI_N_CELLS_IN_GROUP] = pd.to_numeric(df[OLI_N_CELLS_IN_GROUP], errors='coerce')
        df[OLI_KS_STATISTIC] = pd.to_numeric(df[OLI_KS_STATISTIC], errors='coerce')

        nan_cells = df[OLI_N_CELLS_IN_GROUP].isna().sum()
        if nan_cells > 0:
            logging.warning(f"{nan_cells} OLI rows have non-numeric values in '{OLI_N_CELLS_IN_GROUP}' and will be excluded from cell count filtering if NaN.")
        nan_ks = df[OLI_KS_STATISTIC].isna().sum()
        if nan_ks > 0:
            logging.warning(f"{nan_ks} OLI rows have non-numeric values in '{OLI_KS_STATISTIC}'. These rows might be excluded if KS is needed for sorting and is NaN.")

        df.dropna(subset=[OLI_N_CELLS_IN_GROUP], inplace=True)
        logging.info(f"OLI DataFrame shape after ensuring '{OLI_N_CELLS_IN_GROUP}' is numeric: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"OLI file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading OLI file {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

def process_oli_data(df: pd.DataFrame, min_cells: int = 5) -> pd.DataFrame:
    if df.empty:
        logging.warning("OLI DataFrame is empty for processing.")
        return df
    logging.info(f"Processing OLI data. Initial rows: {len(df)}")
    logging.info(f"Filtering OLI data criteria: min_cells_in_group >= {min_cells}")
    df[OLI_N_CELLS_IN_GROUP] = pd.to_numeric(df[OLI_N_CELLS_IN_GROUP], errors='coerce')
    df_valid_cells = df.dropna(subset=[OLI_N_CELLS_IN_GROUP])
    df_filtered_cells = df_valid_cells[df_valid_cells[OLI_N_CELLS_IN_GROUP] >= min_cells]
    logging.info(f"OLI rows after filtering by min_cells ({min_cells}): {len(df_filtered_cells)}")
    return df_filtered_cells.copy()

def load_cl_uberon_file(file_path: str) -> pd.DataFrame:
    logging.info(f"Attempting to load CL/UBERON file: {file_path}")
    try:
        df = pd.read_csv(file_path, sep='\t', low_memory=False)
        if CL_HPO_ID not in df.columns:
            logging.error(f"CL/UBERON file must contain '{CL_HPO_ID}' column for phenotype filtering.")
            return pd.DataFrame()
        logging.info(f"Successfully loaded CL/UBERON file. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"CL/UBERON file not found at {file_path}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Error loading CL/UBERON file {file_path}: {e}", exc_info=True)
        return pd.DataFrame()

def find_oli_matches_for_cl_term(cl_term_to_match: str, cell_filtered_oli_df: pd.DataFrame,
                                  similarity_threshold: int = 80) -> List[Dict[str, Any]]:
    found_oli_matches = []
    if cell_filtered_oli_df.empty or not cl_term_to_match or not isinstance(cl_term_to_match, str):
        return found_oli_matches
    cl_term_lower = cl_term_to_match.lower()
    logging.debug(f"Searching for OLI matches for CL/UBERON term: '{cl_term_to_match}'")
    for oli_idx, oli_row_series in cell_filtered_oli_df.iterrows():
        oli_ks_value = oli_row_series.get(OLI_KS_STATISTIC)
        if pd.isna(oli_ks_value):
            logging.debug(f"  OLI Row Index {oli_idx} has NaN for KS statistic. It will be included in matches but may rank low.")
        for oli_field_to_search in OLI_FIELDS_TO_SEARCH_IN:
            oli_value = oli_row_series.get(oli_field_to_search)
            if pd.notna(oli_value) and isinstance(oli_value, str) and oli_value.strip():
                oli_value_lower = oli_value.lower()
                score = fuzz.token_set_ratio(cl_term_lower, oli_value_lower)
                if score >= similarity_threshold:
                    match_details = {
                        'oli_row_index': oli_idx,
                        'oli_row_series': oli_row_series,
                        'matched_oli_field_name': oli_field_to_search,
                        'matched_oli_value': oli_value,
                        'similarity_score': score,
                        'oli_ks_statistic': oli_ks_value
                    }
                    found_oli_matches.append(match_details)
    return found_oli_matches

def generate_reports(phenotype_filtered_cl_uberon_df: pd.DataFrame,
                      cell_filtered_oli_df: pd.DataFrame,
                      output_detailed_report_path: str, output_summary_tsv_path: str,
                      output_relevant_cl_terms_report_path: str,
                      oli_phenotype_hpo_id: str, oli_phenotype_hpo_name: str,
                      similarity_threshold: int = 80, top_n_ks: int = 5,
                      min_cells_filter: int = 5):
    logging.info(f"Generating detailed text report at: {output_detailed_report_path}")
    logging.info(f"Generating summary TSV report at: {output_summary_tsv_path}")
    logging.info(f"Generating relevant CL/UBERON terms report at: {output_relevant_cl_terms_report_path}")

    globally_reported_links = set()
    summary_data_for_tsv = []

    relevant_cl_uberon_term_links_summary = {}

    report_phenotype_context = f"Analysis for OLI Phenotype: {oli_phenotype_hpo_name} ({oli_phenotype_hpo_id})"

    if phenotype_filtered_cl_uberon_df.empty:
        logging.warning(f"Phenotype-filtered CL/UBERON DataFrame is empty (for OLI phenotype {oli_phenotype_hpo_id}). No CL/UBERON terms to process for matching.")
        with open(output_detailed_report_path, 'w') as f:
            f.write(f"Analysis Report: CL/UBERON Terms Matched to Top N OLI Data Entries (by KS Statistic)\n")
            f.write(f"==================================================================================\n\n")
            f.write(f"{report_phenotype_context}\n")
            f.write("Phenotype-filtered CL/UBERON data was empty. No CL/UBERON terms processed for this phenotype.\n")
        pd.DataFrame([]).to_csv(output_summary_tsv_path, sep='\t', index=False)
        with open(output_relevant_cl_terms_report_path, 'w') as f:
            f.write(f"Summary of Relevant CL/UBERON Terms and Their Links to Top N OLI Data\n")
            f.write(f"====================================================================\n\n")
            f.write(f"{report_phenotype_context}\n")
            f.write("Phenotype-filtered CL/UBERON data was empty. No relevant terms identified for this phenotype.\n")
        return

    if cell_filtered_oli_df.empty:
        logging.warning("Cell-filtered OLI DataFrame is empty. No matches can be found.")
        with open(output_detailed_report_path, 'w') as f:
            f.write(f"Analysis Report: CL/UBERON Terms Matched to Top N OLI Data Entries (by KS Statistic)\n")
            f.write(f"==================================================================================\n\n")
            f.write(f"{report_phenotype_context}\n")
            f.write("Cell-filtered OLI data was empty. No analysis performed.\n")
        pd.DataFrame([]).to_csv(output_summary_tsv_path, sep='\t', index=False)
        with open(output_relevant_cl_terms_report_path, 'w') as f:
            f.write(f"Summary of Relevant CL/UBERON Terms and Their Links to Top N OLI Data\n")
            f.write(f"====================================================================\n\n")
            f.write(f"{report_phenotype_context}\n")
            f.write("Cell-filtered OLI data was empty. No relevant CL/UBERON terms identified.\n")
        return

    any_cl_entry_had_reportable_links = False
    try:
        with open(output_detailed_report_path, 'w') as detailed_report_file:
            detailed_report_file.write("Analysis Report: CL/UBERON Terms Matched to Top N OLI Data Entries (by KS Statistic)\n")
            detailed_report_file.write("==================================================================================\n\n")
            detailed_report_file.write(f"{report_phenotype_context}\n")
            detailed_report_file.write(f"OLI data initial filtering: min_cells_in_group >= {min_cells_filter}\n")
            detailed_report_file.write(f"For each CL/UBERON term (related to OLI phenotype), top {top_n_ks} OLI matches by KS statistic are considered (including ties for the {top_n_ks}th KS value).\n")
            detailed_report_file.write(f"Global de-duplication of reported (CL ID, OLI Index, OLI Field, OLI Term) links applied.\n")
            detailed_report_file.write(f"Similarity threshold for matching: {similarity_threshold}%\n\n")

            for _, cl_row in phenotype_filtered_cl_uberon_df.iterrows():
                cl_term_name = cl_row.get(CL_ASSOC_TERM_NAME)
                cl_term_id = cl_row.get(CL_ASSOC_TERM_ID)
                cl_row_hpo_name = cl_row.get(CL_HPO_NAME, "N/A")
                cl_row_hpo_id = cl_row.get(CL_HPO_ID, "N/A")

                if not cl_term_name or not isinstance(cl_term_name, str) or not cl_term_name.strip() or not cl_term_id:
                    continue

                all_potential_oli_matches = find_oli_matches_for_cl_term(
                    cl_term_name, cell_filtered_oli_df, similarity_threshold
                )

                if not all_potential_oli_matches:
                    detailed_report_file.write(f"--------------------------------------------------\n")
                    detailed_report_file.write(f"CL/UBERON Term: '{cl_term_name}' (ID: {cl_term_id})\n")
                    detailed_report_file.write(f"  (Associated with Phenotype: '{cl_row_hpo_name}' ({cl_row_hpo_id}))\n")
                    detailed_report_file.write("  No OLI matches found based on current criteria.\n")
                    detailed_report_file.write(f"--------------------------------------------------\n\n")
                    logging.info(f"CL/UBERON term '{cl_term_name}' (ID: {cl_term_id}) had no OLI matches.")
                    continue
                
                all_potential_oli_matches.sort(
                    key=lambda x: (x.get('oli_ks_statistic', -float('inf')), x.get('similarity_score', 0)),
                    reverse=True
                )
                unique_oli_rows_with_ks = {}
                for match in all_potential_oli_matches:
                    idx = match['oli_row_index']
                    ks = match.get('oli_ks_statistic', -float('inf'))
                    if idx not in unique_oli_rows_with_ks:
                        unique_oli_rows_with_ks[idx] = ks
                sorted_unique_oli_rows_tuples = sorted(unique_oli_rows_with_ks.items(), key=lambda item: item[1], reverse=True)
                ordered_top_n_oli_indices = []
                if sorted_unique_oli_rows_tuples:
                    ks_score_of_nth = -float('inf')
                    if len(sorted_unique_oli_rows_tuples) >= top_n_ks:
                        ks_score_of_nth = sorted_unique_oli_rows_tuples[top_n_ks - 1][1]
                    for oli_idx, ks_val in sorted_unique_oli_rows_tuples:
                        if len(ordered_top_n_oli_indices) < top_n_ks:
                            ordered_top_n_oli_indices.append(oli_idx)
                        elif pd.notna(ks_val) and pd.notna(ks_score_of_nth) and ks_val == ks_score_of_nth:
                            ordered_top_n_oli_indices.append(oli_idx)
                        elif pd.notna(ks_val) and pd.notna(ks_score_of_nth) and ks_val < ks_score_of_nth:
                            break
                        elif pd.isna(ks_val) and pd.notna(ks_score_of_nth):
                            break
                        elif pd.isna(ks_score_of_nth) and len(ordered_top_n_oli_indices) >= top_n_ks:
                            break
                matches_from_top_n_oli_rows = []
                if ordered_top_n_oli_indices:
                    for match_dict in all_potential_oli_matches:
                        if match_dict['oli_row_index'] in ordered_top_n_oli_indices:
                            matches_from_top_n_oli_rows.append(match_dict)

                actual_oli_matches_to_report_for_this_cl_term = []
                cl_term_contributed_new_link_this_iteration = False

                if matches_from_top_n_oli_rows:
                    for oli_match in matches_from_top_n_oli_rows:
                        global_link_key = (cl_term_id,
                                           oli_match['oli_row_index'],
                                           oli_match['matched_oli_field_name'],
                                           oli_match['matched_oli_value'].lower())

                        if global_link_key not in globally_reported_links:
                            actual_oli_matches_to_report_for_this_cl_term.append(oli_match)
                            globally_reported_links.add(global_link_key)
                            cl_term_contributed_new_link_this_iteration = True

                            summary_row_data = cl_row.to_dict()
                            summary_row_data.update(oli_match['oli_row_series'].to_dict())
                            summary_row_data['CL_Term_Matched_To_OLI_Field'] = oli_match['matched_oli_field_name']
                            summary_row_data['OLI_Term_Value_Matched'] = oli_match['matched_oli_value']
                            summary_row_data['Match_Similarity_Score'] = oli_match['similarity_score']
                            summary_data_for_tsv.append(summary_row_data)

                            cl_summary_key = (cl_term_id, cl_term_name)
                            if cl_summary_key not in relevant_cl_uberon_term_links_summary:
                                relevant_cl_uberon_term_links_summary[cl_summary_key] = {
                                    'occurrences_finding_novel_oli_links': 0,
                                    'linked_oli_details': set()  # Stores (oli_row_index, ks_value)
                                }
                            ks_of_this_oli_entry = oli_match.get('oli_ks_statistic')
                            relevant_cl_uberon_term_links_summary[cl_summary_key]['linked_oli_details'].add(
                                (oli_match['oli_row_index'], ks_of_this_oli_entry)
                            )

                if cl_term_contributed_new_link_this_iteration:
                    cl_summary_key = (cl_term_id, cl_term_name)
                    if cl_summary_key in relevant_cl_uberon_term_links_summary:
                         relevant_cl_uberon_term_links_summary[cl_summary_key]['occurrences_finding_novel_oli_links'] += 1

                if actual_oli_matches_to_report_for_this_cl_term:
                    any_cl_entry_had_reportable_links = True
                    detailed_report_file.write("--------------------------------------------------\n")
                    detailed_report_file.write(f"CL/UBERON Term: '{cl_term_name}' (ID: {cl_term_id})\n")
                    detailed_report_file.write(f"  (Associated with Phenotype: '{cl_row_hpo_name}' ({cl_row_hpo_id}))\n")
                    matches_grouped_by_oli_row = {}
                    for m in actual_oli_matches_to_report_for_this_cl_term:
                        idx = m['oli_row_index']
                        if idx not in matches_grouped_by_oli_row:
                            matches_grouped_by_oli_row[idx] = {
                                'oli_row_series': m['oli_row_series'],
                                'ks': m.get('oli_ks_statistic', -float('inf')),
                                'field_matches': []
                            }
                        matches_grouped_by_oli_row[idx]['field_matches'].append(m)
                    sorted_oli_entries_to_report = sorted(matches_grouped_by_oli_row.items(),key=lambda item: item[1]['ks'],reverse=True)
                    detailed_report_file.write(f"  Displaying OLI match(es) from {len(sorted_oli_entries_to_report)} unique Top OLI row(s) (by KS, up to {top_n_ks} plus ties):\n\n")
                    for oli_idx, data in sorted_oli_entries_to_report:
                        oli_row_series = data['oli_row_series']
                        ks_val = data['ks']
                        field_matches_for_this_oli_row = data['field_matches']
                        ks_val_str = f"{ks_val:.4f}" if pd.notna(ks_val) else "N/A"
                        detailed_report_file.write(f"  --- OLI Entry (Original Index: {oli_idx}, KS: {ks_val_str}) ---\n")
                        for oli_col_name in OLI_ALL_COLUMNS:
                            detailed_report_file.write(f"    {oli_col_name}: {oli_row_series.get(oli_col_name, 'N/A')}\n")
                        detailed_report_file.write(f"    Field Matches to CL/UBERON term '{cl_term_name}':\n")
                        for oli_field_match in field_matches_for_this_oli_row:
                            detailed_report_file.write(f"      - OLI Field: {oli_field_match['matched_oli_field_name']}\n")
                            detailed_report_file.write(f"        OLI Value: '{oli_field_match['matched_oli_value']}'\n")
                            detailed_report_file.write(f"        Similarity: {oli_field_match['similarity_score']}%\n")
                        detailed_report_file.write("\n")
                    detailed_report_file.write("--------------------------------------------------\n\n")

                elif not all_potential_oli_matches:
                    pass # Already handled
                else: # Potential matches, but no new globally unique links from top N
                    detailed_report_file.write(f"--------------------------------------------------\n")
                    detailed_report_file.write(f"CL/UBERON Term: '{cl_term_name}' (ID: {cl_term_id})\n")
                    detailed_report_file.write(f"  (Associated with Phenotype: '{cl_row_hpo_name}' ({cl_row_hpo_id}))\n")
                    detailed_report_file.write(f"  Potential OLI matches were found, but none were new globally unique links from the Top OLI rows (up to {top_n_ks} plus ties).\n")
                    detailed_report_file.write(f"--------------------------------------------------\n\n")
                    logging.info(f"CL/UBERON term '{cl_term_name}' (ID: {cl_term_id}) had potential OLI matches, but no new globally unique ones from Top {top_n_ks} (plus ties) to report.")


            if not any_cl_entry_had_reportable_links:
                detailed_report_file.write("No CL/UBERON terms (for the specified OLI phenotype) had any new, globally unique OLI links (from Top N by KS, plus ties) to report.\n")
        logging.info(f"Detailed text report generation complete.")
    except IOError as e:
        logging.error(f"Could not write detailed text report: {e}", exc_info=True)

    # Write summary TSV (remains the same as it already includes all OLI columns including KS)
    try:
        if summary_data_for_tsv:
            summary_df = pd.DataFrame(summary_data_for_tsv)
            summary_cols_order = CL_UBERON_ALL_COLUMNS + OLI_ALL_COLUMNS + [
                'CL_Term_Matched_To_OLI_Field', 'OLI_Term_Value_Matched', 'Match_Similarity_Score'
            ]
            for col in summary_cols_order:
                if col not in summary_df.columns: summary_df[col] = pd.NA
            summary_df = summary_df[summary_cols_order]
            summary_df.to_csv(output_summary_tsv_path, sep='\t', index=False)
            logging.info(f"Summary TSV report generated: {output_summary_tsv_path}")
        else:
            empty_cols = CL_UBERON_ALL_COLUMNS + OLI_ALL_COLUMNS + [
                'CL_Term_Matched_To_OLI_Field', 'OLI_Term_Value_Matched', 'Match_Similarity_Score'
            ]
            pd.DataFrame(columns=empty_cols).to_csv(output_summary_tsv_path, sep='\t', index=False)
            logging.info(f"Empty summary TSV report generated: {output_summary_tsv_path}")
    except IOError as e:
        logging.error(f"Could not write summary TSV: {e}", exc_info=True)

    try:
        with open(output_relevant_cl_terms_report_path, 'w') as rel_file:
            rel_file.write(f"Summary of Relevant CL/UBERON Terms (for OLI Phenotype: {oli_phenotype_hpo_name} ({oli_phenotype_hpo_id})) and Their Links to Top N OLI Data\n")
            rel_file.write("================================================================================================================================\n\n")
            if relevant_cl_uberon_term_links_summary:
                sorted_terms_data = []
                for (cl_id, cl_name), data in relevant_cl_uberon_term_links_summary.items():
                    linked_details_set = data.get('linked_oli_details', set())
                    
                    distinct_oli_indices_linked = {idx for idx, ks in linked_details_set}
                    distinct_oli_entry_count = len(distinct_oli_indices_linked)

                    ks_values_for_term = [ks for idx, ks in linked_details_set if pd.notna(ks)]
                    ks_summary_str = "N/A"
                    if ks_values_for_term:
                        unique_sorted_ks = sorted(list(set(ks_values_for_term)), reverse=True)
                        if len(unique_sorted_ks) > 3:
                            ks_summary_str = ", ".join([f"{ks:.4f}" for ks in unique_sorted_ks[:3]]) + ", etc."
                        else:
                            ks_summary_str = ", ".join([f"{ks:.4f}" for ks in unique_sorted_ks])
                    
                    sorted_terms_data.append({
                        'cl_uberon_term_id': cl_id,
                        'cl_uberon_term_name': cl_name,
                        'occurrences_finding_novel_oli_links': data['occurrences_finding_novel_oli_links'],
                        'distinct_oli_entries_linked_to': distinct_oli_entry_count,
                        'linked_oli_ks_summary': ks_summary_str
                    })
                
                # Sort by original criteria: occurrences, then distinct OLI entries
                sorted_terms_data.sort(key=lambda x: (x['occurrences_finding_novel_oli_links'], x['distinct_oli_entries_linked_to']), reverse=True)

                # Define column widths
                col_width_name = 45
                col_width_id = 15
                col_width_novel_links = 20 # "Novel Links Count"
                col_width_distinct_oli = 25 # "Unique OLI Rows Linked"
                col_width_ks = 35        # "KS of Linked OLI (Top)"

                header_format = f"{{:<{col_width_name}}} {{:<{col_width_id}}} {{:<{col_width_novel_links}}} {{:<{col_width_distinct_oli}}} {{:<{col_width_ks}}}\n"
                
                rel_file.write(header_format.format(
                    "CL/UBERON Term Name", "ID", "Novel Links Count", "Unique OLI Rows Linked", "KS of Linked OLI (Top)"
                ))
                rel_file.write(header_format.format(
                    '-'*col_width_name, '-'*col_width_id, '-'*col_width_novel_links, '-'*col_width_distinct_oli, '-'*col_width_ks
                ))
                for item in sorted_terms_data:
                    rel_file.write(header_format.format(
                        item['cl_uberon_term_name'],
                        item['cl_uberon_term_id'],
                        item['occurrences_finding_novel_oli_links'],
                        item['distinct_oli_entries_linked_to'],
                        item['linked_oli_ks_summary']
                    ))
            else:
                rel_file.write("No relevant CL/UBERON terms found with links to OLI data for this phenotype.\n")
            logging.info(f"Relevant CL/UBERON terms report generated: {output_relevant_cl_terms_report_path}")
    except IOError as e:
        logging.error(f"Could not write relevant CL/UBERON terms report: {e}", exc_info=True)


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Iterate CL/UBERON terms (filtered by OLI file's phenotype), find matches in OLI data, keep Top N OLI by KS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--cl_uberon_file", type=str, required=True, help="Path to CL/UBERON TSV file.")
    parser.add_argument("--oli_file", type=str, required=True, help="Path to OLI TSV file (determines the phenotype of interest).")
    parser.add_argument("--output_detailed_report", type=str, default="cl_to_oli_topN_pheno_report.txt")
    parser.add_argument("--output_summary_tsv", type=str, default="cl_to_oli_topN_pheno_summary.tsv")
    parser.add_argument("--output_relevant_cl_terms_report", type=str, default="cl_to_oli_topN_pheno_relevant_terms.txt")
    parser.add_argument("--similarity_threshold", type=int, default=80, choices=range(101))
    parser.add_argument("--min_cells", type=int, default=5, help="Min cells in OLI group for initial OLI filtering.")
    parser.add_argument("--top_n_ks", type=int, default=5, help="Keep Top N OLI matches by KS statistic for each CL/UBERON term (includes ties for Nth KS).")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--log_file", type=str, default=None)
    args = parser.parse_args()

    # Setup logging
    log_format = '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int): raise ValueError(f'Invalid log level: {args.log_level}')
    log_handlers = [logging.StreamHandler()]
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(log_format))
        log_handlers.append(file_handler)
    logging.basicConfig(level=numeric_level, format=log_format, handlers=log_handlers)
    if args.log_file: logging.info(f"Logging to console and to file: {args.log_file}")
    else: logging.info("Logging to console")

    logging.info("Script started: CL/UBERON to OLI Analyzer (Phenotype-Specific, Top N by KS).")
    logging.info(f"Arguments: {args}")

    if not os.path.exists(args.oli_file):
        logging.critical(f"OLI file not found: {args.oli_file}. Exiting.")
        return
    raw_oli_df = load_oli_file(args.oli_file)
    if raw_oli_df.empty:
        logging.critical("OLI data is empty after loading or missing critical HPO columns. Cannot determine phenotype. Exiting.")
        generate_reports(pd.DataFrame(), pd.DataFrame(), args.output_detailed_report,
                         args.output_summary_tsv, args.output_relevant_cl_terms_report,
                         "N/A", "N/A",
                         args.similarity_threshold, args.top_n_ks, args.min_cells)
        return

    try:
        oli_phenotype_hpo_id = raw_oli_df[OLI_HPO_ID].iloc[0]
        oli_phenotype_hpo_name = raw_oli_df[OLI_HPO_NAME].iloc[0]
        if pd.isna(oli_phenotype_hpo_id) or pd.isna(oli_phenotype_hpo_name):
            logging.critical(f"Could not determine HPO ID or Name from the first row of OLI file: {args.oli_file} (values are NaN). Exiting.")
            generate_reports(pd.DataFrame(), raw_oli_df, args.output_detailed_report,
                             args.output_summary_tsv, args.output_relevant_cl_terms_report,
                             str(oli_phenotype_hpo_id), str(oli_phenotype_hpo_name), # Pass NaN as str
                             args.similarity_threshold, args.top_n_ks, args.min_cells)
            return
    except IndexError:
        logging.critical(f"OLI file {args.oli_file} is empty after initial processing, cannot determine phenotype. Exiting.")
        generate_reports(pd.DataFrame(), pd.DataFrame(), args.output_detailed_report,
                         args.output_summary_tsv, args.output_relevant_cl_terms_report,
                         "N/A", "N/A",
                         args.similarity_threshold, args.top_n_ks, args.min_cells)
        return
    except KeyError as e:
        logging.critical(f"OLI file {args.oli_file} is missing critical column {e} for phenotype determination. Exiting.")
        generate_reports(pd.DataFrame(), pd.DataFrame(), args.output_detailed_report,
                         args.output_summary_tsv, args.output_relevant_cl_terms_report,
                         "N/A", "N/A",
                         args.similarity_threshold, args.top_n_ks, args.min_cells)
        return
        
    logging.info(f"Determined OLI Phenotype for analysis: {oli_phenotype_hpo_name} ({oli_phenotype_hpo_id})")

    cell_filtered_oli_df = process_oli_data(raw_oli_df, min_cells=args.min_cells)
    if cell_filtered_oli_df.empty:
        logging.warning("OLI data is empty after min_cells filtering. No matches possible.")

    if not os.path.exists(args.cl_uberon_file):
        logging.critical(f"CL/UBERON file not found: {args.cl_uberon_file}. Exiting.")
        return
    cl_uberon_df_full = load_cl_uberon_file(args.cl_uberon_file)
    if cl_uberon_df_full.empty:
        logging.warning("CL/UBERON data is empty after loading or missing HPO_ID. Proceeding without CL/UBERON filtering for phenotype.")
        phenotype_filtered_cl_uberon_df = pd.DataFrame() # Ensure it's an empty DataFrame
    else:
        phenotype_filtered_cl_uberon_df = cl_uberon_df_full[
            cl_uberon_df_full[CL_HPO_ID] == oli_phenotype_hpo_id
        ].copy()
        if phenotype_filtered_cl_uberon_df.empty:
            logging.warning(f"No CL/UBERON terms found for the OLI phenotype: {oli_phenotype_hpo_name} ({oli_phenotype_hpo_id}).")
        else:
            logging.info(f"Found {len(phenotype_filtered_cl_uberon_df)} CL/UBERON terms associated with OLI phenotype {oli_phenotype_hpo_name} ({oli_phenotype_hpo_id}).")

    generate_reports(phenotype_filtered_cl_uberon_df, cell_filtered_oli_df,
                      args.output_detailed_report, args.output_summary_tsv, args.output_relevant_cl_terms_report,
                      oli_phenotype_hpo_id, oli_phenotype_hpo_name,
                      args.similarity_threshold, args.top_n_ks, args.min_cells)

    logging.info("Script finished.")

if __name__ == "__main__":
    main()
