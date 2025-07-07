#!/usr/bin/python

import sys
import numpy as np
import math
from scipy import stats
from sklearn.linear_model import LinearRegression
import time
from collections import defaultdict

# --- Header (TSV Format) - Add new columns for dissection data ---
print("#1-hpo_id", "#2-hpo_name", "#3-n_related_genes",
      "#4-grouping_level", "#5-group_name", "#6-tissue", "#7-celltype", "#8-supercluster",
      "#9-cluster", "#10-subcluster", "#11-n_cells_in_group",
      "#12-n_background_cells", "#13-ks_statistic", "#14-ks_pvalue",
      "#15-effect_size", "#16-significant",
      "#17-dissection", "#18-dissection_celltype", "#19-dissection_supercluster", sep="\t", flush=True)

def convert_to_integers(string_list):
    try:
        return [int(float(item)) for item in string_list if item.strip()]
    except ValueError as e:
        print(f"Error converting list to integers: {e}", file=sys.stderr)
        return []

def cal_effect_size(Z, n1, n2):
    return Z / math.sqrt((n1*n2)/(n1+n2))

def perform_ks_test(group_indices, all_indices, distances):
    """Helper function to perform KS test and calculate statistics"""
    agrupation_distances = [distances[i] for i in group_indices]
    rest_indices = all_indices - group_indices
    rest_distances = [distances[i] for i in rest_indices]

    if len(rest_distances) == 0 or len(agrupation_distances) == 0:
        return "NA", "NA", "NA", len(agrupation_distances), len(rest_distances)

    try:
        ks_statistic, p_value = stats.ks_2samp(agrupation_distances, rest_distances)
        significative = int(p_value < 0.01)
        effect_size = cal_effect_size(ks_statistic, len(agrupation_distances), len(rest_distances))
        return ks_statistic, p_value, effect_size, len(agrupation_distances), len(rest_distances), significative
    except Exception as e:
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error in KS test: {e}", file=sys.stderr)
        return "NA", "NA", "NA", len(agrupation_distances), len(rest_distances), "NA"

def log_message(message):
    """Helper function for consistent logging"""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr)

# --- Input Files ---
ngenes_cutoff = 10
f_cell_annotation = sys.argv[1]
f_rel_hpos = sys.argv[2]
f_gene_expression = sys.argv[3]

# --- Logging ---
log_message("Starting script...")
log_message(f"Cell annotation file: {f_cell_annotation}")
log_message(f"HPO file: {f_rel_hpos}")
log_message(f"Gene expression file: {f_gene_expression}")

# --- Read Cell Annotation File (get column indices) ---
with open(f_cell_annotation, "r") as f:
    header = f.readline().strip().split(",")
    tissue_col = header.index('tissue')
    supercluster_col = header.index('supercluster_term')
    cluster_col = header.index('cluster_id')
    subcluster_col = header.index('subcluster_id')
    celltype_col = header.index('cell_type')
    dissection_col = 12  # 13th column (0-indexed)
log_message("Read header from cell annotation file.")

# --- Read HPO Information ---
l_hpos = []
with open(f_rel_hpos, "r") as f:
    for line in f:
        l_line = line.strip().split("\t")
        hpo_id = l_line[0]
        hpo_name = l_line[1]
        n_related_genes = int(l_line[3])
        if n_related_genes >= ngenes_cutoff:
            l_hpos.append([hpo_id, hpo_name, n_related_genes])
log_message(f"Read HPO information. {len(l_hpos)} HPOs loaded.")

# --- Read Gene Expression File ---
hpo_data = {}
total_expressed_genes = []
num_cells = 0

with open(f_gene_expression, "r") as f:
    # --- Read Total Expressed Genes (First Data Line) ---
    first_data_line = f.readline()
    total_expressed_genes = convert_to_integers(first_data_line.strip().split("\t")[1:])
    num_cells = len(total_expressed_genes)
    log_message(f"Read total expressed genes: {len(total_expressed_genes)} values")

    # --- Read HPO Data (Remaining Lines) ---
    for line in f:
        l_line = line.strip().split("\t")
        hpo_id = l_line[0]
        expression_values = convert_to_integers(l_line[1:])
        if len(expression_values) != num_cells:
            log_message(f"ERROR: Mismatch in data lengths for HPO {hpo_id}. Expected {num_cells}, got {len(expression_values)}")
            sys.exit(1)
        hpo_data[hpo_id] = expression_values
log_message("Read HPO expression data.")

# --- Cell Categorization (Create Dictionaries) ---
# Using defaultdict to simplify code
d_tissue2cell = defaultdict(set)
d_supercluster2cell = defaultdict(set)
d_cluster2cell = defaultdict(set)
d_subcluster2cell = defaultdict(set)
d_celltype2cell = defaultdict(set)
d_global_celltype2cell = defaultdict(set)
d_dissection2cell = defaultdict(set)
d_dissection_celltype2cell = defaultdict(set)
d_dissection_celltype_supercluster2cell = defaultdict(set)

# --- VALID TISSUES ---
valid_tissues = {
    "cerebellum", "cerebral cortex", "cerebral nuclei", "hippocampal formation",
    "hypothalamus", "midbrain", "myelencephalon", "pons", "spinal cord",
    "thalamic complex"
}

with open(f_cell_annotation, "r") as f:
    f.readline()  # Skip header

    for cell_index, line in enumerate(f):
        l_line = line.strip().split(",")
        if len(l_line) <= dissection_col:
            log_message(f"Warning: Line has fewer columns than expected: {line}")
            continue

        tissue = l_line[tissue_col].lower()
        supercluster = l_line[supercluster_col].lower()
        cluster = l_line[cluster_col].lower()
        subcluster = l_line[subcluster_col].lower()
        celltype = l_line[celltype_col].lower()
        dissection = l_line[dissection_col].lower() if l_line[dissection_col].strip() else "unknown"

        if tissue in valid_tissues:
            # --- Original categorization ---
            d_tissue2cell[tissue].add(cell_index)
            d_global_celltype2cell[celltype].add(cell_index)

            # --- Tissue-specific Cell Type ---
            celltype_key = f"{tissue}/{celltype}"
            d_celltype2cell[celltype_key].add(cell_index)

            # --- Supercluster (WITH celltype) ---
            supercluster_key = f"{tissue}/{celltype}/{supercluster}"
            d_supercluster2cell[supercluster_key].add(cell_index)

            # --- Cluster (WITH celltype) ---
            cluster_key = f"{tissue}/{celltype}/{supercluster}/{cluster}"
            d_cluster2cell[cluster_key].add(cell_index)

            # --- Subcluster (WITH celltype) ---
            subcluster_key = f"{tissue}/{celltype}/{supercluster}/{cluster}/{subcluster}"
            d_subcluster2cell[subcluster_key].add(cell_index)

            # --- NEW CATEGORIZATION FOR ADDITIONAL ANALYSIS ---
            d_dissection2cell[dissection].add(cell_index)

            dissection_celltype_key = f"{dissection}/{celltype}"
            d_dissection_celltype2cell[dissection_celltype_key].add(cell_index)

            dissection_celltype_supercluster_key = f"{dissection}/{celltype}/{supercluster}"
            d_dissection_celltype_supercluster2cell[dissection_celltype_supercluster_key].add(cell_index)

log_message(f"Cell categorization complete. {cell_index + 1} cells processed.")
log_message(f"Dissection categories: {len(d_dissection2cell)}")
log_message(f"Dissection-celltype categories: {len(d_dissection_celltype2cell)}")
log_message(f"Dissection-celltype-supercluster categories: {len(d_dissection_celltype_supercluster2cell)}")

# --- Cache linear regression model for reuse ---
X_tofit = np.array(total_expressed_genes).reshape(-1, 1)
model = LinearRegression()

# --- Precompute all_indices once ---
all_indices = set(range(len(total_expressed_genes)))

# --- HPO Processing (Hierarchical) ---
for n, id_n_name in enumerate(l_hpos, 1):
    hpo_id = id_n_name[0]
    hpo_name = id_n_name[1]
    n_related_genes = id_n_name[2]

    if hpo_id not in hpo_data:
        log_message(f"ERROR: HPO {hpo_id} not found in gene expression file.")
        sys.exit(1)

    l_related_expresed = hpo_data[hpo_id]
    log_message(f"Processing HPO: {hpo_id} ({hpo_name}) [{n}/{len(l_hpos)}]")

    # --- Linear Regression ---
    y_tofit = np.array(l_related_expresed)
    model.fit(X_tofit, y_tofit)
    m = model.coef_[0]
    b = model.intercept_
    log_message(f"  Linear regression complete for HPO: {hpo_id}")

    # --- Calculate Distances (vectorized) ---
    y_expected = X_tofit.flatten() * m + b
    distances = y_tofit - y_expected
    distances = distances.tolist()  # Convert to list for indexing

    # --- ORIGINAL HIERARCHICAL ANALYSIS ---
    # 0. Global Cell Types (across all tissues)
    log_message(f"  Analyzing Global Cell Types for HPO: {hpo_id}")
    for celltype, indices in d_global_celltype2cell.items():
        try:
            ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

            print(hpo_id, hpo_name, n_related_genes, "global_celltype",
                  celltype, "all_tissues", celltype, "NA", "NA", "NA",
                  n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                  "NA", "NA", "NA", sep="\t")
        except Exception as e:
            log_message(f"Error processing {hpo_id} at global celltype level: {e}")

    # 1. Tissues
    log_message(f"  Analyzing Tissues for HPO: {hpo_id}")
    for tissue, indices in d_tissue2cell.items():
        try:
            ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

            print(hpo_id, hpo_name, n_related_genes, "tissue",
                  tissue, tissue, "NA", "NA", "NA", "NA",
                  n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                  "NA", "NA", "NA", sep="\t")
        except Exception as e:
            log_message(f"Error processing {hpo_id} at tissue level: {e}")

    # 2. Cell Types (within each tissue)
    log_message(f"  Analyzing Cell Types for HPO: {hpo_id}")
    for celltype_key, indices in d_celltype2cell.items():
        tissue, celltype = celltype_key.split("/")
        try:
            ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

            print(hpo_id, hpo_name, n_related_genes, "celltype",
                  celltype, tissue, celltype, "NA", "NA", "NA",
                  n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                  "NA", "NA", "NA", sep="\t")
        except Exception as e:
            log_message(f"Error processing {hpo_id} at celltype level: {e}")

    # 3. Superclusters (within each cell type and tissue)
    log_message(f"  Analyzing Superclusters for HPO: {hpo_id}")
    for supercluster_key, indices in d_supercluster2cell.items():
        parts = supercluster_key.split("/")
        if len(parts) == 3:  # Ensure correct key format
            tissue, celltype, supercluster = parts
            try:
                ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

                print(hpo_id, hpo_name, n_related_genes, "supercluster",
                      supercluster, tissue, celltype, supercluster, "NA", "NA",
                      n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                      "NA", "NA", "NA", sep="\t")
            except Exception as e:
                log_message(f"Error processing {hpo_id} at supercluster level: {e}")

    # 4. Clusters (within each supercluster)
    log_message(f"  Analyzing Clusters for HPO: {hpo_id}")
    for cluster_key, indices in d_cluster2cell.items():
        parts = cluster_key.split("/")
        if len(parts) == 4:  # Ensure it is a tissue/celltype/supercluster/cluster
            tissue, celltype, supercluster, cluster = parts
            try:
                ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

                print(hpo_id, hpo_name, n_related_genes, "cluster",
                      cluster, tissue, celltype, supercluster, cluster, "NA",
                      n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                      "NA", "NA", "NA", sep="\t")
            except Exception as e:
                log_message(f"Error processing {hpo_id} at cluster level: {e}")

    # 5. Subclusters (within each cluster)
    log_message(f"  Analyzing Subclusters for HPO: {hpo_id}")
    for subcluster_key, indices in d_subcluster2cell.items():
        parts = subcluster_key.split("/")
        if len(parts) == 5:  # Ensure it is a tissue/celltype/supercluster/cluster/subcluster
            tissue, celltype, supercluster, cluster, subcluster = parts
            try:
                ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

                print(hpo_id, hpo_name, n_related_genes, "subcluster",
                      subcluster, tissue, celltype, supercluster, cluster, subcluster,
                      n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                      "NA", "NA", "NA", sep="\t")
            except Exception as e:
                log_message(f"Error processing {hpo_id} at subcluster level: {e}")

    # --- NEW ADDITIONAL HIERARCHICAL ANALYSIS ---
    log_message(f"  Starting additional dissection-based analysis for HPO: {hpo_id}")

    # 1. Dissection Level
    log_message(f"  Analyzing Dissections for HPO: {hpo_id}")
    for dissection, indices in d_dissection2cell.items():
        try:
            ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

            print(hpo_id, hpo_name, n_related_genes, "dissection",
                  dissection, "NA", "NA", "NA", "NA", "NA",
                  n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                  dissection, "NA", "NA", sep="\t")
        except Exception as e:
            log_message(f"Error processing {hpo_id} at dissection level: {e}")

    # 2. Dissection + Cell Type Level
    log_message(f"  Analyzing Dissection-CellTypes for HPO: {hpo_id}")
    for dissection_celltype_key, indices in d_dissection_celltype2cell.items():
        parts = dissection_celltype_key.split("/")
        if len(parts) == 2:  # Ensure correct key format
            dissection, celltype = parts
            try:
                ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

                print(hpo_id, hpo_name, n_related_genes, "dissection_celltype",
                      f"{dissection}_{celltype}", "NA", celltype, "NA", "NA", "NA",
                      n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                      dissection, f"{dissection}_{celltype}", "NA", sep="\t")
            except Exception as e:
                log_message(f"Error processing {hpo_id} at dissection-celltype level: {e}")

    # 3. Dissection + Cell Type + Supercluster Level
    log_message(f"  Analyzing Dissection-CellType-Superclusters for HPO: {hpo_id}")
    for dissection_celltype_supercluster_key, indices in d_dissection_celltype_supercluster2cell.items():
        parts = dissection_celltype_supercluster_key.split("/")
        if len(parts) == 3:  # Ensure correct key format
            dissection, celltype, supercluster = parts
            try:
                ks_statistic, p_value, effect_size, n_group, n_rest, significative = perform_ks_test(indices, all_indices, distances)

                print(hpo_id, hpo_name, n_related_genes, "dissection_celltype_supercluster",
                      f"{dissection}_{celltype}_{supercluster}", "NA", celltype, supercluster, "NA", "NA",
                      n_group, n_rest, ks_statistic, p_value, effect_size, significative,
                      dissection, f"{dissection}_{celltype}", f"{dissection}_{celltype}_{supercluster}", sep="\t")
            except Exception as e:
                log_message(f"Error processing {hpo_id} at dissection-celltype-supercluster level: {e}")

log_message("Script finished.")

