#!/usr/bin/python

import sys
import json
import numpy as np
import datetime
import scanpy
import time
from datetime import datetime

# Updated Command:
# Usage: python3 h5ad2genes.py singlecell_file_tissue.h5ad hpo_list_file hpo2genes.json [optional: "tissue_name"] > ngenes_output.tsv

##-- INPUTS --##

# singlecell_file_tissue.h5ad: h5ad file from single-cell data.
# hpo_list_file: Tab-separated file listing HPO IDs and names.
# hpo2genes.json: JSON mapping HPOs to related genes.
# tissue_name (optional): Name of tissue/organ. If omitted, tissue-specific output is skipped.

##-- OUTPUTS --##

# ngenes_output.tsv: Number of HPO-related genes expressed per cell.

#-- FIXED PARAMS --#
ngenes_cutoff = 10  # Only HPOs with at least this number of related genes will be processed.

#------------------#

# Print start time
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("Launched at:", dt_string, file=sys.stderr)
sys.stderr.flush() # Flush stderr

# Read single-cell data
try:
    ts_file = scanpy.read_h5ad(sys.argv[1])
    raw_counts = ts_file.X
    l_genes_wvariants = ts_file.var_names
    l_genes = [gene.split(".")[0] for gene in l_genes_wvariants]  # Remove variant IDs from Ensembl IDs
except FileNotFoundError:
    print(f"Error: h5ad file '{sys.argv[1]}' not found.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error reading h5ad file: {e}", file=sys.stderr)
    sys.exit(1)
# Check if tissue name is provided
tissue_name = sys.argv[4].lower() if len(sys.argv) > 4 else None

# Load HPO to genes mapping
try:
    with open(sys.argv[3], 'r') as f:
        hpo2genes = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file '{sys.argv[3]}' not found.", file=sys.stderr)
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{sys.argv[3]}'.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error loading JSON file: {e}", file=sys.stderr)
    sys.exit(1)

# Load list of HPOs to analyze
l_hpos = []
try:
    with open(sys.argv[2], "r") as f:
        for line in f:
            l_line = line.strip().split("\t")

            hpo_id, hpo_name = l_line[0], l_line[1]

            try:
                n_related_genes_json = len(hpo2genes[hpo_id]['EnsemblID'])  # Use the JSON for the cutoff
                if n_related_genes_json >= ngenes_cutoff:
                    l_hpos.append([hpo_id, hpo_name])
            except KeyError:
                pass  # Skip HPOs not found in hpo2genes.json
except FileNotFoundError:
    print(f"Error: HPO list file '{sys.argv[2]}' not found.", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error reading HPO list file: {e}", file=sys.stderr)
    sys.exit(1)

print("INPUT FILES READ SUCCESSFULLY!", file=sys.stderr)
sys.stderr.flush()
print("PREPROCESSING...", file=sys.stderr)
sys.stderr.flush()

# Begin processing
n_hpos = len(l_hpos)
print("DONE! Number of HPOs that are going to be processed:", str(n_hpos), file=sys.stderr)
sys.stderr.flush()

l_total_expressed = []
first_lap = True
n = 0

# Loop through each HPO term
for hpoid_and_name in l_hpos:
    start_time = time.perf_counter()
    hpo, hponame = hpoid_and_name
    l_related_expressed = []
    n += 1
    print(f"Processing {hpo} ({hponame}) [{n}/{n_hpos}]", file=sys.stderr)
    sys.stderr.flush()

    related_genes = hpo2genes[hpo]['EnsemblID']
    # Convert related_genes to a set for faster lookup
    related_genes_set = set(related_genes)

    cell_indices = range(raw_counts.shape[0])

    # Loop through each cell
    for n_cell in cell_indices:
        n_related = 0
        non0_indices = raw_counts[n_cell].indices  # Indices of expressed genes in this cell

        #Efficient lookup using the set
        for expressed_gene_index in non0_indices:
           expressed_gene = l_genes[expressed_gene_index]
           if expressed_gene in related_genes_set: # Use set for lookup
              n_related += 1

        if first_lap:
            l_total_expressed.append(len(non0_indices))  # Total genes expressed per cell

        l_related_expressed.append(n_related)

    # Output total genes expressed (only on first HPO)
    if first_lap:
        if tissue_name:
            print(f"{tissue_name}|all_genes" + "\t" + "\t".join(str(x) for x in l_total_expressed))
            sys.stdout.flush() #flush output
        else:
            print("all_genes" + "\t" + "\t".join(str(x) for x in l_total_expressed))
            sys.stdout.flush() #flush output

    # Output HPO-related gene expression
    print(hpo + "\t" + "\t".join(str(x) for x in l_related_expressed))
    sys.stdout.flush() # CRITICAL: Flush after each HPO's results

    first_lap = False

    # Timing for each HPO
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds", file=sys.stderr)
    sys.stderr.flush()

print("ENDED SUCCESSFULLY", file=sys.stderr)
sys.stderr.flush()
