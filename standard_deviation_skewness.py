#!/usr/bin/env python3

import sys
import numpy as np
from scipy.stats import skew, linregress
import matplotlib.pyplot as plt


def calculate_residual_stats_and_plot(input_file):
    """
    Calculates skewness and standard deviation of residuals from a linear
    regression for each phenotype and generates a scatter plot.
    """
    try:
        print(f"[INFO] Reading input file: {input_file}", file=sys.stderr)
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[ERROR] Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    if not lines:
        print("[ERROR] Input file is empty.", file=sys.stderr)
        sys.exit(1)

    print("[INFO] Processing input file...")
    phenotype_data = {}
    header = lines[0].strip().split('\t')  # First line is header
    if "all_genes" not in header[0]:
        print("[ERROR] 'all_genes' not found in the first column header.", file=sys.stderr)
        sys.exit(1)

    # Extract total expressed genes (first row, all columns except the first)
    print("[INFO] Extracting total expressed genes...")
    total_expressed_genes = np.array([int(x) for x in lines[1].strip().split('\t')[1:]])
    X_tofit = total_expressed_genes.reshape(-1, 1)

    # Skip the "all_genes" row (first row)
    for line in lines[2:]:  # Start from third line
        parts = line.strip().split('\t')
        phenotype = parts[0]
        try:
            counts = np.array([int(x) for x in parts[1:]])
        except ValueError:
            print(f"[ERROR] Non-numeric data found in line: {line.strip()}", file=sys.stderr)
            sys.exit(1)

        phenotype_data[phenotype] = counts

    print("[INFO] Computing skewness and standard deviation of residuals...")
    skewness_values = []
    sd_values = []
    phenotype_labels = []

    print("Phenotype\tSkewness_Residuals\tSD_Residuals")

    for phenotype, counts in phenotype_data.items():
        print(f"[INFO] Processing phenotype: {phenotype}", file=sys.stderr)
        # Perform linear regression
        model = linregress(X_tofit.flatten(), counts)
        
        # Calculate fitted values and residuals
        fitted_values = model.slope * X_tofit.flatten() + model.intercept
        residuals = counts - fitted_values

        # Compute skewness and standard deviation of residuals
        phenotype_skewness = skew(residuals)
        phenotype_sd = np.std(residuals)

        skewness_values.append(phenotype_skewness)
        sd_values.append(phenotype_sd)
        phenotype_labels.append(phenotype)

        print(f"{phenotype}\t{phenotype_skewness:.4f}\t{phenotype_sd:.4f}")

    print("[INFO] Generating scatter plot...", file=sys.stderr)
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(skewness_values, sd_values)

    for i, label in enumerate(phenotype_labels):
        plt.annotate(label, (skewness_values[i], sd_values[i]), textcoords="offset points", xytext=(5, 5), ha='left')

    plt.xlabel("Skewness of Residuals")
    plt.ylabel("Standard Deviation of Residuals")
    plt.title("Skewness vs. Standard Deviation of Phenotype Gene Expression Residuals")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("skewness_vs_sd_residuals.png")
    print("[INFO] Plot saved to skewness_vs_sd_residuals.png", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("[ERROR] Usage: python3 h5ad2genes_stats.py ngenes_output.tsv", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    print("[INFO] Starting script execution...", file=sys.stderr)
    calculate_residual_stats_and_plot(input_file)
    print("[INFO] Script execution completed.", file=sys.stderr)

