#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from scipy.stats import linregress
import sys

def plot_residual_distribution(input_file, phenotype_name):
    """
    Calculates residuals from a linear regression for a SPECIFIC phenotype and
    plots the distribution of those residuals as a histogram, using seaborn.
    Adds mean and median lines.
    """
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)

    if not lines:
        print("Error: Input file is empty.", file=sys.stderr)
        sys.exit(1)

    phenotype_data = {}
    header = lines[0].strip().split('\t')  # First line is header
    if "all_genes" not in header[0]:
        print("Error: 'all_genes' not found in the first column header.", file=sys.stderr)
        sys.exit(1)

    # Extract total expressed genes (first row, all columns except the first)
    total_expressed_genes = np.array([int(x) for x in lines[1].strip().split('\t')[1:]])
    X_tofit = total_expressed_genes.reshape(-1, 1)

    # Skip the "all_genes" row (first row)
    for line in lines[2:]:  # Start from third line
        parts = line.strip().split('\t')
        phenotype = parts[0]
        try:
            counts = np.array([int(x) for x in parts[1:]])
        except ValueError:
            print(f"Error: Non-numeric data found in line: {line.strip()}", file=sys.stderr)
            sys.exit(1)
        phenotype_data[phenotype] = counts

    if phenotype_name not in phenotype_data:
        print(f"Error: Phenotype '{phenotype_name}' not found in the input file.", file=sys.stderr)
        sys.exit(1)

    # Get the counts for the specified phenotype
    counts = phenotype_data[phenotype_name]

    # Perform linear regression
    model = linregress(X_tofit.flatten(), counts)

    # Calculate fitted values and residuals
    fitted_values = model.slope * X_tofit.flatten() + model.intercept
    residuals = counts - fitted_values

    # --- Plotting using Seaborn ---
    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True, bins=30, color='skyblue')

    # Add vertical lines for mean and median
    mean_res = np.mean(residuals)
    median_res = np.median(residuals)
    plt.axvline(mean_res, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_res:.2f}')
    plt.axvline(median_res, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_res:.2f}')

    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of Residuals for Phenotype: {phenotype_name}")  # Include phenotype in title
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"residual_distribution_{phenotype_name}.png")  # Unique file name
    print(f"Plot saved to residual_distribution_{phenotype_name}.png", file=sys.stderr)



if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Plot the distribution of residuals for a specific phenotype from a TSV file.")
    parser.add_argument("file_path", type=str, help="Path to the input TSV file.")
    parser.add_argument("phenotype_name", type=str, help="Name of the phenotype to analyze.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the plotting function with the provided file path and phenotype name
    plot_residual_distribution(args.file_path, args.phenotype_name)
