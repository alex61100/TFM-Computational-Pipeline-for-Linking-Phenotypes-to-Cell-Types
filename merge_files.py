#!/usr/bin/python

import sys

def merge_files(file1, file2, output_file):
    merged_data = {}

    def read_file(file_path):
        """ Reads a file and returns a dictionary {row_name: [list of numbers]} """
        data = {}
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                key = parts[0]  # First column (row name)
                values = parts[1:]  # Keep as strings to preserve formatting
                data[key] = values
        return data

    print(f"Reading {file1}...")
    data1 = read_file(file1)

    print(f"Reading {file2}...")
    data2 = read_file(file2)

    # Merge the first '|all_genes' row from both files
    first_key1 = next((key for key in data1 if '|all_genes' in key), None)
    first_key2 = next((key for key in data2 if '|all_genes' in key), None)

    if first_key1 and first_key2:
        # If both files have '|all_genes' rows, merge them
        print(f"Merging first '|all_genes' rows...")
        merged_data[first_key1] = data1[first_key1] + data2[first_key2]
    elif first_key1:
        # If only the first file has a '|all_genes' row, add it as is
        merged_data[first_key1] = data1[first_key1]
    elif first_key2:
        # If only the second file has a '|all_genes' row, add it as is
        merged_data[first_key2] = data2[first_key2]

    # Merge other rows based on matching keys
    print("Merging remaining rows...")
    for key in data1:
        if key != first_key1:  # Skip the already merged first row
            if key in data2:
                # If key exists in both files, combine the values
                merged_data[key] = data1[key] + data2[key]
            else:
                # If key is only in the first file, just add it
                merged_data[key] = data1[key]

    for key in data2:
        if key != first_key2:  # Skip the already merged first row
            if key not in data1:
                # If key is only in the second file, just add it
                merged_data[key] = data2[key]

    # Write merged output
    print(f"Writing output to {output_file}...")
    with open(output_file, 'w') as out:
        for key, values in merged_data.items():
            out.write(f"{key}\t" + '\t'.join(values) + "\n")

    print(f"Merging complete! Output saved in {output_file}")

if __name__ == "__main__":
    merge_files(sys.argv[1], sys.argv[2], sys.argv[3])

