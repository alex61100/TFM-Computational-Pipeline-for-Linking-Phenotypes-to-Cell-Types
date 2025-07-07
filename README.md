Computational Pipeline for Linking Phenotypes to Cell Types  
________________________________________
This repository contains the complete computational pipeline developed to identify and analyze associations between human disease phenotypes and specific brain cell types using single-nucleus RNA-sequencing data.   
The workflow begins by processing the transcriptomic data to quantify the expression of phenotype-associated gene sets for each of nearly four million cells. Subsequent scripts perform a robust statistical analysis to generate phenotype-cell group associations, run quality control analyses to determine reliable data thresholds, and finally, produce a suite of visualizations and validation reports to interpret the findings.  
The sections below provide detailed instructions for each script in the pipeline  
________________________________________
README: h5ad2genes.py  
Description  
This script processes a single-cell dataset (.h5ad) to quantify the expression of gene sets associated with Human Phenotype Ontology (HPO) terms. It counts, for each cell, how many genes linked to a given phenotype show non-zero expression.  
Requirements  
•	Python 3   
•	scanpy  
•	numpy  
Usage  
The script is run from the command line and requires redirecting the output to a file.  
python3 h5ad2genes.py [h5ad_file] [hpo_list_file] [hpo_genes_json] [optional_tissue_name] > [output.tsv]  
Example:  
python3 h5ad2genes.py data.h5ad hpos.tsv genes.json "brain" > results.tsv  
Inputs  
1.	[h5ad_file]: Path to the .h5ad file with raw counts.  
2.	[hpo_list_file]: Path to a tab-separated file (Column 1: HPO ID, Column 2: HPO Name).  
3.	[hpo_genes_json]: Path to a JSON file mapping HPO IDs to lists of Ensembl gene IDs.  
4.	[optional_tissue_name]: An optional string to label the output.  
Output  
A tab-separated file where:  
•	The first row details the total number of unique genes expressed per cell.  
•	Each subsequent row begins with an HPO ID and lists the per-cell counts of its associated expressed genes.  
Note  
A hardcoded cutoff (ngenes_cutoff = 10) is used; HPO terms with fewer than 10 associated genes will be ignored.    
________________________________________
README: merge_files.py  
Description  
This script merges two tab-separated data files that share a similar row structure, such as the outputs for neuronal and non-neuronal cells from the h5ad2genes.py script. It performs a row-wise merge by matching identifiers in the first column (e.g., an HPO ID) and concatenating the data from both files into a single row in the final output.  
Requirements  
•	Python 3  
•	No external libraries are required.  
Usage  
The script takes three positional arguments from the command line.  
python3 merge_files.py [input_file_1] [input_file_2] [output_file]  
Example:  
python3 merge_files.py neuron_counts.tsv non_neuron_counts.tsv all_counts.tsv  
Inputs  
1.	[input_file_1]: Path to the first input tab-separated file.  
2.	[input_file_2]: Path to the second input tab-separated file.  
3.	[output_file]: Path for the new, merged output file to be created.  
________________________________________  
README: extract_h5ad_obs.py  
Description  
This script extracts the cell metadata table (the .obs attribute) from a single-cell AnnData file (.h5ad) and saves it as a comma-separated values (CSV) file.  
Requirements  
•	Python 3  
•	scanpy  
•	pandas  
Usage  
The script takes two positional arguments: the input file and the output file.  
python3 extract_h5ad_obs.py [input.h5ad] [output.csv]  
Example:  
python3 extract_h5ad_obs.py brain_data.h5ad brain_cell_metadata.csv  
Inputs  
1.	[input.h5ad]: Path to the source AnnData file.  
2.	[output.csv]: Path for the output CSV file to be created.  
Output  
A single CSV file containing all the cell annotations from the .obs table of the input file.  
________________________________________  
README: OLI_file_generator.py  
Description  
This is the primary statistical analysis script that generates the Ontology Linkage Information (OLI) files. For each input HPO term, it tests for significant associations against numerous, hierarchically-defined cell groups.  
Requirements  
•	Python 3  
•	numpy  
•	scipy  
•	scikit-learn  
Usage  
The script takes three positional arguments and prints a comprehensive TSV file to standard output, which should be redirected to a file.  
python3 OLI_file_generator.py [cell_annotations.csv] [hpo_list.tsv] [gene_counts.tsv] > [output_OLI_file.tsv]  
Example:  
python3 OLI_file_generator.py cell_meta.csv hpos.tsv counts.tsv > HPO_Seizure.oli.tsv  
Inputs  
1.	[cell_annotations.csv]: A CSV file with cell metadata (tissue, cell type, cluster, etc.).  
2.	[hpo_list.tsv]: A tab-separated file listing the HPO terms to analyze.  
3.	[gene_counts.tsv]: The consolidated gene count file from the merge_files.py script.  
Output  
A comprehensive tab-separated (TSV) file where each row contains the results of a single statistical test for an HPO term against a specific cell group. Columns include the cell group definition, cell counts, KS statistic, p-value, effect size, and a significance flag.  
Methodology Note  
For each HPO term, the script first performs a linear regression to calculate residuals, correcting for overall gene expression. It then performs a two-sample KS test on these residuals to compare each cell group against the background population of all other cells.  
________________________________________  
README: determine_cell_cutoff.py  
Description  
This script performs a quality control analysis on a directory of OLI files to help determine a data-driven minimum cell count threshold. It analyzes how statistical metrics change as the minimum required cell group size increases and generates several summary plots to visualize these trends.  
Requirements  
•	Python 3  
•	pandas  
•	numpy  
•	matplotlib  
•	scipy  
Usage  
The script is run from the command line, taking a directory path as the main input and optional flags for the output prefix and p-value cutoff.  
python3 determine_cell_cutoff.py [directory_path] -o [output_prefix]  
Example:  
python3 determine_cell_cutoff.py ./oli_results/ -o ./qc_plots/cutoff_analysis  
Inputs & Arguments  
•	[directory_path]: (Required) The path to the directory containing the input OLI (.tsv) files.  
•	-o, --output_prefix: (Optional) A prefix for all output plot filenames. Default: hpo_analysis.  
•	-p, --p_value_cutoff: (Optional) The p-value threshold for significance, used for plotting a reference line. Default: 0.05.  
Outputs  
A series of PNG plots are saved to the path specified by the output prefix. Key plots include:  
•	[prefix]_aggregated_plots.png: A three-panel plot showing how aggregated statistics change with the cell count threshold.  
•	[prefix]_ks_cell_relationship.png: A scatter plot of KS statistic vs. cell count for all data points.  
•	[prefix]_ks_cell_hexbin.png: A 2D density plot of the same data.  
•	[prefix]_knee_point_histogram.png: A histogram of suggested thresholds calculated from each input file.  
________________________________________  
README: standard_deviation_skewness.py  
Description  
This script analyzes the statistical properties of gene expression residuals for each phenotype from a consolidated gene count file. For each phenotype, it performs a linear regression against total gene expression, calculates the residuals, and then computes their standard deviation and skewness.  
Requirements  
•	Python 3  
•	numpy  
•	scipy  
•	matplotlib  
Usage  
The script takes a single input file path as a positional argument. It prints a summary table to the console and saves a plot to a file.  
python3 standard_deviation_skewness.py [ngenes_output.tsv]  
To save the summary table, redirect the output:  
python3 standard_deviation_skewness.py [ngenes_output.tsv] > skewness_sd_summary.tsv  
Input  
•	[ngenes_output.tsv]: (Required) The path to the consolidated gene count file generated by merge_files.py. The first data row must correspond to 'all_genes' and contain the total genes expressed per cell.  
Outputs  
1.	A tab-separated table printed to the console with three columns: Phenotype, Skewness_Residuals, and SD_Residuals.  
2.	A PNG image file named skewness_vs_sd_residuals.png is saved in the current directory. This file is a scatter plot showing the standard deviation versus the skewness for all analyzed phenotypes.  
________________________________________  
README: visualize_distances_distribution.py  
Description  
This script generates a detailed visualization for a single, specified phenotype. It calculates the residuals from a linear regression and plots their frequency distribution as a histogram to allow for in-depth inspection of its statistical properties (like skewness and dispersion).  
Requirements  
•	Python 3  
•	pandas  
•	numpy  
•	matplotlib  
•	seaborn  
•	scipy  
Usage  
The script takes two required positional arguments from the command line.  
python3 visualize_distances_distribution.py [ngenes_output.tsv] [phenotype_id]  
Example:  
python3 visualize_distances_distribution.py all_counts.tsv HP:0001342  
Inputs / Arguments  
1.	[ngenes_output.tsv]: (Required) The path to the consolidated gene count file generated by merge_files.py.  
2.	[phenotype_id]: (Required) The specific HPO ID string for the phenotype you want to analyze (e.g., "HP:0001342").  
Output  
A single PNG image file is saved in the current directory.  
•	Filename: The name is generated dynamically based on the input, e.g., residual_distribution_HP:0001342.png.  
•	Content: A histogram showing the distribution of residuals for the specified phenotype, enhanced with a Kernel Density Estimate (KDE) curve and vertical lines for the mean and median values.  
________________________________________  
README: matrix_filtered_cell_tissues.py  
Description  
This script consolidates all individual OLI result files from a directory into a single, comprehensive HPO-by-cell-group data matrix. It filters the data to keep only significant associations (p < 0.01) that meet a specified minimum cell count threshold before generating the final matrix.  
Requirements  
•	Python 3  
•	pandas  
•	numpy  
Usage  
The script takes a directory of OLI files as input and generates a single matrix file.  
python3 matrix_filtered_cell_tissues.py [directory_path] -o [output_matrix.tsv] -m [min_cell_count]  
Example:  
python3 matrix_filtered_cell_tissues.py ./oli_results/ -o hpo_cellgroup_matrix.tsv -m 70  
Inputs / Arguments  
•	[directory_path]: (Required) The path to the directory containing the input OLI (.tsv) files.  
•	-o, --output: (Optional) The path for the final output matrix file. Default: "hpo_cellgroup_matrix.tsv".  
•	-m, --min-cells: (Optional) The minimum number of cells required for an association to be included. Default: 0 (no filtering).  
Output  
A single tab-separated (TSV) file containing the final data matrix where:  
•	Rows are HPO phenotypes.  
•	Columns are unique, hierarchically-defined cell groups.  
•	Values are the KS statistics for each association. A value of 0 indicates no significant association was found after filtering.  
________________________________________  
README: visualization_matrix.py  
Description  
This script takes the final HPO-by-cell-group data matrix and generates a suite of summary analyses and visualizations. It identifies top-ranking associations and creates several plots to help interpret the overall results, including ranked bar charts and a comprehensive clustered heatmap.  
Requirements  
•	Python 3  
•	pandas  
•	numpy  
•	matplotlib  
•	seaborn  
•	scipy  
Usage  
The script takes the path to the matrix file as input, along with optional flags to control the output.  
python3 visualization_matrix.py [input_matrix.tsv] -o [output_prefix] -n [top_n_count]  
Example:  
python3 visualization_matrix.py hpo_cellgroup_matrix.tsv -o ./plots/final_analysis -n 25  
Inputs / Arguments  
•	[input_matrix.tsv]: (Required) The path to the input HPO-by-cell-group matrix file.  
•	-o, --output_prefix: (Optional) A prefix for all output plot and text filenames. Default: hpo_cg_maxks_analysis.  
•	-n, --top_n: (Optional) The number of top items to display in ranking plots. Default: 25.  
•	-m, --max_cell_groups: (Optional) The maximum number of cell groups to include in the full heatmap to ensure it is plottable. Default: 600.  
Outputs  
A collection of PNG plots and text summaries saved with the specified prefix. Key outputs include:  
•	[prefix]_full_heatmap.png: A large, hierarchically-clustered heatmap of the HPO-cell group associations.  
•	[prefix]_top_[n]_hpo_by_max_ks.png: A bar plot of the top-ranked HPO terms by their strongest association.  
•	[prefix]_top_[n]_cellgroup_by_max_ks.png: A bar plot of the top-ranked cell groups.  
•	[prefix]_comprehensive_hpo_analysis.png: A scatter plot summarizing the association profile of each HPO term.  
•	[prefix]_comprehensive_hpo_analysis_summary.txt: A text file with summary statistics and top 10 lists.  
________________________________________  
README: phenotype_group_meta_analyzer.py  
Description  
This script performs a meta-analysis on the HPO-by-cell-group matrix. It moves beyond individual phenotype associations to test whether entire categories of functionally related phenotypes (defined by the HPO hierarchy) show a collective association with specific cell types.  
Requirements  
•	Python 3  
•	pandas  
•	numpy  
•	scipy  
•	pronto (for reading the .obo ontology file)  
Usage  
The script requires paths to the ontology file, the input matrix, and an output directory.  
python3 phenotype_group_meta_analyzer.py --hpo_obo [hp.obo] --input_ks_matrix [matrix.tsv] --output_dir [results_dir]  
Example:  
python3 phenotype_group_meta_analyzer.py --hpo_obo hp.obo --input_ks_matrix hpo_cellgroup_matrix.tsv --output_dir ./meta_analysis_results/  
Inputs / Arguments  
•	--hpo_obo: (Required) Path to the HPO ontology file (hp.obo).  
•	--input_ks_matrix: (Required) Path to the input HPO-by-cell-group matrix.  
•	--output_dir: (Required) Path to the directory where output files will be saved.  
•	--start_hpo_id: (Optional) The root HPO term to start the analysis from.  
•	--min_hpos: (Optional) Minimum descendants a term must have for its children to be analyzed.  
•	--min_pvalue: (Optional) If provided, filters the final output matrices to only include highly significant meta-associations.  
Output  
Two tab-separated (TSV) files saved in the specified output directory:  
1.	meta_ks_statistic.tsv: A matrix where rows are HPO groups, columns are cell types, and values are the "meta" KS statistics.  
2.	meta_pvalue.tsv: An identically structured matrix containing the corresponding p-values.  
Methodology Note  
For each selected HPO group and cell type, the script performs a secondary two-sample KS test comparing the distribution of primary KS statistics from HPOs within the group to those outside the group.  
________________________________________
README: visualize_hpo_ontology_order_ks_pvalue.py  
Description  
This script creates a suite of visualizations for the meta-analysis results (the output of phenotype_group_meta_analyzer.py). Its key feature is generating a heatmap where HPO groups (rows) are clustered based on their semantic similarity within the HPO ontology itself, not just by their data values.  
Requirements  
•	Python 3  
•	pandas  
•	numpy  
•	matplotlib  
•	seaborn  
•	scipy  
•	pronto  
•	networkx  
Usage  
The script requires the meta-analysis matrix, the HPO ontology file, and the type of data ('ks' or 'pvalue') as input.  
python3 visualize_hpo_ontology_order_ks_pvalue.py [input_matrix.tsv] --hpo_obo [hp.obo] --type_of_data [ks|pvalue] -o [prefix]  
Example:  
python3 visualize_hpo_ontology_order_ks_pvalue.py meta_pvalue.tsv --hpo_obo hp.obo --type_of_data pvalue -o ./final_plots/meta_viz  
Inputs / Arguments  
•	[input_matrix.tsv]: (Required) Path to the input meta-analysis matrix (either KS or p-value).  
•	--hpo_obo: (Required) Path to the HPO ontology file (hp.obo).  
•	--type_of_data: (Required) The type of data in the matrix. Must be 'ks' or 'pvalue'. If 'pvalue', a -log10 transformation is applied.  
•	-o, --output_prefix: (Optional) A prefix for all output plot and text filenames.  
•	-n, --top_n: (Optional) The number of top items for ranking plots.  
•	--max_rows: (Optional) Maximum number of HPO groups to include in the main heatmap.  
•	--select_hpos: (Optional) A list of specific HPO group IDs to plot in a separate heatmap.  
Output  
A collection of PNG plots and text summaries saved with the specified prefix. The main output is a dual-clustered heatmap where rows (HPO groups) are ordered by their similarity in the ontology, and columns (cell types) are ordered by their association data profiles.  
________________________________________  
README: compare_cl_to_oli.py  
Description  
This script validates and annotates the results from a single OLI file. It takes the OLI file for one phenotype and a knowledge base of known HPO-to-anatomical/cell-type (CL/Uberon) associations. It then uses fuzzy string matching to find and rank correspondences between the two, grounding the statistical findings in known biology.  
Requirements  
•	Python 3  
•	pandas  
•	thefuzz (and its dependency python-Levenshtein for speed)  
Usage  
The script requires paths to the OLI file for the phenotype of interest and the CL/UBERON knowledge base file.  
python3 compare_cl_to_oli.py --oli_file [path_to_oli.tsv] --cl_uberon_file [path_to_cl_uberon.tsv]  
Example:  
python3 compare_cl_to_oli.py --oli_file ./results/HP0007166.tsv --cl_uberon_file ./data/comentg_data.tsv --min_cells 70  
Inputs / Arguments  
•	--oli_file: (Required) Path to the single OLI file for the phenotype of interest.  
•	--cl_uberon_file: (Required) Path to the tab-separated knowledge base file linking HPOs to CL/Uberon terms.  
•	--min_cells: (Optional) Minimum cell count to pre-filter the OLI data before matching. Default: 5.  
•	--top_n_ks: (Optional) For each CL/Uberon term, the script ranks all its OLI matches by KS statistic and reports the top N. Default: 5.  
•	--output...: (Optional) Flags to specify custom names for the three output report files.  
Output  
By default, the script generates three report files in the current directory:  
1.	cl_to_oli_topN_pheno_report.txt: A detailed, human-readable report of all validated matches found.  
2.	cl_to_oli_topN_pheno_summary.tsv: A machine-readable TSV file combining all data for every successful link.  
3.	cl_to_oli_topN_pheno_relevant_terms.txt: A summary that ranks the CL/Uberon terms by how successfully they were matched to the OLI data.  

