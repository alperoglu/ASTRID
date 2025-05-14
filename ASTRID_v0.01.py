import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import warnings
import time
import json
import textwrap
from ASTRID_Tools import clustering_recipe_custom, process_cluster, calculate_nmi, plot_confusion_matrix, runSingleR, evaluate_formula, calculate_odds_ratios, calculate_chr_damage

from os.path import abspath, dirname
os.chdir(dirname(abspath(__file__)))

warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize']=(5,5) #rescale figures
plt.rcParams['pdf.fonttype']=42 #for vectorized text in pdfs
sns.set_theme(style="white")


def astrid_clustering(adata, input_prefix, outDir, cutoff_level=None):

    from multiprocessing import Pool
    from umap import UMAP

    # print message that clustering is running
    from scipy.sparse import csr_matrix

    print("Running clustering")

    # check if there is a column in adata.obs that matches the authorType in adata.uns and if not give an error
    if adata.uns["authorType"] not in adata.obs.columns:
        raise ValueError("authorType is not in adata.obs. Please check the authorType in adata.uns")

    # convert adata.X to sparse matrix if it is an array
    if type(adata.X) == np.ndarray:
        adata.X = csr_matrix(adata.X)

    adata.layers["raw"] = adata.X

    adata = clustering_recipe_custom(adata, k=100, resolution=0.1, clustering_key="clustering_level_1")

    mapper = UMAP().fit(adata.obsm['X_pca_norm_counts'][:, 0:100])
    adata.obsm['X_umap'] = mapper.embedding_

    plot_confusion_matrix(adata, "clustering_level_1", adata.uns["authorType"], outDir + input_prefix + "_" + "clustering_level_1_vs_" + adata.uns["authorType"] +"_confusion_matrix.pdf")

    # print that first level of clustering is done and also print the number of clusters identified and norm_mi between the author cell type column and clustering_level_1 using norm_mi from RefSTRATools
    print("First level of clustering done")
    print("Number of clusters: " + str(adata.obs["clustering_level_1"].nunique()))
    print("Normalized mutual information between clustering_level_1 and " + adata.uns["authorType"] + ": " + str(calculate_nmi(adata, "clustering_level_1", adata.uns["authorType"])))

    if cutoff_level is not None and cutoff_level == 1:

        adata.obs["clustering_level_1"] = adata.obs["clustering_level_1"].astype(str)

        adata.uns["final_clustering_level"] = "clustering_level_1"
        adata.obs["final_clustering_level"] = adata.obs["clustering_level_1"]

        return adata

    clusters = adata.obs["clustering_level_1"].unique().tolist()

    clusters = sorted([clus for clus in clusters if clus != "unassigned"])

    # recluster each subset of cluster wu_data and save the obs of each subset to a pandas dataframe
    level2_clustering = pd.DataFrame()

    for cl1 in clusters:
        tmp_adata = adata[adata.obs.clustering_level_1 == cl1].copy()
        
        if "log1p" in tmp_adata.uns:
            del tmp_adata.uns["log1p"]

        tmp_adata = clustering_recipe_custom(tmp_adata, clustering_key="clustering_level_2", k =12, resolution=0.3, do_highly_variable = True)

        # add the level 1 cluster to the level 2 cluster
        tmp_adata.obs['clustering_level_2'] = str(cl1) + "_" + tmp_adata.obs['clustering_level_2']

        level2_clustering = pd.concat([level2_clustering, tmp_adata.obs.loc[:, 'clustering_level_2']], axis=0)
        del tmp_adata

    # give teh columns names from old coh4_adata and also the new clusterings based on model, distance, k, and resolution
    level2_clustering.columns = ["clustering_level_2"]

    # drop the columns starting with clustering_level_2 if they exist
    adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if col.startswith("clustering_level_2")])

    adata.obs = adata.obs.merge(level2_clustering, left_index=True, right_index=True, how='left')
    adata.obs.loc[adata.obs.clustering_level_1 == "unassigned", "clustering_level_2"] = "unassigned"

    adata = adata[adata.obs["clustering_level_2"].str.contains("unassigned") == False]

    current_clustering_level = 3

    clustering_depth = 12

    if cutoff_level is not None:
        clustering_depth = cutoff_level

    while current_clustering_level < clustering_depth and adata.obs["clustering_level_" + str(current_clustering_level-2)].to_list() != adata.obs["clustering_level_" + str(current_clustering_level-1)].to_list():
        
        # print(current_clustering_level)

        clusters = adata.obs["clustering_level_" + str(current_clustering_level - 1)].unique().tolist()

        clusters = sorted(clusters)

        level2_clustering = pd.DataFrame()

        new_key = "clustering_level_" + str(current_clustering_level)

        # split the adata object into subsets based on the clusters in "clustering_level_" + str(current_clustering_level - 1) and zip that along with clusters list and current_clustering_level to be passed to the process_cluster function
        tmp_adata = [adata[adata.obs["clustering_level_" + str(current_clustering_level - 1)] == cl1].copy() for cl1 in clusters]

        with Pool(processes=10) as pool:
            results = pool.map(process_cluster, zip(clusters, [current_clustering_level] * len(clusters), tmp_adata, [adata.n_obs] * len(clusters) ))

        del tmp_adata

        level2_clustering = pd.concat(results, axis=0)
        # give teh columns names from old coh4_adata and also the new clusterings based on model, distance, k, and resolution
        level2_clustering.columns = [new_key]

        # drop the columns starting with clustering_level_2 if they exist
        adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if col.startswith(new_key)])

        adata.obs = adata.obs.merge(level2_clustering, left_index=True, right_index=True, how='left')
        
        adata = adata[adata.obs[new_key].str.contains("unassigned") == False]

        current_clustering_level = current_clustering_level + 1

    final_key = "clustering_level_" + str(current_clustering_level-1)

    # store the final clustering level in adata object inside uns
    adata.uns["final_clustering_level"] = final_key

    # add a column to adata.obs called final_clustering_level and set it to the final_key column
    adata.obs["final_clustering_level"] = adata.obs[final_key]
    
    plot_confusion_matrix(adata, final_key, adata.uns["authorType"], outDir + input_prefix + "_" +  final_key + "_vs_" + adata.uns["authorType"] + "_confusion_matrix.pdf")

    # print that clustering is done and also print the final clustering level and number of clusters identified and some other information
    print("Clustering done")
    print("Final clustering level: " + final_key)
    print("Number of clusters: " + str(adata.obs[final_key].nunique()))
    print("Normalized mutual information between " + final_key + " and " + adata.uns["authorType"] + ": " + str(calculate_nmi(adata, final_key, adata.uns["authorType"])))
    
    return adata

def astrid_annotation(adata, output_file, input_prefix, outDir, skip_annotation=False):

    from adpbulk import ADPBulk

    # check if the final_clustering_level is in adata.uns and not None if not then give an error
    if "final_clustering_level" not in adata.uns:
        raise ValueError("final_clustering_level is not in adata.uns. Please run clustering first")
    if adata.uns["final_clustering_level"] is None:
        raise ValueError("final_clustering_level is None. Please run clustering first")
    
    # check if there is a column in adata.obs that matches the authorType in adata.uns and if not give an error
    if adata.uns["authorType"] not in adata.obs.columns:
        raise ValueError("authorType is not in adata.obs. Please check the authorType in adata.uns")
    
    # print message that annotation is running
    print("Running annotation")

    final_key = adata.uns["final_clustering_level"]

    adata.X = adata.layers["raw"]
    adpb = ADPBulk(adata, [final_key])
    pseudobulk_matrix = adpb.fit_transform()
    pseudobulk_matrix.index = pseudobulk_matrix.index.map(lambda x: x.split(".")[1])
    pseudobulk_matrix.to_csv( os.path.splitext(output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv")

    adata.X = adata.layers["norm_counts"]

    if skip_annotation:
        print("Annotation skipped, cell types are Undecided")

        adata.obs["SingleR_CellType"] = "Undecided"
        adata.obs["SingleR_Pruned_CellType"] = "Undecided"
        adata.obs["SingleR_All_deltanext"] = 0

    else:
        singleR_file = os.path.splitext(output_file)[0] + "_singleR_v2.csv"

        singleR_result = runSingleR(adata_file= os.path.splitext(output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv", output_file=singleR_file, rscript_path="/usr/bin/Rscript")

        singleR_result = singleR_result.rename(columns={'pruned.labels': 'SingleR_Pruned_CellType', 'labels': 'SingleR_CellType', 'delta.next': 'SingleR_All_deltanext'})

        # drop the columns in adata.obs that start with SingleR if they exist
        adata.obs = adata.obs.drop(columns=[col for col in adata.obs.columns if col.startswith('SingleR')])

        # merge only the columns SingleR_All_labels	SingleR_All_delta.next	SingleR_All_CellType from singleR_annotations to adata.obs on the index
        adata.obs = adata.obs.merge(singleR_result[['SingleR_Pruned_CellType', 'SingleR_All_deltanext', 'SingleR_CellType']], how="left", left_on=final_key, right_index=True)

        plot_confusion_matrix(adata, "SingleR_CellType", adata.uns["authorType"], outDir + input_prefix + "_" +  "SingleR_CellType_vs_" + adata.uns["authorType"]+ "_confusion_matrix.pdf")

    adata.write_h5ad(output_file)

    # print that annotation is done and also print some other information such as number of SingleR_CellType found and also the normalized mutual information between the author cell type column and SingleR_CellType using norm_mi from RefSTRATools
    print("Annotation done")
    print("Number of SingleR_CellType found: " + str(adata.obs["SingleR_CellType"].nunique()))
    print("Normalized mutual information between SingleR_CellType and " + adata.uns["authorType"] + ": " + str(calculate_nmi(adata, "SingleR_CellType", adata.uns["authorType"])))
    
    return adata

def astrid_validation(adata, pseudobulk_matrix, input_prefix, outDir, output_clustering_results, skip_annotation=False):

    from scipy.cluster import hierarchy
    from scipy.spatial import distance
    from itertools import groupby
    from colorir import Palette, StackPalette, Grad
    from multiprocessing import Pool

    # check if the final_clustering_level is in adata.uns and not None if not then give an error
    if "final_clustering_level" not in adata.uns:
        raise ValueError("final_clustering_level is not in adata.uns. Please run clustering first")
    if adata.uns["final_clustering_level"] is None:
        raise ValueError("final_clustering_level is None. Please run clustering first")
    
    final_key = adata.uns["final_clustering_level"]

    # check if the adata.obs has columns SingleR_CellType and SingleR_Pruned_CellType if not then give an error
    if "SingleR_CellType" not in adata.obs.columns:
        raise ValueError("SingleR_CellType is not in adata.obs. Please run annotation first")
    if "SingleR_Pruned_CellType" not in adata.obs.columns:
        raise ValueError("SingleR_Pruned_CellType is not in adata.obs. Please run annotation first")

    # check if there is a column in adata.obs that matches the authorType in adata.uns and if not give an error
    if adata.uns["authorType"] not in adata.obs.columns:
        raise ValueError("authorType is not in adata.obs. Please check the authorType in adata.uns")
    
    # print message that validation is running
    print("Running validation")

    marker_genes = pd.read_csv('data/ExpectedCellTypesMarkers.csv')

    marker_genes_list = []
    for i in range(len(marker_genes)):
        if type(marker_genes.loc[i, "marker_genes"]) == str:
            marker_genes_list += marker_genes.loc[i, "marker_genes"].split(';')
    marker_genes_list = list(set(marker_genes_list))

    marker_genes_list = np.sort(marker_genes_list).tolist()

    marker_genes_list = [gene for gene in marker_genes_list if gene in adata.var_names]

    # create a named dictionary of lists of marker genes for each cell type
    marker_genes_dict = dict()
    for i in range(len(marker_genes)):
        if type(marker_genes.loc[i, "marker_genes"]) == str:
            marker_genes_dict[marker_genes.loc[i, "cell_type"]] = marker_genes.loc[i, "marker_genes"].split(';')

    temp_final_key = final_key
    final_key = "SingleR_CellType"

    # adata.obs[final_key] = adata.obs[final_key].astype('category')
    adata.obs[final_key] = adata.obs[final_key].astype('str')
    if adata.n_obs > 1000:
        sub_adata = sc.pp.subsample(adata, n_obs=1000, random_state=1, copy=True)
    else:
        sub_adata = adata.copy()

    all_counts = pd.DataFrame(columns=marker_genes_list + [final_key])

    for cl2 in np.sort(sub_adata.obs[final_key].unique()):
        # print(cl2)
        tmp_counts = sc.get.obs_df(sub_adata[sub_adata.obs[final_key] == cl2], keys=marker_genes_list, layer="norm_counts")
        tmp_counts.fillna(0, inplace=True)

        if(tmp_counts.shape[0] > 1):
            row_linkage = hierarchy.linkage(distance.pdist(tmp_counts))
            tmp_counts = tmp_counts.iloc[hierarchy.dendrogram(row_linkage,no_plot=True)['leaves'],:]
        tmp_counts[final_key] = cl2
        
        all_counts = pd.concat([all_counts, tmp_counts], axis=0)

    cs = Palette.load()
    pal = StackPalette.load("spectral")  # Load a categorical palette, a full list can be found in the docs
    grad = Grad(pal)  # Object to automatically "mix" the colors

    # Now to generate a dynamic list of colors based on the number of inputs:
    # Get a bigger list by interpolating the colors if necessary
    if len(all_counts[final_key].unique()) > len(pal):
        colors = grad.n_colors(len(all_counts[final_key].unique())) 
    else:
        colors = pal.colors

    lut = dict(zip(all_counts[final_key].unique(), colors))

    row_colors = all_counts[final_key].map(lut)

    cm = sns.clustermap(all_counts.drop(columns=[final_key]), cmap="Blues", yticklabels=False, col_cluster=False, row_cluster=False, row_colors=row_colors, figsize=(10, 10))
    cm.cax.set_visible(False)
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)
    borders = np.cumsum([0] + [sum(1 for i in cm) for k, cm in groupby(row_colors)])
    for b0, b1, label in zip(borders[:-1], borders[1:], list(lut.keys())):
        cm.ax_row_colors.text(-0.06, (b0 + b1) / 2, label, color='black', ha='right', va='center', rotation=0,
                                transform=cm.ax_row_colors.get_yaxis_transform())
    groups = all_counts[final_key]
    for i, group in enumerate(groups):
        if i > 0 and group.split("_")[0] != groups[i - 1].split("_")[0]:
            cm.ax_heatmap.axhline(i, c="w", linewidth=6)
            cm.ax_row_colors.axhline(i, c="w", linewidth=6)
        elif i > 0 and group != groups[i - 1]:
            cm.ax_heatmap.axhline(i, c="w", linewidth=3)
            cm.ax_row_colors.axhline(i, c="w", linewidth=3)

    plt.savefig(outDir + input_prefix + "_marker_genes_clustermap.pdf", bbox_inches='tight')

    final_key = temp_final_key

    del sub_adata, all_counts

    # for each of the clusters in clustering_level_2 of adata find the authorType abundance inside the clusters and report in a dataframe column seperated by commas with the cell_type(percentage%) format and only report cell types with abundnace more than 10%
    tableInterest = adata.obs.groupby([final_key, 'SingleR_CellType', adata.uns["authorType"]]).size().reset_index(name='counts')
    tableInterest["paperCellTypeAbundance"] = tableInterest.groupby([final_key])['counts'].apply(lambda x: x/x.sum()*100).values
    tableInterest = tableInterest[tableInterest["paperCellTypeAbundance"] > 10]

    # sort the values by clustering_level_2 and cell_type_abundance
    tableInterest = tableInterest.sort_values(by=[final_key, 'paperCellTypeAbundance'], ascending=False)
    tableInterest["paperCellTypeAbundance"] = tableInterest[adata.uns["authorType"]].astype(str) + "(" + tableInterest["paperCellTypeAbundance"].round(2).astype(str) + "%)"
    tableInterest = tableInterest.groupby([final_key])['paperCellTypeAbundance'].apply(lambda x: ','.join(x)).reset_index(name='paperCellTypeAbundance')

    # get the unique clustering_level_2 and SingleR_CellType pairs from th anndata object and merge that with the tableInterest dataframe
    uniqueClusters = adata.obs[[final_key, "SingleR_CellType", "SingleR_Pruned_CellType"]].drop_duplicates()
    
    cellCount = adata.obs.groupby([final_key]).size().reset_index(name='CellCount')
    uniqueClusters = pd.merge(uniqueClusters, cellCount, on=final_key, how="left")

    totalCount = adata.obs.groupby([final_key])["total_counts"].sum().reset_index(name='TotalDepth')
    uniqueClusters = pd.merge(uniqueClusters, totalCount, on=final_key, how="left")
    
    medianCount = adata.obs.groupby([final_key])["total_counts"].median().reset_index(name='MedianDepth')
    uniqueClusters = pd.merge(uniqueClusters, medianCount, on=final_key, how="left")

    if "n_genes" not in adata.obs.columns:
        adata.obs["n_genes"] = (adata.X > 0).sum(axis=1)

    medianFeatures = adata.obs.groupby([final_key])["n_genes"].median().reset_index(name='MedianGenes')
    uniqueClusters = pd.merge(uniqueClusters, medianFeatures, on=final_key, how="left")

    if "pct_counts_mt" in adata.obs.columns:

        meanMT = adata.obs.groupby([final_key])["pct_counts_mt"].mean().reset_index(name='MeanMT')
        uniqueClusters = pd.merge(uniqueClusters, meanMT, on=final_key, how="left")

    tableInterest = pd.merge(uniqueClusters, tableInterest, on=final_key, how="left")
    tableInterest = tableInterest.drop_duplicates()

    # check if there are columns in adata.obs that are is_cancer and is_unassigned and if not give an error
    if "is_cancer" not in adata.obs.columns:
        adata.obs["is_cancer"] = adata.obs[adata.uns["authorType"]].str.contains('Cancer', case=False)
    if "is_unassigned" not in adata.obs.columns:
        adata.obs["is_unassigned"] = adata.obs[adata.uns["authorType"]].isin(["unassigned", "nan", "Unassigned"])
    
    # for each cluster in the final_key column of adata.obs, calculate the percentage of cells that have is_cancer == True and create a dataframe with the cluster names and percentages and merge with tableInterest
    cancer_percentages = adata.obs.groupby(final_key)["is_cancer"].apply(lambda x: (x == True).sum() / len(x)).reset_index()
    cancer_percentages.columns = [final_key, "cancer_percentage"]
    tableInterest = tableInterest.merge(cancer_percentages, on=final_key)

    # calculate the cancer percentages of each cluster without the unassigned cells and merge with tableInterest
    cancer_percentages = adata.obs[~adata.obs["is_unassigned"]].groupby(final_key)["is_cancer"].apply(lambda x: (x == True).sum() / len(x)).reset_index()
    cancer_percentages.columns = [final_key, "cancer_percentage_no_unassigned"]
    tableInterest = tableInterest.merge(cancer_percentages, on=final_key)

    # for each cluster in the final_key column of adata.obs, calculate the percentage of cells that have is_unassigned == True and create a dataframe with the cluster names and percentages and merge with tableInterest
    unassigned_percentages = adata.obs.groupby(final_key)["is_unassigned"].apply(lambda x: (x == True).sum() / len(x)).reset_index()
    unassigned_percentages.columns = [final_key, "unassigned_percentage"]
    tableInterest = tableInterest.merge(unassigned_percentages, on=final_key)

    clusters = adata.obs[final_key].astype(str).unique()

    # tableInterest.to_csv(outDir + input_prefix + "_tableInterest_df.csv", index=False)

    with Pool(processes=10) as pool:
        results = pool.map(calculate_odds_ratios, zip(clusters, [pseudobulk_matrix] * len(clusters), [tableInterest] * len(clusters), ["SingleR_CellType"] * len(clusters), [skip_annotation] * len(clusters) ))

    # odds_ratio_df = [item for sublist in results for item in sublist]
    odds_ratio_df = np.vstack(results)
    odds_ratio_df = pd.DataFrame(odds_ratio_df, columns=[final_key, "gene", "odds_ratio", "fold_change"])

    # odds_ratio_df.to_csv(outDir + input_prefix + "_odds_ratio_df.csv", index=False)

    dict_data = json.load(open("data/CellTypeAliases.json"))

    # create a boolean variable to store if all the cell types in tableInterest SingleR_CellType column are in dict_data["immune"]
    allImmune = all([cell_type in dict_data["immune"] for cell_type in tableInterest["SingleR_CellType"].unique()])
 
    tableInterest["FormulaPassed"] = False
    tableInterest["OtherCellTypesPassed"] = ""
    tableInterest["OddsRatioMarkerGenes"] = ""
    tableInterest["TopOddsRatioGenes"] = ""

    unique_cell_types = marker_genes["cell_type"].unique()

    for cl2 in tableInterest[final_key].unique():
        tmpOR = odds_ratio_df[odds_ratio_df[final_key] == cl2]
        tmpOR_dict = dict(zip(tmpOR["gene"], (tmpOR["odds_ratio"] > 1) & (tmpOR["fold_change"] > 1)))

        # if all the cell types in tableInterest SingleR_CellType columns are in dict_data["immune"], then set the PTPRC to True in tmpOR_dict and tmpOR
        if allImmune:
            tmpOR_dict["PTPRC"] = True
            
        cell_type = tableInterest.loc[tableInterest[final_key] == cl2, "SingleR_CellType"].values[0]
        if cell_type in unique_cell_types:
            formula = marker_genes.loc[marker_genes["cell_type"] == cell_type, "expected_markers"].values[0]
            tableInterest.loc[tableInterest[final_key] == cl2, "FormulaPassed"] = evaluate_formula(formula, tmpOR_dict) if formula == formula else "not applicable"
        else:
            tableInterest.loc[tableInterest[final_key] == cl2, "FormulaPassed"] = "not applicable"

        formulas = marker_genes.loc[marker_genes["cell_type"].isin(unique_cell_types), "expected_markers"]
        other_passed = unique_cell_types[formulas.apply(lambda x: evaluate_formula(x, tmpOR_dict) if x == x else False)]
        tableInterest.loc[tableInterest[final_key] == cl2, "OtherCellTypesPassed"] = ";".join(other_passed)

        tmp_df = tmpOR[tmpOR.gene.isin(marker_genes_list) & (tmpOR.fold_change > 1) & (tmpOR.odds_ratio > 1)].sort_values(by = "odds_ratio", ascending = False)
        if allImmune and "PTPRC" not in tmp_df["gene"].values:
            tmp_df = pd.concat([tmp_df, pd.DataFrame({"gene": ["PTPRC"], "odds_ratio": [1.1], "fold_change": [1.1]})])
        tableInterest.loc[tableInterest[final_key] == cl2, "OddsRatioMarkerGenes"] = ";".join(sorted(tmp_df["gene"])) if len(tmp_df) > 0 else ""

        tmp_df = tmpOR[(tmpOR.fold_change > 1) & (tmpOR.odds_ratio > 1)].sort_values(by = "odds_ratio", ascending = False)
        top_genes = tmp_df["gene"].head(n=5).str.cat(sep = ";") if len(tmp_df) > 0 else ""
        tableInterest.loc[tableInterest[final_key] == cl2, "TopOddsRatioGenes"] = top_genes

    tableInterest["OtherCellTypesPassed"] = tableInterest["OtherCellTypesPassed"].fillna("")
    tableInterest["CountCellTypesPassed"] = tableInterest["OtherCellTypesPassed"].apply(lambda x: len(x.split(";")))
    tableInterest.loc[tableInterest["OtherCellTypesPassed"] == "", "CountCellTypesPassed"] = 0

    # sort tableInterest by clutsering_level_2 labels
    tableInterest = tableInterest.sort_values(by=[final_key])
    tableInterest.to_csv(output_clustering_results, index=False)

    # print that validation is done and also print some other information such as number of clusters that have passed teh formula out of the total and that were applicable
    print("Validation done")
    print("Number of clusters that passed the formula out of the applicable/total: " + str(len(tableInterest[tableInterest["FormulaPassed"] == True])) + "/" + str(len(tableInterest[tableInterest["FormulaPassed"] != "not applicable"])) + "/" + str(len(tableInterest)))
    # print the number of cells that have passed the formula out of the total and that were applicable, base this on the weighted average of the cell count in each cluster
    print("Number of cells that passed the formula out of the applicable/total: " + str(tableInterest[tableInterest["FormulaPassed"] == True]["CellCount"].sum()) + "/" + str(tableInterest[tableInterest["FormulaPassed"] != "not applicable"]["CellCount"].sum()) + "/" + str(tableInterest["CellCount"].sum()))
    print("Percentage of cells that passed the formula out of the applicable/total: {:.2f}/{:.2f}".format(tableInterest[tableInterest["FormulaPassed"] == True]["CellCount"].sum() / tableInterest[tableInterest["FormulaPassed"] != "not applicable"]["CellCount"].sum() * 100, tableInterest[tableInterest["FormulaPassed"] == True]["CellCount"].sum() / tableInterest["CellCount"].sum() * 100))

    return tableInterest

def astrid_damage(adata, tableInterest, pseudobulk_matrix, output_clustering_results, output_file):

    from multiprocessing import Pool

    # check if the final_clustering_level is in adata.uns and not None if not then give an error
    if "final_clustering_level" not in adata.uns:
        raise ValueError("final_clustering_level is not in adata.uns. Please run clustering first")
    if adata.uns["final_clustering_level"] is None:
        raise ValueError("final_clustering_level is None. Please run clustering first")
    
    final_key = adata.uns["final_clustering_level"]

    # check if the adata.obs has columns SingleR_CellType and SingleR_Pruned_CellType if not then give an error
    if "SingleR_CellType" not in adata.obs.columns:
        raise ValueError("SingleR_CellType is not in adata.obs. Please run annotation first")
    if "SingleR_Pruned_CellType" not in adata.obs.columns:
        raise ValueError("SingleR_Pruned_CellType is not in adata.obs. Please run annotation first")

    # check if there is a column in adata.obs that matches the authorType in adata.uns and if not give an error
    if adata.uns["authorType"] not in adata.obs.columns:
        raise ValueError("authorType is not in adata.obs. Please check the authorType in adata.uns")
    
    # print message that validation is running
    print("Running cancer cell classification")

    # infer the chromosome damage
    referenceCounts = pd.read_csv("data/reference_REFSTRA_SingleR_NormCounts.csv", index_col=0, sep=",")
    referenceMetadata = pd.read_csv("data/reference_REFSTRA_SingleR_NormCounts_Metadata.csv", index_col=0, sep=",")

    cnvFC, cnvLoc = calculate_chr_damage(adata, referenceCounts)
    adata.obsm["X_cnv"] = cnvFC.loc[adata.obs_names,:].to_numpy()
    adata.uns["cnv"] = cnvLoc

    del cnvFC, cnvLoc

    # calculate the average CNV profile for each cluster and correlate it with the CNV profile of each cell and create a new column in adata.obs that contains the correlation values
    adata.obs["newCNVScoreABS"] = np.abs(adata.obsm["X_cnv"]).sum(axis=1) /adata.obsm["X_cnv"].shape[1]
    adata.obs["newCNVScoreABS"] = adata.obs.groupby(final_key)["newCNVScoreABS"].transform("mean")

    adata.obs["newCNVScoreSQR"] = np.square(adata.obsm["X_cnv"]).sum(axis=1) / adata.obsm["X_cnv"].shape[1]
    adata.obs["newCNVScoreSQR"] = adata.obs.groupby(final_key)["newCNVScoreSQR"].transform("mean")

    # Calculate the number of rows that make up 5% of the DataFrame
    top5pct_rows = adata.obs.nlargest(int(adata.n_obs * 0.05)  , 'newCNVScoreABS')
    # Get the row names (index) of the top 5% rows
    top5pct_rownames = top5pct_rows.index
    average_CNV_profile = adata[top5pct_rownames, :].obsm["X_cnv"].mean(axis=0)

    # correlate the average CNV profile with the CNV profile of each cell and create a new column in adata.obs that contains the correlation values
    adata.obs["CNVCorrelation"] = np.corrcoef(average_CNV_profile, adata.obsm["X_cnv"])[0][1:]

    if "newCNVScoreABS" in tableInterest.columns:
        del tableInterest["newCNVScoreABS"]

    if "newCNVScoreSQR" in tableInterest.columns:
        del tableInterest["newCNVScoreSQR"]

    if "CNVCorrelation" in tableInterest.columns:
        del tableInterest["CNVCorrelation"]

    tableInterest = tableInterest.merge(adata.obs.groupby(final_key)[["newCNVScoreABS", 'newCNVScoreSQR', 'CNVCorrelation']].mean().reset_index(), on=final_key) 

    common_genes = pseudobulk_matrix.columns[pseudobulk_matrix.columns.isin(referenceCounts.index)]
    pseudobulk_matrix = pseudobulk_matrix.loc[:,common_genes]

    oncoGeneReference = referenceCounts.T
    oncoGeneMetadata = referenceMetadata[["cellTypeGRINT"]]
    # group the rows of the same cell type in OncogeneReference and sum them to get the total counts for each gene in each cell type
    oncoGeneReference = oncoGeneMetadata.merge(oncoGeneReference, left_index=True, right_index=True)

    final_key = adata.uns['final_clustering_level']

    clusters = tableInterest[final_key].astype(str).unique()

    with Pool(processes=10) as pool:
        results = pool.map(calculate_odds_ratios, zip(clusters, [pseudobulk_matrix] * len(clusters), [oncoGeneReference] * len(clusters),  [tableInterest] * len(clusters), ["SingleR_CellType"] * len(clusters), ["cellTypeGRINT"] * len(clusters) ))

    # odds_ratio_df = [item for sublist in results for item in sublist]
    odds_ratio_df = np.vstack(results)
    odds_ratio_df = pd.DataFrame(odds_ratio_df, columns=[final_key, "gene", "odds_ratio", "fold_change"])

    # oncogenes = pd.read_table("data/IntOGen-DriverGenes_BRCA.tsv")
    oncogenes = pd.read_table("data/IntOGen-DriverGenes_CSCC.tsv")
    oncogenes = oncogenes[oncogenes.Symbol.isin(adata.var_names)] 
    oncogenes.rename(columns={'Samples (%)': 'SamplesPercentage'}, inplace=True)

    odds_ratio_df = odds_ratio_df[odds_ratio_df["gene"].isin(oncogenes.Symbol)]

    if "odds_ratio_gt_1" in tableInterest.columns:
        del tableInterest["odds_ratio_gt_1"]

    if "odds_ratio_spread" in tableInterest.columns:
        del tableInterest["odds_ratio_spread"]

    # each cluster in the final_key column of odds_ratio_df, calcuate the number of genes with odds_ratio > 1 and create a dataframe with the clusters and number of genes gretaer than 1 and merge it with tableInterest
    # tableInterest = odds_ratio_df.groupby(final_key).apply(lambda x: pd.Series({"odds_ratio_gt_1": (x["odds_ratio"] > 1).sum()})).reset_index().merge(tableInterest, on=final_key)

    odds_ratio_summary = odds_ratio_df.groupby(final_key).apply(
        lambda x:
        pd.Series({"odds_ratio_gt_1": (x["odds_ratio"] > 1).sum(),
                   "odds_ratio_spread": (x["odds_ratio"].quantile(0.95) - x["odds_ratio"].quantile(0.05))
                   })).reset_index()

    tableInterest = tableInterest.merge(odds_ratio_summary, on=final_key)

    # sort tableInterest by clutsering_level_2 labels
    tableInterest = tableInterest.sort_values(by=[final_key])
    tableInterest.to_csv(output_clustering_results, index=False)
    adata.write_h5ad(output_file)

    # print that clustering is done and also print the final clustering level and number of clusters identified and some other information
    print("Cancer classification done")

    return
    

def main():
    
    import argparse

    parser = argparse.ArgumentParser(description='Run ASTRID (Automatized Single-cell Typing for tumoR transcrIptomics Data) ᛅᛋᛏᚱᛁᛏ pipeline')
    parser.add_argument('--all', action='store_true', help='Run all tasks')
    parser.add_argument('--clustering', action='store_true', help='Run clustering')
    parser.add_argument('--annotation', action='store_true', help='Run annotation')
    parser.add_argument('--validation', action='store_true', help='Run validation')
    parser.add_argument('--damage', action='store_true', help='Run cancer damage')
    parser.add_argument('--skip_cell_typing', action='store_true', help='Skip cell typing and assign "Undecided" to all cells')
    parser.add_argument('--cutoff_level', type=int, default=None, help='Cut off clustering early at the specified level')
    parser.add_argument('--input_file', type=str, help='Input file path')
    parser.add_argument('--input_prefix', type=str, help='Input prefix')
    parser.add_argument('--output_file', type=str, help='Output file path')
    parser.add_argument('--output_clustering_results', type=str, help='Output clustering results path')
    parser.add_argument('--final_key', type=str, help='Key for final level of clustering (column in AnnData.obs)')
    parser.add_argument('--author_type', type=str, help='Author cell type column name')
    parser.add_argument('--out_dir', type=str, help='Output directory for plots')

    args = parser.parse_args()
    
    outDir = os.getcwd()

    if args.all:
        if not args.input_file or not args.input_prefix or not args.output_file or not args.output_clustering_results:
            raise ValueError("Missing required arguments for running all tasks")

        # print message saying that the all option is selected and running all tasks
        print("Running all tasks for the sample " + args.input_prefix)

        if args.out_dir:
            outDir = args.out_dir + "/" + args.input_prefix + "/"
        else:
            outDir = outDir + args.input_prefix + "/"
        
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        print("Outputs written into " + outDir)

        adata = sc.read_h5ad(args.input_file)

        if args.author_type:
            adata.uns["authorType"] = args.author_type

        # check if the adata has authorType inside adata.uns and if not set it to cellTypeMinor
        if "authorType" not in adata.uns:
            adata.uns["authorType"] = "cellTypeMinor"

        start_time = time.time()
        adata = astrid_clustering(adata, args.input_prefix, outDir)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Clustering took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
        
        start_time = time.time()
        adata = astrid_annotation(adata, args.output_file, args.input_prefix, outDir)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Annotation took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
        
        final_key = adata.uns["final_clustering_level"]
        pseudobulk_matrix = pd.read_csv(os.path.splitext(args.output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv", index_col=0)

        start_time = time.time()
        tableInterest = astrid_validation(adata, pseudobulk_matrix, args.input_prefix, outDir, args.output_clustering_results)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Validation took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

        start_time = time.time()
        astrid_damage(adata, tableInterest, pseudobulk_matrix, args.output_clustering_results, args.output_file)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Cancer damage estimation took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

        # Print the output directory and its contents in abbreviated format
        print("Outputs:")
        print(f"Directory: {outDir}")

        # List the files in the output directory
        files = os.listdir(outDir)

        # Abbreviate and print the file names
        for file in files:
            abbreviated_file = textwrap.shorten(file, width=50, placeholder="...")
            print(f"  {abbreviated_file}")
        
    else:
        if args.clustering:
            if not args.input_file or not args.input_prefix or not args.output_clustering_results:
                raise ValueError("Missing required arguments for running clustering")

            if args.out_dir:
                outDir = args.out_dir + "/" + args.input_prefix + "/"
            else:
                outDir = outDir + args.input_prefix + "/"

            if not os.path.exists(outDir):
                os.makedirs(outDir)

            print("Outputs written into " + outDir + "for clustering.")

            # print that clusterinf option is selected
            print("Clustering started for the sample " + args.input_prefix )

            adata = sc.read_h5ad(args.input_file)

            if args.author_type:
                adata.uns["authorType"] = args.author_type

            # check if the adata has authorType inside adata.uns and if not set it to cellTypeMinor
            if "authorType" not in adata.uns:
                adata.uns["authorType"] = "cellTypeMinor"

            start_time = time.time()
            adata = astrid_clustering(adata, args.input_prefix, outDir, cutoff_level=args.cutoff_level)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Clustering took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")
            print("Outputs ")

        if args.annotation:
            if not args.input_file or not args.output_file:
                raise ValueError("Missing required arguments for running annotation")

            if args.out_dir:
                outDir = args.out_dir + "/" + args.input_prefix + "/"
            else:
                outDir = outDir + args.input_prefix + "/"

            if not os.path.exists(outDir):
                os.makedirs(outDir)

            print("Outputs written into " + outDir + "for annotation.")

            #if clustering is not run then read the adata object from the input file else use the adata object from clustering
            if not args.clustering:
                adata = sc.read_h5ad(args.output_file)

            if args.author_type:
                adata.uns["authorType"] = args.author_type

            # check if the adata has authorType inside adata.uns and if not set it to cellTypeMinor
            if "authorType" not in adata.uns:
                adata.uns["authorType"] = "cellTypeMinor"
            
            print("Annotation started for the sample " + args.input_prefix)

            start_time = time.time()
            adata = astrid_annotation(adata, args.output_file, args.input_prefix, outDir, args.skip_cell_typing)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Annotation took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")


        if args.validation:
            if not args.input_file or not args.output_file or not args.output_clustering_results or not args.input_prefix:
                raise ValueError("Missing required arguments for running validation")
            
            if args.out_dir:
                outDir = args.out_dir + "/" + args.input_prefix + "/"
            else:
                outDir = outDir + args.input_prefix + "/"

            if not os.path.exists(outDir):
                os.makedirs(outDir)

            print("Outputs written into " + outDir + "for validation.")

            #if annotation is not run then read the adata object from the output file else use the adata object from annotation
            if not args.annotation:
                adata = sc.read_h5ad(args.output_file)
            
            if args.author_type:
                adata.uns["authorType"] = args.author_type

            # check if the adata has authorType inside adata.uns and if not set it to cellTypeMinor
            if "authorType" not in adata.uns:
                adata.uns["authorType"] = "cellTypeMinor"

            print("Validation started for the sample " + args.input_prefix)
            
            final_key = adata.uns["final_clustering_level"]
                        
            print(final_key)
            
            # check if the pseudobulk_matrix file exists and if exists then read it else give an error
            if not os.path.exists(os.path.splitext(args.output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv"):
                raise ValueError("Pseudobulk matrix file " + os.path.splitext(args.output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv" + " does not exist")

            pseudobulk_matrix = pd.read_csv(os.path.splitext(args.output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv", index_col=0)
            pseudobulk_matrix.index = pseudobulk_matrix.index.astype(str)
            
            start_time = time.time()
            tableInterest = astrid_validation(adata, pseudobulk_matrix, args.input_prefix, outDir, args.output_clustering_results, args.skip_cell_typing)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Validation took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

        if args.damage:
            if not args.input_file or not args.output_file or not args.output_clustering_results or not args.input_prefix:
                raise ValueError("Missing required arguments for running cancer damage")
            
            if args.out_dir:
                outDir = args.out_dir + "/" + args.input_prefix + "/"
            else:
                outDir = outDir + args.input_prefix + "/"

            if not os.path.exists(outDir):
                os.makedirs(outDir)

            #if validation is not run then read the adata object from the output file else use the adata object from validation
            if not args.validation:
                adata = sc.read_h5ad(args.output_file)
                tableInterest = pd.read_csv(args.output_clustering_results)
            
            if args.author_type:
                adata.uns["authorType"] = args.author_type

            # check if the adata has authorType inside adata.uns and if not set it to cellTypeMinor
            if "authorType" not in adata.uns:
                adata.uns["authorType"] = "cellTypeMinor"

            print("Cancer damage estimation started for the sample " + args.input_prefix)
            
            final_key = adata.uns["final_clustering_level"]
                        
            print(final_key)
            
            # check if the pseudobulk_matrix file exists and if exists then read it else give an error
            if not os.path.exists(os.path.splitext(args.output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv"):
                raise ValueError("Pseudobulk matrix file " + os.path.splitext(args.output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv" + " does not exist")

            pseudobulk_matrix = pd.read_csv(os.path.splitext(args.output_file)[0] + "_" + final_key + "_pseudobulk_matrix.csv", index_col=0)
            
            start_time = time.time()
            astrid_damage(adata, tableInterest, pseudobulk_matrix, args.output_clustering_results, args.output_file)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Cancer damage estimation took {int(elapsed_time // 60)} minutes and {elapsed_time % 60:.2f} seconds")

        # Print the output directory and its contents in abbreviated format
        print("Outputs:")
        print(f"Directory: {outDir}")

        # List the files in the output directory
        files = os.listdir(outDir)

        # Abbreviate and print the file names
        for file in files:
            abbreviated_file = textwrap.shorten(file, width=50, placeholder="...")
            print(f"  {abbreviated_file}")

        # if no task is selected then give a warning, show the help message and exit
        if not args.clustering and not args.annotation and not args.validation and not args.all and not args.damage:
            print("No task selected. Please select a task using --clustering, --annotation, --validation, --damage or --all")
            parser.print_help()
            sys.exit(1)        
    return

if __name__ == "__main__":
    main()

