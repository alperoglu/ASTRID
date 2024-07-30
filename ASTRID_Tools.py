import os
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import pairwise_distances
from scipy.sparse import coo_matrix
import leidenalg
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import betaln
import re
import warnings
import json

warnings.filterwarnings("ignore")

def evaluate_formula(formula, x):
    """
    Evaluates a logical formula based on the provided dictionary.

    Parameters:
    formula (str): The logical formula to evaluate.
    x (dict): A dictionary where keys are variable names and values are their truth values.

    Returns:
    bool: The result of the logical formula.
    """

    if formula == "":
        return True
    if not re.search(r"(x|\(|\)|\+|\!)", formula) and formula not in x.keys():
        return False
    if not re.search(r"(x|\(|\)|\+|\!)", formula):
        return x[formula]
    if not re.search(r"(x|\(|\)|\+)", formula) and re.search(r"!", formula):
        return not evaluate_formula(formula.replace("!", ""), x)
    if not re.search(r"(x|\(|\))", formula) and re.search(r"\+", formula):
        return any([evaluate_formula(f, x) for f in formula.split("+")])
    if re.search(r"(x|\(|\))", formula):
        return all([evaluate_formula(f, x) for f in re.split(r"x|\(|\)", formula)])

def entropy(p):
    """
    Calculates the entropy of a list of probabilities.

    Parameters:
    p (list): A list of probabilities.

    Returns:
    float: The entropy of the list of probabilities.
    """
    tmp = 0
    for i in range(len(p)):
        if p[i] == 0:
           tmp = tmp + 0
        else:
            tmp = tmp + p[i] * np.log(p[i])
    return -tmp

def mutual_info_score(contingency=None):
    """
    Calculates the mutual information score of a contingency table.

    Parameters:
    contingency (numpy.ndarray): A contingency table.

    Returns:
    float: The mutual information score.
    """

    nzx, nzy = np.nonzero(contingency)
    nz_val = contingency[nzx, nzy]

    contingency_sum = contingency.sum()
    pi = np.ravel(contingency.sum(axis=1))
    pj = np.ravel(contingency.sum(axis=0))

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum

    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())

    mi = (
        contingency_nm * (log_contingency_nm - np.log(contingency_sum))
        + contingency_nm * log_outer
    )

    mi = np.where(np.abs(mi) < np.finfo(mi.dtype).eps, 0.0, mi)

    return np.clip(mi.sum(), 0.0, None)

def norm_mi(contingency = None, average_method = 'arithmetic'):
    """
    Calculates the normalized mutual information score of a contingency table.

    Parameters:
    contingency (numpy.ndarray): A contingency table.
    average_method (str): The method to use for averaging the entropies. Options are 'min', 'geometric', 'arithmetic', 'max', 'custom'.

    Returns:
    float: The normalized mutual information score.
    """
    
    mi = mutual_info_score(contingency)
    
    h_1, h_2 = entropy(contingency.sum(axis=1)), entropy(contingency.sum(axis=0))

    if average_method == "min":
        normalizer = min(h_1, h_2)
    elif average_method == "geometric":
        normalizer = np.sqrt(h_1 * h_2)
    elif average_method == "arithmetic":
        normalizer = np.mean([h_1, h_2])
    elif average_method == "max":
        normalizer = max(h_1, h_2)
    elif average_method == "custom":
        normalizer = h_1
    else:
        normalizer = 1

    return mi / normalizer


def calculate_nmi(adata, clustering_key, control_key):
    """
    Calculates the normalized mutual information between two categorical variables from adata.obs.

    Parameters:
    adata (anndata.AnnData): An AnnData object containing the gene expression data and metadata.
    clustering_key (str): The key of the first categorical variable in adata.obs.
    control_key (str): The key of the second categorical variable in adata.obs.

    Returns:
    float: The normalized mutual information between the two categorical variables.
    """
        
    # convert the clustering key to string and control key to category 
    adata.obs[clustering_key] = adata.obs[clustering_key].astype(str)
    adata.obs[control_key] = adata.obs[control_key].astype("category")

    # create the confusion matrix 
    confusion_matrix = pd.crosstab(adata.obs[control_key], adata.obs[clustering_key])

    # normalize the confusion matrix by column so thay they sum up to 1
    confusion_matrix = confusion_matrix.div(confusion_matrix.values.sum())

    nmi = norm_mi(confusion_matrix.values, average_method="custom")

    return nmi

def get_indices_distances_from_dense_matrix(D, k: int):
    """
    Gets the indices and distances of the k nearest neighbors for each point in a dense distance matrix.

    Parameters:
    D (numpy.ndarray): A dense distance matrix.
    k (int): The number of nearest neighbors to find.

    Returns:
    tuple: A tuple containing two numpy.ndarrays. The first array contains the indices of the k nearest neighbors for each point. The second array contains the distances to the k nearest neighbors for each point.
    """

    indices = np.argpartition(D, k, axis=1)[:, :k]
    distances = np.take_along_axis(D, indices, axis=1)
    return indices, distances

def clustering_modularity_custom(adata, distance_key, k: int, resolution, key = "leiden_modularity_"):
    """
    Performs Leiden clustering on an AnnData object using a custom modularity function.

    Parameters:
    adata (anndata.AnnData): An AnnData object containing the gene expression data and metadata.
    distance_key (str): The key of the distance matrix in adata.obsp.
    k (int): The number of nearest neighbors to use for the clustering.
    resolution (float): The resolution to use for the clustering. Higher values result in more clusters.
    key (str, optional): The key to use for storing the clustering assignments in the AnnData object's obs attribute. Defaults to "leiden_modularity_".

    Returns:
    anndata.AnnData: The AnnData object with the clustering assignments added.
    """
    
    indices, distances = get_indices_distances_from_dense_matrix(adata.obsp[distance_key], k=k)

    adjacency_matrix = coo_matrix((distances.ravel(), (np.repeat(indices[:, 0], k), indices.ravel())), shape=(adata.obsp[distance_key].shape[0], adata.obsp[distance_key].shape[0]))

    sc.tl.leiden(adata, 
            adjacency=adjacency_matrix.tocsr(),
                        resolution=resolution, 
                        key_added=key, 
                        use_weights=False,
                        partition_type = leidenalg.RBConfigurationVertexPartition,
                        # initial_membership = np.zeros(adata.n_obs, dtype=int).tolist(),
                        n_iterations=-1,

    )

    # for the custom clusters set the assignments to -1 if the cluster size is equal to 1
    cluster_sizes = adata.obs[key].value_counts()

    adata.obs[key] = adata.obs[key].astype(str)
    clusters_to_replace = cluster_sizes[cluster_sizes == 1].index
    adata.obs.loc[adata.obs[key].isin(clusters_to_replace), key] = "unassigned"
    # adata.obs[key] = adata.obs[key].astype('category')

    return adata

def normalize_data(adata):
    """
    Normalizes the gene expression data in an AnnData object.

    Parameters:
    adata (anndata.AnnData): An AnnData object containing the gene expression data and metadata.

    Returns:
    anndata.AnnData: The AnnData object with the normalized gene expression data.
    """

    if "raw" not in adata.layers.keys():
        adata.layers["raw"] = adata.X.copy()
    adata.X = adata.layers["raw"]
    total_counts = np.ravel(adata.X.sum(axis=1))
    median_total_counts = np.median(total_counts)
    scaling_factor = median_total_counts / total_counts

    adata.layers['norm_counts'] = adata.X.toarray() * scaling_factor[:, np.newaxis]

    adata.layers['norm_counts'] = np.log1p(adata.layers["norm_counts"])
    adata.X = adata.layers["norm_counts"]
    return adata

def clustering_recipe_custom(adata, k: int, resolution, do_highly_variable = False, clustering_key = "clustering_level_"):
    """
    This function performs a custom clustering recipe on the given AnnData object.
    It normalizes the data, selects highly variable genes, performs PCA, and then performs clustering.

    Parameters:
    adata (anndata.AnnData): An AnnData object containing the gene expression data and metadata.
    k (int): The number of neighbors to use for the clustering.
    resolution (float): The resolution to use for the clustering. Higher values result in more clusters.
    do_highly_variable (bool, optional): Whether to select highly variable genes before performing PCA. Defaults to False.
    clustering_key (str, optional): The key to use for storing the clustering assignments in the AnnData object's obs attribute. Defaults to "clustering_level_".

    Returns:
    anndata.AnnData: The AnnData object with the clustering assignments added.
    """
    
    adata = normalize_data(adata)
    ndims_measured = max(2, np.floor(adata.n_obs/100).astype(int))

    if do_highly_variable:
        bin_count = max(20, int(np.floor(adata.n_vars/1000)))

        sc.pp.highly_variable_genes(adata, flavor="seurat", min_disp=0.5, max_mean=np.inf, min_mean = np.log(2), n_bins = bin_count, layer = "norm_counts")

        num_highly_variable_genes = adata.var['highly_variable'].sum()

        if num_highly_variable_genes < 1000:
            sc.pp.highly_variable_genes(adata, flavor="seurat", n_bins = bin_count, layer = "norm_counts", n_top_genes=1000)
        
        if ndims_measured > adata.var_names[adata.var["highly_variable"]].shape[0]:
            ndims_measured = adata.var_names[adata.var["highly_variable"]].shape[0] - 1

    sc.tl.pca(adata, svd_solver='arpack', n_comps=ndims_measured, use_highly_variable=do_highly_variable)

    adata.obsm['X_pca_norm_counts'] = adata.obsm['X_pca']
    adata.varm['PCs_norm_counts'] = adata.varm['PCs']
    adata.uns['pca_norm_counts'] = adata.uns['pca']

    adata.obsp['pcCustom_norm_counts_distances'] = pairwise_distances(adata.obsm['X_pca_norm_counts'][:,0:ndims_measured])

    adata = clustering_modularity_custom(adata, "pcCustom_norm_counts_distances", k = k, resolution = resolution, key= clustering_key)

    return adata

def runSingleR(adata_file, output_file, rscript_path = '/scratch/alper.eroglu/miniconda3/envs/r4/bin/Rscript'):

    print('Running SingleR ...')
    os.system('nice -19 ' + rscript_path + ' /scratch/alper.eroglu/tools/ASTRID/RunSingleR.R '+adata_file + ' ' + output_file)
    result=pd.read_csv(output_file)

    print('SingleR completed.')
    
    return result

def plot_confusion_matrix(adata, clustering_key, control_key, output_file):
    """
    This function plots a confusion matrix for the given clustering and control keys in the AnnData object.
    It also plots UMAP visualizations for the clustering and control keys, and a bar plot of the cluster counts.

    Parameters:
    adata (anndata.AnnData): An AnnData object containing the gene expression data and metadata.
    clustering_key (str): The key in the AnnData object's obs attribute that contains the clustering assignments.
    control_key (str): The key in the AnnData object's obs attribute that contains the control assignments.
    output_file (str): The path to the file where the plot should be saved.

    Returns:
    None
    """
    
    # Convert the clustering key to string and control key to category 
    adata.obs[clustering_key] = adata.obs[clustering_key].astype(str)
    adata.obs[control_key] = adata.obs[control_key].astype("category")

    # Create the confusion matrix 
    confusion_matrix = pd.crosstab(adata.obs[control_key], adata.obs[clustering_key], margins=True)
    confusion_matrix = confusion_matrix.drop('All', axis=1)
    confusion_matrix = confusion_matrix.drop('All', axis=0)

    # Normalize the confusion matrix by column so that they sum up to 1
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=0), axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10), gridspec_kw={'wspace':0.5})

    ax1_dict = sc.pl.umap(adata, color=control_key, show=False, ax = ax1)

    ax1_dict = sc.pl.umap(adata, color=clustering_key, show=False, ax = ax2)

    ax2_dict = sns.heatmap(confusion_matrix, fmt='g', cmap='Blues', cbar=False, xticklabels=True, yticklabels=True, ax = ax3)

    confusion_matrix = pd.crosstab(adata.obs[control_key], adata.obs[clustering_key])

    # Normalize the confusion matrix by column so that they sum up to 1
    confusion_matrix = confusion_matrix.div(confusion_matrix.values.sum())

    nmi = norm_mi(confusion_matrix.values, average_method="custom")
    ax2_dict.set_title( 'Normalized Mutual Info Score = {:.2f}'.format(nmi))

    tableText = adata.obs[clustering_key].astype(str).value_counts().to_frame()
    tableText["cluster"] = tableText.index
    tableText = tableText.reset_index(drop=True)

    ax4.bar(tableText["cluster"], tableText["count"])
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90)

    plt.savefig(output_file, bbox_inches='tight')

def odds_ratio_differential(refCounts, otherCounts):
    """
    This function calculates the log odds ratio of the reference counts to the other counts.

    Parameters:
    refCounts (numpy.array): An array containing the reference counts.
    otherCounts (numpy.array): An array containing the other counts.

    Returns:
    float: The log odds ratio of the reference counts to the other counts.
    """
    
    totalRefCounts = np.sum(refCounts)
    totalOtherCounts = np.sum(otherCounts)
    
    term1 = betaln(refCounts + 1, totalRefCounts - refCounts + 1)
    term2 = betaln(otherCounts + 1, totalOtherCounts - otherCounts + 1)
    term3 = betaln(refCounts + otherCounts + 1, totalRefCounts + totalOtherCounts - refCounts - otherCounts + 1)
    
    result = (term1 + term2) - term3
    result = result / np.log(10)
    
    return result

def odds_ratio_differential_efficient(refCounts, totalRefCounts, otherCounts, totalOtherCounts):
    """
    This function calculates the log odds ratio of the reference counts to the other counts.
    Unlike the previous function, this function takes the total counts as parameters, which makes it more efficient if the total counts have already been calculated.

    Parameters:
    refCounts (numpy.array): An array containing the reference counts.
    totalRefCounts (int): The total reference counts.
    otherCounts (numpy.array): An array containing the other counts.
    totalOtherCounts (int): The total other counts.

    Returns:
    float: The log odds ratio of the reference counts to the other counts.
    """
        
    term1 = betaln(refCounts + 1, totalRefCounts - refCounts + 1)
    term2 = betaln(otherCounts + 1, totalOtherCounts - otherCounts + 1)
    term3 = betaln(refCounts + otherCounts + 1, totalRefCounts + totalOtherCounts - refCounts - otherCounts + 1)
    
    result = (term1 + term2) - term3
    result = result / np.log(10)
    
    return result

def process_cluster(params):
    """
    This function processes a single cluster of cells, performing additional clustering if the cluster is large enough.

    Parameters:
    params (tuple): A tuple containing the following parameters:
        cl1 (str): The identifier of the cluster to process.
        current_clustering_level (int): The current level of clustering.
        tmp_adata (anndata.AnnData): An AnnData object containing the gene expression data for the cluster.
        n_obs (int): The total number of observations (cells) in the original dataset.

    Returns:
    pandas.Series: A series containing the new cluster assignments for the cells in the cluster.
    """

    cl1, current_clustering_level, tmp_adata, n_obs = params

    if "log1p" in tmp_adata.uns:
        del tmp_adata.uns["log1p"]  

    new_key = "clustering_level_" + str(current_clustering_level)

    if tmp_adata.n_obs < 13 or tmp_adata.n_obs < (n_obs / 1000):
        tmp_adata.obs[new_key] = str(cl1)
    else:
        tmp_adata = clustering_recipe_custom(tmp_adata, clustering_key=new_key, k=12, resolution=0.3, do_highly_variable=True)

        clustersFound = tmp_adata.obs[new_key].unique().tolist()
        clustersFound = [x for x in clustersFound if x != "unassigned"]

        if len(clustersFound) < 2:
            tmp_adata.obs.loc[tmp_adata.obs[new_key] != "unassigned", new_key] = str(cl1)
        else:
            tmp_adata.obs[new_key] = str(cl1) + "_" + tmp_adata.obs[new_key]

    return tmp_adata.obs.loc[:, new_key]

def calculate_odds_ratios(params):
    """
    Calculate odds ratios and fold changes.

    Args:
        params (tuple): A tuple containing cluster, pseudobulk matrix, cluster info, and optionally reference counts.

    Returns:
        ndarray: An array containing cluster, gene, odds ratios, and fold changes.
    """
    # Unpack params
    if len(params) == 6:
        cl2, pseudobulk_matrix, referenceCounts, clusterInfo, typeCol, refCol = params
    else:
        cl2, pseudobulk_matrix, clusterInfo, typeCol = params
        referenceCounts = None

    cellType = clusterInfo.loc[clusterInfo.iloc[:, 0] == cl2, typeCol].values[0]
    countsRef = pseudobulk_matrix.loc[cl2, :].to_numpy()

    if referenceCounts is not None:
        countsRest = referenceCounts.loc[referenceCounts[refCol] == cellType, pseudobulk_matrix.columns].sum(axis=0).to_numpy()
    else:
        dropSamples = clusterInfo.iloc[np.where(clusterInfo[typeCol] == cellType)[0], 0].to_list()
        countsRest = pseudobulk_matrix.loc[~pseudobulk_matrix.index.isin(dropSamples), :].sum(axis=0).to_numpy()

    oddsRatios = odds_ratio_differential(countsRef, countsRest)
    foldChanges = (countsRef/countsRef.sum()) / (countsRest/countsRest.sum())

    if referenceCounts is None and len(params) == 4:
        dict_data = json.load(open("data/CellTypeAliases.json"))

        ## special case for CD4
        if "CD4" in pseudobulk_matrix.columns:
            keepTypes = dict_data["lymphoid"]
            keepTypes = [i for i in keepTypes if i != cellType]

            keepSamples = clusterInfo.iloc[np.where(clusterInfo[typeCol].isin(keepTypes))[0], 0].to_list()

            if len(keepSamples) == 0:
                return np.column_stack((np.repeat(cl2, len(oddsRatios)), pseudobulk_matrix.columns, oddsRatios, foldChanges))

            # find which column of the pseudobulk matrix corresponds to the gene CD4
            geneIndex = np.where(pseudobulk_matrix.columns == "CD4")[0][0]

            # find the counts of the gene CD4 in the reference and the rest of the cells
            countsRef = pseudobulk_matrix.loc[cl2, "CD4"]
            totalRef = pseudobulk_matrix.loc[cl2, :].sum()
            countsRest = pseudobulk_matrix.loc[pseudobulk_matrix.index.isin(keepSamples), "CD4"].sum()
            totalRest = pseudobulk_matrix.loc[pseudobulk_matrix.index.isin(keepSamples), :].sum().sum()

            # calculate the odds ratio and fold change using odds_ratio_efficient function
            oddsRatios[geneIndex] = odds_ratio_differential_efficient(countsRef, totalRef, countsRest, totalRest)
            foldChanges[geneIndex] = (countsRef/totalRef) / (countsRest/totalRest)
        
        ## special case for PTPRC
        #check if PTPRC is in the pseudobulk matrix
        if "PTPRC" in pseudobulk_matrix.columns:
            dropTypes = dict_data["immune"] + [cellType]
            dropTypes = list(set(dropTypes))

            dropSamples = clusterInfo.iloc[np.where(clusterInfo[typeCol].isin(dropTypes))[0], 0].to_list()

            if len(dropSamples) == pseudobulk_matrix.shape[0]:
                return np.column_stack((np.repeat(cl2, len(oddsRatios)), pseudobulk_matrix.columns, oddsRatios, foldChanges))
            
            geneIndex = np.where(pseudobulk_matrix.columns == "PTPRC")[0][0]

            countsRef = pseudobulk_matrix.loc[cl2, "PTPRC"]
            totalRef = pseudobulk_matrix.loc[cl2, :].sum()
            countsRest = pseudobulk_matrix.loc[~pseudobulk_matrix.index.isin(dropSamples), "PTPRC"].sum()
            totalRest = pseudobulk_matrix.loc[~pseudobulk_matrix.index.isin(dropSamples), :].sum().sum()

            oddsRatios[geneIndex] = odds_ratio_differential_efficient(countsRef, totalRef, countsRest, totalRest)
            foldChanges[geneIndex] = (countsRef/totalRef) / (countsRest/totalRest)

        ## special case for AIF1
        if "AIF1" in pseudobulk_matrix.columns:
            dropTypes = dict_data["myeloid"] + [cellType]
            dropTypes = list(set(dropTypes))

            dropSamples = clusterInfo.iloc[np.where(clusterInfo[typeCol].isin(dropTypes))[0], 0].to_list()

            if len(dropSamples) == pseudobulk_matrix.shape[0]:
                return np.column_stack((np.repeat(cl2, len(oddsRatios)), pseudobulk_matrix.columns, oddsRatios, foldChanges))
            
            geneIndex = np.where(pseudobulk_matrix.columns == "AIF1")[0][0]

            countsRef = pseudobulk_matrix.loc[cl2, "AIF1"]
            totalRef = pseudobulk_matrix.loc[cl2, :].sum()
            countsRest = pseudobulk_matrix.loc[~pseudobulk_matrix.index.isin(dropSamples), "AIF1"].sum()
            totalRest = pseudobulk_matrix.loc[~pseudobulk_matrix.index.isin(dropSamples), :].sum().sum()
            
            oddsRatios[geneIndex] = odds_ratio_differential_efficient(countsRef, totalRef, countsRest, totalRest)
            foldChanges[geneIndex] = (countsRef/totalRef) / (countsRest/totalRest)

    return np.column_stack((np.repeat(cl2, len(oddsRatios)), pseudobulk_matrix.columns, oddsRatios, foldChanges))

def calculate_chr_damage(adata, tempRef):
    """
    This function calculates chromosomal damage based on gene expression data.

    Parameters:
    adata (anndata.AnnData): An AnnData object containing the gene expression data.
    tempRef (pandas.DataFrame): A DataFrame containing reference gene expression data.

    Returns:
    cnvFC (pandas.DataFrame): A DataFrame containing the copy number variation for each cell.
    cnvLoc (dict): A dictionary containing the location of the copy number variation.
    """
    from infercnvpy.io import genomic_position_from_gtf
    from infercnvpy.tl import infercnv
    
    # Copy the input AnnData object
    cnv_adata = adata.copy()

    # Add genomic position information to the AnnData object
    genomic_position_from_gtf("data/Homo_sapiens.GRCh38.110.gtf.gz", adata=cnv_adata)

    # Add 'chr' prefix to chromosome names and filter out null and sex chromosomes
    cnv_adata.var['chromosome'] = cnv_adata.var['chromosome'].apply(lambda x: 'chr' + str(x) if pd.notnull(x) else x)
    var_mask = cnv_adata.var["chromosome"].isnull()
    var_mask = var_mask | cnv_adata.var["chromosome"].isin(["chrX", "chrY"])
    cnv_adata = cnv_adata[:, ~var_mask]    

    # Filter out genes not present in the reference
    common_genes = cnv_adata.var_names[cnv_adata.var_names.isin(tempRef.index)]
    cnv_adata = cnv_adata[:, common_genes]

    # Normalize the counts in the AnnData object
    cnv_adata.X = cnv_adata.layers["raw"].todense()
    median_total_counts = 1e5
    scaling_factor = median_total_counts / cnv_adata.X.sum(axis=1)
    cnv_adata.layers['norm_counts'] = cnv_adata.X.toarray() * scaling_factor[:, np.newaxis]
    sc.pp.log1p(cnv_adata, layer='norm_counts', base = 10)
    cnv_adata.X = cnv_adata.layers["norm_counts"]

    # Normalize the reference and calculate the mean log-transformed reference
    tempRef = tempRef.loc[common_genes, :].to_numpy()
    tempRef = tempRef / tempRef.sum(axis=0)
    tempRef = np.log10(tempRef * median_total_counts + 1)
    tempRef = tempRef.mean(axis=1)
    tempRef = tempRef[np.newaxis,:]

    # Infer copy number variation
    infercnv(cnv_adata, reference=tempRef, reference_cat=None, reference_key=None, key_added="cnv" )

    # Extract the copy number variation and its location
    cnvFC = cnv_adata.obsm["X_cnv"].toarray()
    cnvFC = pd.DataFrame(cnvFC, index=cnv_adata.obs_names)
    cnvLoc = cnv_adata.uns["cnv"]

    return cnvFC, cnvLoc