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

# def evaluate_formula(formula, x):
#     if formula == "":
#         return True
#     if not re.search(r"(x|\(|\)|\+|\!)", formula) and formula not in x.keys():
#         return False
#     if not re.search(r"(x|\(|\)|\+|\!)", formula):
#         return x[formula]
#     if not re.search(r"(x|\(|\)|\+)", formula) and re.search(r"!", formula) and formula.replace("!", "") not in x.keys():
#         return False
#     if not re.search(r"(x|\(|\)|\+)", formula) and re.search(r"!", formula):
#         return not x[formula.replace("!", "")]
#     if not re.search(r"(x|\(|\))", formula) and re.search(r"\+", formula):
#         return any([evaluate_formula(f, x) for f in formula.split("+")])
#     if re.search(r"(x|\(|\))", formula):
#         return all([evaluate_formula(f, x) for f in re.split(r"x|\(|\)", formula)])


# entropy function for a array of percentages and return zero if percentage is zero
def entropy(p):
    tmp = 0
    for i in range(len(p)):
        if p[i] == 0:
           tmp = tmp + 0
        else:
            tmp = tmp + p[i] * np.log(p[i])
    return -tmp

def mutual_info_score(contingency=None):
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

# function to calculate the normalized mutual information between two categorical variables from adata.obs using norm_mi function
def calculate_nmi(adata, clustering_key, control_key):
        
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
    indices = np.argpartition(D, k, axis=1)[:, :k]
    distances = np.take_along_axis(D, indices, axis=1)
    return indices, distances

def clustering_modularity_custom(adata, distance_key, k: int, resolution, key = "leiden_modularity_"):
    
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
    adata = normalize_data(adata)
    ndims_measured = max(2, np.floor(adata.n_obs/100).astype(int))

    # # check the median total counts of the cells in the cluster from the raw data and if it is less than 1500 set do highly variable to False
    # total_counts = np.ravel(adata.layers["raw"].sum(axis=1))
    # median_total_counts = np.median(total_counts)
    # 
    # if median_total_counts < 1500 and do_highly_variable:
    #     do_highly_variable = False

    if do_highly_variable: # and ndims_measured != 2:
        bin_count = max(20, int(np.floor(adata.n_vars/1000)))

        sc.pp.highly_variable_genes(adata, flavor="seurat", min_disp=0.5, max_mean=np.inf, min_mean = np.log(2), n_bins = bin_count, layer = "norm_counts")

        num_highly_variable_genes = adata.var['highly_variable'].sum()

        if num_highly_variable_genes < 1000:
            sc.pp.highly_variable_genes(adata, flavor="seurat", n_bins = bin_count, layer = "norm_counts", n_top_genes=1000)
        
        if ndims_measured > adata.var_names[adata.var["highly_variable"]].shape[0]:
            ndims_measured = adata.var_names[adata.var["highly_variable"]].shape[0] - 1

    # sc.tl.pca(adata, svd_solver='arpack', n_comps=ndims_measured, use_highly_variable=do_highly_variable and ndims_measured != 2)
    sc.tl.pca(adata, svd_solver='arpack', n_comps=ndims_measured, use_highly_variable=do_highly_variable)

    adata.obsm['X_pca_norm_counts'] = adata.obsm['X_pca']
    adata.varm['PCs_norm_counts'] = adata.varm['PCs']
    adata.uns['pca_norm_counts'] = adata.uns['pca']

    adata.obsp['pcCustom_norm_counts_distances'] = pairwise_distances(adata.obsm['X_pca_norm_counts'][:,0:ndims_measured])

    adata = clustering_modularity_custom(adata, "pcCustom_norm_counts_distances", k = k, resolution = resolution, key= clustering_key)

    return adata

def runSingleR(adata_file, output_file, rscript_path = '/scratch/alper.eroglu/miniconda3/envs/r4/bin/Rscript'):
    # os.system('mkdir '+tmpPath)
    # print('Writing files ...')
    # dat.write_h5ad(tmpPath+'/tmp_singler.h5ad')
    print('Running SingleR ...')
    os.system('nice -19 ' + rscript_path + ' /scratch/alper.eroglu/GRINT/GRINT_R/RunSingleR.R '+adata_file + ' ' + output_file)
    result=pd.read_csv(output_file)
    #os.system('rm -r /scratch/jakob.rosenbauer/data_local/scRNAseq/tmpSR/')
    return result

def plot_confusion_matrix(adata, clustering_key, control_key, output_file):
    
    # convert the clustering key to string and control key to category 
    adata.obs[clustering_key] = adata.obs[clustering_key].astype(str)
    adata.obs[control_key] = adata.obs[control_key].astype("category")

    # create the confusion matrix 
    confusion_matrix = pd.crosstab(adata.obs[control_key], adata.obs[clustering_key], margins=True)
    confusion_matrix = confusion_matrix.drop('All', axis=1)
    confusion_matrix = confusion_matrix.drop('All', axis=0)

    # normalize the confusion matrix by column so thay they sum up to 1
    confusion_matrix = confusion_matrix.div(confusion_matrix.sum(axis=0), axis=1)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10,10), gridspec_kw={'wspace':0.5})

    ax1_dict = sc.pl.umap(adata, color=control_key, show=False, ax = ax1)

    ax1_dict = sc.pl.umap(adata, color=clustering_key, show=False, ax = ax2)

    ax2_dict = sns.heatmap(confusion_matrix, fmt='g', cmap='Blues', cbar=False, xticklabels=True, yticklabels=True, ax = ax3)

    confusion_matrix = pd.crosstab(adata.obs[control_key], adata.obs[clustering_key])

    # normalize the confusion matrix by column so thay they sum up to 1
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
    
    totalRefCounts = np.sum(refCounts)
    totalOtherCounts = np.sum(otherCounts)
    
    term1 = betaln(refCounts + 1, totalRefCounts - refCounts + 1)
    term2 = betaln(otherCounts + 1, totalOtherCounts - otherCounts + 1)
    term3 = betaln(refCounts + otherCounts + 1, totalRefCounts + totalOtherCounts - refCounts - otherCounts + 1)
    
    result = (term1 + term2) - term3
    result = result / np.log(10)
    
    return result

def odds_ratio_differential_efficient(refCounts, totalRefCounts, otherCounts, totalOtherCounts):
        
    term1 = betaln(refCounts + 1, totalRefCounts - refCounts + 1)
    term2 = betaln(otherCounts + 1, totalOtherCounts - otherCounts + 1)
    term3 = betaln(refCounts + otherCounts + 1, totalRefCounts + totalOtherCounts - refCounts - otherCounts + 1)
    
    result = (term1 + term2) - term3
    result = result / np.log(10)
    
    return result

def process_cluster(params):

    cl1, current_clustering_level, tmp_adata, n_obs = params

    if "log1p" in tmp_adata.uns:
        del tmp_adata.uns["log1p"]  

    new_key = "clustering_level_" + str(current_clustering_level)
    # tmp_adata = adata[adata.obs["clustering_level_" + str(current_clustering_level - 1)] == cl1].copy()

    if tmp_adata.n_obs < 13 or tmp_adata.n_obs < (n_obs / 1000):
        tmp_adata.obs[new_key] = str(cl1)
    else:
        tmp_adata = clustering_recipe_custom(tmp_adata, clustering_key=new_key, k=12, resolution=0.3, do_highly_variable=True)

        clustersFound = tmp_adata.obs[new_key].unique().tolist()
        clustersFound = [x for x in clustersFound if x != "unassigned"]

        if len(clustersFound) < 2:
            tmp_adata.obs.loc[tmp_adata.obs[new_key] != "unassigned", new_key] = str(cl1)
        else:
            # add the level 1 cluster to the level 2 cluster
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
        dict_data = json.load(open("/scratch/alper.eroglu/GRINT/data/CellTypeAliases.json"))

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
    from infercnvpy.io import genomic_position_from_gtf
    from infercnvpy.tl import infercnv
    
    cnv_adata = adata.copy()

    genomic_position_from_gtf("/scratch/alper.eroglu/GRINT/data/Homo_sapiens.GRCh38.110.gtf.gz", adata=cnv_adata)

    cnv_adata.var['chromosome'] = cnv_adata.var['chromosome'].apply(lambda x: 'chr' + str(x) if pd.notnull(x) else x)
    var_mask = cnv_adata.var["chromosome"].isnull()
    var_mask = var_mask | cnv_adata.var["chromosome"].isin(["chrX", "chrY"])
    cnv_adata = cnv_adata[:, ~var_mask]    

    common_genes = cnv_adata.var_names[cnv_adata.var_names.isin(tempRef.index)]
    cnv_adata = cnv_adata[:, common_genes]

    cnv_adata.X = cnv_adata.layers["raw"].todense()
    median_total_counts = 1e5
    scaling_factor = median_total_counts / cnv_adata.X.sum(axis=1)
    cnv_adata.layers['norm_counts'] = cnv_adata.X.toarray() * scaling_factor[:, np.newaxis]
    sc.pp.log1p(cnv_adata, layer='norm_counts', base = 10)
    cnv_adata.X = cnv_adata.layers["norm_counts"]

    tempRef = tempRef.loc[common_genes, :].to_numpy()
    tempRef = tempRef / tempRef.sum(axis=0)
    tempRef = np.log10(tempRef * median_total_counts + 1)
    tempRef = tempRef.mean(axis=1)
    tempRef = tempRef[np.newaxis,:]

    infercnv(cnv_adata, reference=tempRef, reference_cat=None, reference_key=None, key_added="cnv" )

    cnvFC = cnv_adata.obsm["X_cnv"].toarray()
    cnvFC = pd.DataFrame(cnvFC, index=cnv_adata.obs_names)

    cnvLoc = cnv_adata.uns["cnv"]

    return cnvFC, cnvLoc