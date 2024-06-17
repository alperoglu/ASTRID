options(warn = -1)
suppressMessages(library(tidyverse))
suppressMessages(library(Matrix))
suppressMessages(library(SingleCellExperiment))
suppressMessages(library(SingleR))

args <- commandArgs(TRUE)

input_adata <- args[1]
output_csv <- args[2]

excludeSamples <- read.table(file = "/scratch/alper.eroglu/GRINT/GRINT_R/excludeSingleR.csv", header = T, sep = ",")

pseudobulk_matrix <- read.csv(input_adata, row.names=1)

# newRef <- readRDS("/scratch/alper.eroglu/GRINT/data/ASTRID_SingleR_Reference_20240408.Rds")
newRef <- readRDS("/scratch/alper.eroglu/GRINT/data/ASTRID_SingleR_Reference_20240422.Rds")

if(nrow(excludeSamples > 0)){
  
  newRef <- newRef[,!(colnames(newRef) %in% excludeSamples$samples)]
  
}

newRefCounts <- assays(newRef)[["counts"]]

pseudobulk_matrix <- t(pseudobulk_matrix)

common_genes <- intersect(rownames(pseudobulk_matrix), rownames(newRefCounts))

pseudobulk_matrix <- pseudobulk_matrix[common_genes,]
pseudobulk_matrix <- sweep(pseudobulk_matrix,2,colSums(pseudobulk_matrix), `/`)
pseudobulk_matrix <- log10(pseudobulk_matrix *1e5 + 1)
colnames(pseudobulk_matrix) <- gsub("clustering\\_level\\_2\\.", replacement = "", colnames(pseudobulk_matrix))

newRefCounts <- newRefCounts[common_genes,]
newRefCounts <- sweep(newRefCounts,2,colSums(newRefCounts), `/`)
newRefCounts <- log10(newRefCounts *1e5 + 1)

predictions <- SingleR(pseudobulk_matrix, ref = newRefCounts, labels = newRef$cellTypeGRINT, de.n = 50)

# predictions <- SingleR(test = obj.sce, ref = hpca.se, assay.type.test="logcounts", assay.type.ref = "logcounts",
#                        labels = hpca.se$label.fine, clusters = obj.sce$clustering_level_2)

write.table(predictions, file = output_csv, sep = ",", quote = F)
