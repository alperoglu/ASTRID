options(warn = -1)
suppressMessages(library(tidyverse))
suppressMessages(library(Matrix))
suppressMessages(library(SingleCellExperiment))
suppressMessages(library(SingleR))

setwd("/scratch/alper.eroglu/tools/ASTRID/")
args <- commandArgs(TRUE)

input_adata <- args[1]
output_csv <- args[2]

excludeSamples <- read.table(file = "data/excludeSingleR.csv", header = T, sep = ",")

pseudobulk_matrix <- read.csv(input_adata, row.names=1)

newRef <- readRDS("data/ASTRID_SingleR_Reference_20240701.Rds")

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

write.table(predictions, file = output_csv, sep = ",", quote = F)
