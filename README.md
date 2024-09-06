# ᛅᛋᛏᚱᛁᛏ (ASTRID - Automatized Single-cell Typing for tumoR transcrIptomics Data)

Tool for automatic annotation of cells from scRNA-seq (Single Cell RNA sequencing) and Xenium in situ sequencing datasets. Tailored for samples from tumor. Currently works with AnnData objects from [Scanpy](https://scanpy.readthedocs.io/en/stable/).

```
Run ASTRID (Automatized Single-cell Typing for tumoR transcrIptomics Data) ᛅᛋᛏᚱᛁᛏ pipeline

options:
  -h, --help            show this help message and exit
  --all                 Run all tasks
  --clustering          Run clustering
  --annotation          Run annotation
  --validation          Run validation
  --damage              Run cancer damage
  --input_file INPUT_FILE
                        Input file path (/your/input/folder/file.h5ad)
  --input_prefix INPUT_PREFIX
                        Input prefix (Sample0)
  --output_file OUTPUT_FILE
                        Output file path (/your/output/folder/file.h5ad)
  --output_clustering_results OUTPUT_CLUSTERING_RESULTS
                        Output clustering results path (/your/output/folder/astrid_output_file.csv)
  --final_key FINAL_KEY
                        Key for final level of clustering (ASTRID_Clusters) (column in AnnData.obs) 
  --author_type AUTHOR_TYPE
                        Author cell type column name (column in AnnData.obs)

```

Example bash script in ***RunASTRID_Ji.sh***.

### **Requirements**

* Python (tested in version Python 3.10.10)
  * Numpy - 1.23.4
  * Pandas - 2.2.2
  * Scanpy - 1.9.3
  * scikit-learn - 1.5.0
  * scipy - 1.8.1
  * seaborn - 0.12.2
  * leidenalg - 0.9.1
  * matplotlib - 3.7.2
  * regex
  * infercnvpy - 0.4.3
  * colorir - 2.0.0
  * umap-learn 0.5.3
  * adpbulk - 0.1.3    

* R (tested in R version 4.3.3)
  * SingleR - 2.4.1
  * tidyverse - 2.0.0
  * Matrix - 1.6-5
  * SingleCellExperiment - 1.24.0

