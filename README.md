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

Example bash script in RunASTRID_Ji.sh.
