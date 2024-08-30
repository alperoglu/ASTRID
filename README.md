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
                        Input file path
  --input_prefix INPUT_PREFIX
                        Input prefix
  --output_file OUTPUT_FILE
                        Output file path
  --output_clustering_results OUTPUT_CLUSTERING_RESULTS
                        Output clustering results path
  --final_key FINAL_KEY
                        Key for final level of clustering (column in AnnData.obs)
  --author_type AUTHOR_TYPE
                        Author cell type column name

```

Example bash script in RunASTRID_Ji.sh.
