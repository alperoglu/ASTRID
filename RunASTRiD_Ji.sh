#!/bin/bash

if [ "$#" -eq 2 ]; then
    SAMPLE_TO_TEST=$1
    RUN_TYPE=$2
else
    echo "Enter the sample name:"
    read SAMPLE_TO_TEST
    echo "Do you want to run full, just annotation, just validation or just chromosome damage? (full/annotation/validation/damage)"
    read RUN_TYPE
fi

# Define the paths for the input and output files
ADATA_FILE="/path/to/file/${SAMPLE_TO_TEST}_preASTRID.h5ad"
PREFIX=$SAMPLE_TO_TEST
OUTPUT_FILE="/path/to/file/${SAMPLE_TO_TEST}_v0.03_outASTRID.h5ad"
CLUSTERING_RESULT="/path/to/file/${SAMPLE_TO_TEST}/${SAMPLE_TO_TEST}_ASTRID_v0.01_Result.csv"
AUTHOR_TYPE="level2_celltype"
LOG_FILE="/path/to/file/${SAMPLE_TO_TEST}/${SAMPLE_TO_TEST}_ASTRID_v0.01.log"

# Print the paths to check them
echo "Adata file: $ADATA_FILE"
echo "Prefix: $PREFIX"
echo "Output file: $OUTPUT_FILE"
echo "Clustering result: $CLUSTERING_RESULT"
echo "Log file: $LOG_FILE"

# check if the input file exists
if [ ! -f $ADATA_FILE ]; then
    echo "The input file does not exist!"
    exit 1
fi

# accept only the first letter of the option
RUN_TYPE=${RUN_TYPE:0:1}

# convert the input to lowercase
RUN_TYPE=${RUN_TYPE,,}

# check if the user entered a valid option
if [ $RUN_TYPE != "a" ] && [ $RUN_TYPE != "v" ] && [ $RUN_TYPE != "f" ] && [ $RUN_TYPE != "d" ]; then
    echo "You did not enter a valid option!"
    exit 1
fi

# if the user wants to run all, run the script with the --all option
if [ $RUN_TYPE == "f" ]; then
    echo "Running full ASTRID with the --all option..."
    nice -19 python3 /path/to/script/folder/ASTRID_v0.01.py --all --input_file $ADATA_FILE --input_prefix $PREFIX --output_file $OUTPUT_FILE --output_clustering_results $CLUSTERING_RESULT --author_type $AUTHOR_TYPE > $LOG_FILE 2>&1

fi

if [ $RUN_TYPE == "a" ]; then
    echo "Running ASTRID with the --annotation option..."
    nice -19 python3 /path/to/script/folder/ASTRID_v0.01.py --annotation  --validation --damage --input_file $ADATA_FILE --input_prefix $PREFIX --output_file $OUTPUT_FILE --output_clustering_results $CLUSTERING_RESULT --author_type $AUTHOR_TYPE > $LOG_FILE 2>&1
fi

# if the user wants to run the validation, run the script with the --validation option
if [ $RUN_TYPE == "v" ]; then
    echo "Running ASTRID with the --validation option..."
    nice -19 python3 /path/to/script/folder/ASTRID_v0.01.py --validation --damage --input_file $ADATA_FILE --input_prefix $PREFIX --output_file $OUTPUT_FILE --output_clustering_results $CLUSTERING_RESULT --author_type $AUTHOR_TYPE > $LOG_FILE 2>&1
fi

# if the user wants to run the damage, run the script with the --damage option
if [ $RUN_TYPE == "d" ]; then
    echo "Running ASTRID with the --damage option..."
    nice -19 python3 /path/to/script/folder/ASTRID_v0.01.py --damage --input_file $ADATA_FILE --input_prefix $PREFIX --output_file $OUTPUT_FILE --output_clustering_results $CLUSTERING_RESULT --author_type $AUTHOR_TYPE > $LOG_FILE 2>&1
fi

echo "Done running ASTRID! Check the log file " $LOG_FILE " for more details."
