#!/bin/bash
set -euo pipefail

# This function is called when Ctrl-C or Error occurs
function trap_ctrlc () {
    echo "Ctrl-C or Error caught...performing cleanup, check /tmp/inference.log"

    if [ -d /tmp/inference ]; then
        rm -rf /tmp/inference
    fi

    exit 99
}

trap "trap_ctrlc" 2 9 13 3

help_message() {
    printf "${0}: Automates the workflow for localization and classification inference\n"
    printf "\t-i: Full path to input pre-disaster image\n"
    printf "\t-p: Full path to input post-disaster image\n"
    printf "\t-l: Path to localization JSON (polygons)\n"
    printf "\t-c: Path to classification model weights\n"
    printf "\t-o: Path to save final output image\n"
    printf "\t-e: Path to virtual environment activate script (optional)\n"
    printf "\n"
}

# Variables
input=""
input_post=""
localization_json=""
classification_weights=""
output_file=""
virtual_env=""
inference_base="/tmp/inference"
LOGFILE="/tmp/inference_log"

# Parse command-line arguments
while getopts "i:p:o:l:c:e:h" OPTION
do
    case $OPTION in
        h) help_message; exit 0;;
        i) input="$OPTARG";;
        p) input_post="$OPTARG";;
        o) output_file="$OPTARG";;
        l) localization_json="$OPTARG";;
        c) classification_weights="$OPTARG";;
        e) virtual_env="$OPTARG";;
        ?) help_message; exit 1;;
    esac
done

# Check required arguments
if [ -z "$input" ] || [ -z "$input_post" ] || [ -z "$localization_json" ] || [ -z "$classification_weights" ] || [ -z "$output_file" ]; then
    help_message
    exit 1
fi

# Create output directories
mkdir -p "$inference_base"
touch "$LOGFILE"

# Activate virtual environment if provided
if [ -f "$virtual_env" ]; then
    source "$virtual_env"
else
    echo "No virtual environment provided. Ensure dependencies are installed globally."
fi

# Process localization (extract polygons)
echo "Running localization..."
python3 ./process_data_inference.py \
    --input_img "$input_post" \
    --label_path "$localization_json" \
    --output_dir "$inference_base/output_polygons" \
    --output_csv "$inference_base/output.csv" >> "$LOGFILE" 2>&1

# Perform classification on extracted polygons
echo "Running classification..."
python3 ./inference_proto.py \
    --test_data "$inference_base/output_polygons" \
    --test_csv "$inference_base/output.csv" \
    --model_weights "$classification_weights" \
    --output_json "$inference_base/classification_inference.json" >> "$LOGFILE" 2>&1

# Combine localization and classification results
echo "Combining results..."
python3 ./combine_jsons.py \
    --polys "$localization_json" \
    --classes "$inference_base/classification_inference.json" \
    --output "$inference_base/inference.json" >> "$LOGFILE" 2>&1

# Generate the final inference image
echo "Creating final output image..."
python3 ./inference_image_output.py \
    --input "$inference_base/inference.json" \
    --output "$output_file" >> "$LOGFILE" 2>&1

# Cleanup temporary files
echo "Cleaning up temporary files..."
rm -rf "$inference_base"

echo "Workflow completed. Output saved to: $output_file"
