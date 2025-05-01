#!/bin/bash
set -euo pipefail

# Function to clean up on Ctrl+C or error
trap_ctrlc() {
    echo "Error or interrupt caught. Cleaning up. See log at $LOGFILE"

    if [ -d "$inference_base" ]; then
        rm -rf "$inference_base"
    fi

    exit 99
}

trap trap_ctrlc INT TERM ERR

# Help message
help_message() {
    echo "Usage: $0 -i <pre_image> -p <post_image> -l <localization_json> -c <model_weights> -o <output_file> [-e <venv_activate_script>]"
    echo ""
    echo "Arguments:"
    echo "  -i    Path to pre-disaster image (not used in this version but kept for compatibility)"
    echo "  -p    Path to post-disaster image"
    echo "  -l    Path to localization JSON (polygon annotations)"
    echo "  -c    Path to classification model weights"
    echo "  -o    Path to save final output image"
    echo "  -e    Path to activate virtual environment (optional)"
    echo "  -h    Display help"
    echo ""
}

# Default variables
input=""
input_post=""
localization_json=""
classification_weights=""
output_file=""
virtual_env=""
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
inference_base="/tmp/inference"
LOGFILE="/tmp/inference.log"

# Parse arguments
while getopts "i:p:o:l:c:e:h" OPTION; do
    case $OPTION in
        h) help_message; exit 0 ;;
        i) input="$OPTARG" ;;  # Unused
        p) input_post="$OPTARG" ;;
        o) output_file="$OPTARG" ;;
        l) localization_json="$OPTARG" ;;
        c) classification_weights="$OPTARG" ;;
        e) virtual_env="$OPTARG" ;;
        ?) help_message; exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$input_post" || -z "$localization_json" || -z "$classification_weights" || -z "$output_file" ]]; then
    help_message
    exit 1
fi

# Prepare working environment
mkdir -p "$inference_base"
touch "$LOGFILE"

# Optional virtual environment activation
if [[ -n "$virtual_env" && -f "$virtual_env" ]]; then
    echo "Activating virtual environment from $virtual_env"
    source "$virtual_env"
else
    echo "No virtual environment provided. Using system Python environment."
fi

# Step 1: Localization
echo "[1/4] Extracting polygons from label file..."
python3 "$SCRIPT_DIR/process_data_inference.py" \
    --input_img "$input_post" \
    --label_path "$localization_json" \
    --output_dir "$inference_base/output_polygons" \
    --output_csv "$inference_base/output.csv" >> "$LOGFILE" 2>&1

# Step 2: Classification
echo "[2/4] Running classification model..."
python3 "$SCRIPT_DIR/inference_proto.py" \
    --test_data "$inference_base/output_polygons" \
    --test_csv "$inference_base/output.csv" \
    --model_weights "$classification_weights" \
    --output_json "$inference_base/classification_inference.json" >> "$LOGFILE" 2>&1

# Step 3: Merge classification and geometry
echo "[3/4] Combining classification with geometry..."
python3 "$SCRIPT_DIR/combine_jsons.py" \
    --polys "$localization_json" \
    --classes "$inference_base/classification_inference.json" \
    --output "$inference_base/inference.json" >> "$LOGFILE" 2>&1

# Step 4: Generate inference image
echo "[4/4] Generating output image..."
python3 "$SCRIPT_DIR/inference_image_output.py" \
    --input "$inference_base/inference.json" \
    --output "$output_file" >> "$LOGFILE" 2>&1

# Cleanup
echo "Cleaning up temporary files..."
rm -rf "$inference_base"

echo "Inference workflow completed successfully."
echo "Output image saved to: $output_file"
echo "Log saved to: $LOGFILE"

