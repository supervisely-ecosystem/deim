#!/bin/bash
set -e

echo "=== DEIM DeepStream End-to-End Inference ==="

OUTPUT_MODE=${OUTPUT_MODE:-video}
INPUT_VIDEO=${INPUT_VIDEO:-/data/input.mp4}
MODEL_DIR=${MODEL_DIR:-/data/model}
OUTPUT_FILE=${OUTPUT_FILE:-/data/output}
MODEL_NAME=${MODEL_NAME:-model}

echo "Configuration:"
echo "  Output mode: $OUTPUT_MODE"
echo "  Input video: $INPUT_VIDEO"
echo "  Model directory: $MODEL_DIR"
echo "  Output file: $OUTPUT_FILE"
echo "  Model name: $MODEL_NAME"

if [ ! -f "$INPUT_VIDEO" ]; then
    echo "ERROR: Input video not found: $INPUT_VIDEO"
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

MODEL_PTH=$(find "$MODEL_DIR" -name "*.pth" | head -1)
MODEL_CONFIG=$(find "$MODEL_DIR" -name "*config*.yml" -o -name "*config*.yaml" | head -1)
MODEL_META=$(find "$MODEL_DIR" -name "*meta*.json" -o -name "model_meta.json" | head -1)

if [ -z "$MODEL_PTH" ]; then
    echo "ERROR: No .pth file found in $MODEL_DIR"
    exit 1
fi

if [ -z "$MODEL_CONFIG" ]; then
    echo "ERROR: No config YAML file found in $MODEL_DIR"
    exit 1
fi

if [ -z "$MODEL_META" ]; then
    echo "ERROR: No meta JSON file found in $MODEL_DIR"
    exit 1
fi

echo "Found model files:"
echo "  PTH: $MODEL_PTH"
echo "  Config: $MODEL_CONFIG" 
echo "  Meta: $MODEL_META"

cd /workspace/supervisely_integration/deepstream

echo "Step 1: Generating labels.txt from model_meta.json..."
python3 make_labels_txt.py "$MODEL_META"
LABELS_FILE="$(dirname "$MODEL_META")/labels.txt"
echo "Labels file created: $LABELS_FILE"

NUM_CLASSES=$(wc -l < "$LABELS_FILE")
echo "Number of classes: $NUM_CLASSES"

echo "Step 2: Converting model to TensorRT engine..."
ENGINE_FILE="$(dirname "$MODEL_PTH")/${MODEL_NAME}.engine"
python3 convert.py \
    --pth_path "$MODEL_PTH" \
    --config_path "$MODEL_CONFIG" \
    --output_dir "$(dirname "$MODEL_PTH")" \
    --model_name "$MODEL_NAME" \
    --fp16
echo "Engine file created: $ENGINE_FILE"

echo "Step 3: Updating configuration files..."

CONFIG_INFER="configs/deepstream-app/config_infer_dfine.txt"
cp "$CONFIG_INFER" "${CONFIG_INFER}.bak"

sed -i "s|model-engine-file=.*|model-engine-file=$ENGINE_FILE|" "$CONFIG_INFER"
sed -i "s|labelfile-path=.*|labelfile-path=$LABELS_FILE|" "$CONFIG_INFER"
sed -i "s|num-detected-classes=.*|num-detected-classes=$NUM_CLASSES|" "$CONFIG_INFER"

sed -i '/^\[class-attrs-[0-9]*\]/,/^$/d' "$CONFIG_INFER"
for ((i=0; i<NUM_CLASSES; i++)); do
    echo "" >> "$CONFIG_INFER"
    echo "[class-attrs-$i]" >> "$CONFIG_INFER"
    echo "pre-cluster-threshold=0.3" >> "$CONFIG_INFER"
done

PIPELINE_CONFIG="configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt"
cp "$PIPELINE_CONFIG" "${PIPELINE_CONFIG}.bak"

sed -i "s|uri=.*|uri=file://$INPUT_VIDEO|" "$PIPELINE_CONFIG"

if [ "$OUTPUT_MODE" = "video" ]; then
    OUTPUT_VIDEO="${OUTPUT_FILE}.mp4"
    sed -i "s|output-file=.*|output-file=$OUTPUT_VIDEO|" "$PIPELINE_CONFIG"
fi

sed -i "/^\[primary-gie\]/,/^\[/ { s|model-engine-file=.*|model-engine-file=$ENGINE_FILE|; }" "$PIPELINE_CONFIG"

if [ "$OUTPUT_MODE" = "video" ]; then
    echo "Step 4: Running DeepStream inference (video mode)..."
    cd configs/deepstream-app/
    deepstream-app -c source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
    
    echo "Video output saved to: $OUTPUT_VIDEO"
    
elif [ "$OUTPUT_MODE" = "json" ]; then
    echo "Step 4: Running DeepStream inference (JSON mode)..."
    OUTPUT_JSON="${OUTPUT_FILE}.json"
    cd configs/deepstream-app/
    ./deepstream_save_predictions "$INPUT_VIDEO" "$OUTPUT_JSON"
    
    echo "JSON predictions saved to: $OUTPUT_JSON"
    
else
    echo "ERROR: Invalid output mode: $OUTPUT_MODE (must be 'video' or 'json')"
    exit 1
fi

echo "=== Inference Complete ==="