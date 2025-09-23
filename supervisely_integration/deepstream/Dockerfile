# End-to-End DEIM + DeepStream Docker Container
FROM nvcr.io/nvidia/deepstream:6.4-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    pkg-config \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
WORKDIR /workspace
COPY . /workspace/

# Install Python requirements for DEIM
RUN pip3 install -r dev_requirements.txt

# Create entrypoint script
RUN cat > /workspace/entrypoint.sh << 'EOF'
#!/bin/bash
set -e

echo "=== DEIM DeepStream End-to-End Inference ==="

# Parse arguments
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

# Check input video
if [ ! -f "$INPUT_VIDEO" ]; then
    echo "ERROR: Input video not found: $INPUT_VIDEO"
    exit 1
fi

# Check model directory and find model files
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Auto-detect model files
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

# Step 1: Generate labels.txt from model_meta.json
echo "Step 1: Generating labels.txt from model_meta.json..."
python3 make_labels_file.py "$MODEL_META"
LABELS_FILE="$(dirname "$MODEL_META")/labels.txt"
echo "Labels file created: $LABELS_FILE"

# Count number of classes
NUM_CLASSES=$(wc -l < "$LABELS_FILE")
echo "Number of classes: $NUM_CLASSES"

# Step 2: Convert model to TensorRT engine
echo "Step 2: Converting model to TensorRT engine..."
ENGINE_FILE="$(dirname "$MODEL_PTH")/${MODEL_NAME}.engine"
python3 convert.py \
    --pth_path "$MODEL_PTH" \
    --config_path "$MODEL_CONFIG" \
    --output_dir "$(dirname "$MODEL_PTH")" \
    --model_name "$MODEL_NAME" \
    --fp16
echo "Engine file created: $ENGINE_FILE"

# Step 3: Compile custom C++ parser
echo "Step 3: Compiling custom C++ parser..."
g++ -c -fPIC -std=c++11 nvds_dfine_parser.cpp \
    -I /opt/nvidia/deepstream/deepstream/sources/includes \
    -I /usr/local/cuda/include
g++ -shared nvds_dfine_parser.o -o libnvds_dfine_parser.so
cp libnvds_dfine_parser.so /opt/nvidia/deepstream/deepstream/lib/
echo "Parser compiled and installed"

# Step 4: Update configuration files
echo "Step 4: Updating configuration files..."

# Update inference config
CONFIG_INFER="configs/deepstream-app/config_infer_dfine.txt"
cp "$CONFIG_INFER" "${CONFIG_INFER}.bak"

# Update paths in config_infer_dfine.txt
sed -i "s|model-engine-file=.*|model-engine-file=$ENGINE_FILE|" "$CONFIG_INFER"
sed -i "s|labelfile-path=.*|labelfile-path=$LABELS_FILE|" "$CONFIG_INFER"
sed -i "s|num-detected-classes=.*|num-detected-classes=$NUM_CLASSES|" "$CONFIG_INFER"

# Remove existing class-attrs-* sections and add new ones for each class
sed -i '/^\[class-attrs-[0-9]*\]/,/^$/d' "$CONFIG_INFER"
for ((i=0; i<NUM_CLASSES; i++)); do
    echo "" >> "$CONFIG_INFER"
    echo "[class-attrs-$i]" >> "$CONFIG_INFER"
    echo "pre-cluster-threshold=0.3" >> "$CONFIG_INFER"
done

# Update pipeline config
PIPELINE_CONFIG="configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt"
cp "$PIPELINE_CONFIG" "${PIPELINE_CONFIG}.bak"

# Update input video path in [source0] section
sed -i "s|uri=.*|uri=file://$INPUT_VIDEO|" "$PIPELINE_CONFIG"

# Update output file based on mode
if [ "$OUTPUT_MODE" = "video" ]; then
    OUTPUT_VIDEO="${OUTPUT_FILE}.mp4"
    sed -i "s|output-file=.*|output-file=$OUTPUT_VIDEO|" "$PIPELINE_CONFIG"
fi

# Update engine file path in [primary-gie] section
sed -i "/^\[primary-gie\]/,/^\[/ { s|model-engine-file=.*|model-engine-file=$ENGINE_FILE|; }" "$PIPELINE_CONFIG"

if [ "$OUTPUT_MODE" = "video" ]; then
    # Configure video output
    OUTPUT_VIDEO="${OUTPUT_FILE}.mp4"
    sed -i "s|output-file=.*|output-file=$OUTPUT_VIDEO|" "$PIPELINE_CONFIG"
    
    echo "Step 5: Running DeepStream inference (video mode)..."
    cd configs/deepstream-app/
    deepstream-app -c source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
    
    echo "Video output saved to: $OUTPUT_VIDEO"
    
elif [ "$OUTPUT_MODE" = "json" ]; then
    # Step 5a: Compile prediction extractor for JSON mode
    echo "Step 5a: Compiling prediction extractor..."
    cd configs/deepstream-app/
    make clean
    make
    
    # Configure JSON output
    OUTPUT_JSON="${OUTPUT_FILE}.json"
    
    echo "Step 5b: Running DeepStream inference (JSON mode)..."
    ./deepstream_save_predictions "$INPUT_VIDEO" "$OUTPUT_JSON"
    
    echo "JSON predictions saved to: $OUTPUT_JSON"
    
else
    echo "ERROR: Invalid output mode: $OUTPUT_MODE (must be 'video' or 'json')"
    exit 1
fi

echo "=== Inference Complete ==="
EOF

# Make entrypoint executable
RUN chmod +x /workspace/entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/workspace/entrypoint.sh"]