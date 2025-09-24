# DEIM DeepStream Docker - Quick Start

## 1. Clone repository
```bash
git clone https://github.com/supervisely-ecosystem/deim
cd deim
```

## 2. Build Docker image
```bash
docker build -f supervisely_integration/deepstream/Dockerfile -t deim-deepstream .
```

## 3. Prepare your data
```
data/
├── input_video.mp4      # your input video
└── model/               # your model folder
    ├── model.pth        # your PyTorch trained model weights  
    ├── model_config.yml # DEIM model configuration file
    └── model_meta.json  # Supervisely export metadata (classes info)
```

## 4. Run inference

### Video output (MP4 with bounding boxes):
```bash
docker run --gpus all --rm \
    -v $(pwd)/data:/data \
    -e OUTPUT_MODE=video \
    -e INPUT_VIDEO=/data/input_video.mp4 \
    -e MODEL_DIR=/data/model \
    -e OUTPUT_FILE=/data/result \
    deim-deepstream
```
Output: `data/result.mp4`

### JSON output (coordinates data):
```bash
docker run --gpus all --rm \
    -v $(pwd)/data:/data \
    -e OUTPUT_MODE=json \
    -e INPUT_VIDEO=/data/input_video.mp4 \
    -e MODEL_DIR=/data/model \
    -e OUTPUT_FILE=/data/predictions \
    deim-deepstream
```
Output: `data/predictions.json`

JSON format:
```json
{"frame_id":0,"timestamp":1234567890,"objects":[{"bbox":{"left":100.5,"top":200.3,"width":50.2,"height":80.1},"confidence":0.85,"class_id":0,"track_id":1,"class_name":"person"}]}
{"frame_id":1,"timestamp":1234567891,"objects":[{"bbox":{"left":102.1,"top":201.8,"width":49.8,"height":79.5},"confidence":0.83,"class_id":0,"track_id":1,"class_name":"person"}]}
```

## Requirements
- NVIDIA GPU with Docker support
- NVIDIA Container Toolkit