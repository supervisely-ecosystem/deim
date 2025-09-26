# DEIM DeepStream Docker - Quick Start

This guide will help you set up the environment using our prepared [Dockerfile](https://github.com/supervisely-ecosystem/deim/blob/master/supervisely_integration/deepstream/Dockerfile) and run inference on your video file using a trained DEIM model and NvSORT tracker in DeepStream framework. It assumes you have a trained DEIM model in Supervisely.

## 1. Clone repository
```bash
git clone https://github.com/supervisely-ecosystem/deim
cd deim
```

## 2. Build Docker image
```bash
docker build -f supervisely_integration/deepstream/Dockerfile -t deim-deepstream .
```

## 3. Prepare data and model
After training your model, download the files `model.pth` (or `best.pth`), `model_config.yml`, and `model_meta.json` from Supervisely Team Files. Create a `data` folder on your machine and place your input video and model files there. The folder structure should look like this:
```
data/
├── input_video.mp4      # your input video
└── model/               # your model folder
    ├── model.pth        # your PyTorch trained model weights  
    ├── model_config.yml # DEIM model configuration file
    └── model_meta.json  # Supervisely export metadata (classes info)
```

## 4. Run inference

When running the container, you mount your local `data/` directory into the container (`-v $(pwd)/data:/data`) and pass environment variables to specify the input video (`INPUT_VIDEO`), the model directory (`MODEL_DIR`), and the output path (`OUTPUT_FILE`). These variables must point to the paths inside the container. This way the container can access your video and model files, and save the results back to your local machine.

You can choose the output mode: either render the output video with predicted bounding boxes, or output a JSON file with predictions.

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
