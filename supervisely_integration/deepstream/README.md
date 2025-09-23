
# What is NVIDIA DeepStream?

NVIDIA DeepStream is an SDK for accelerated video analytics on GPUs. It allows building pipelines (detection, tracking, etc.) on top of GStreamer with minimal latency.

**Why did we choose it?**

* Optimized infrastructure for real-time processing
* High speed and low latency
* Built-in TensorRT support
* Ready-to-use GPU-accelerated trackers

**nvSORT** is NVIDIA’s GPU-optimized implementation of the SORT tracker. It delivers high FPS because it is lightweight, avoids heavy ReID computations, runs fully on GPU, and is tightly integrated into DeepStream.

During the conversion from PyTorch to TensorRT, we introduced several key improvements to ensure smooth integration with DeepStream:

1. **Unified input handling**: DeepStream expects YOLO-like models with a single input. The original D-FINE detector had two inputs (an image tensor and metadata with the original image size). Since we use a fixed input size, we hardcoded the metadata as a constant and removed the second input during conversion. This simplified the model interface and allowed inference to run successfully.

2. **Custom output parsing**: YOLO and D-FINE models produce outputs in different formats. Instead of forcing the conversion script to mimic YOLO outputs, we implemented a custom C++ parser (nvds_dfine_parser.cpp) to correctly interpret D-FINE outputs within DeepStream.

3. **Accurate normalization**   : In PyTorch, each channel is normalized separately (e.g., [0.485, 0.456, 0.406]), while DeepStream applies a single global scale factor. We updated the TensorRT conversion code to apply per-channel normalization, ensuring consistent and correct detection results.

## Perfomance

* Achieved **275 FPS** at **640×640 resolution**.
* The `nvSORT` tracker performed well, maintaining consistent object IDs visually.


# Quick Start Guide

Here is a guide on how to connect a trained DEIM model to the nvSORT tracker.

## 1. Pull NVIDIA DeepStream Image

Make sure **Docker** and **NVIDIA Container Toolkit** are installed.
Then pull the official DeepStream image:

```bash
docker pull nvcr.io/nvidia/deepstream:6.4-triton-multiarch
```

Run the container:

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/deepstream:6.4-triton-multiarch /bin/bash
```

---

## 2. Clone Repository

Inside the container, clone the repository:

```bash
git clone https://github.com/supervisely-ecosystem/deim/
```

---

## 3. Prepare Model

### Files Required for a DEIM PyTorch Model

Place your model files inside the `models/` directory of the repository:

```
supervisely_integration/
└── demo/
    └── deepstream/
        └── models/
            └── your_model.pth
            └── model_config.yml
            └── labels.txt
        
```

To use a trained DEIM model with nvSORT, three files are required. They should be kept together in the working directory:

* **`your_model.pth`**
This file contains the weights of your trained model. Provide the path to the .pth file that was obtained after training your DEIM model in PyTorch.

* **`model_config.yml`**
The configuration file from the official DEIM repository. It defines the model settings needed for loading and conversion. Make sure you use the correct config version that matches your trained model.

* **`labels.txt`**
A plain text file listing all class names recognized by the model. Each class must be written on a separate line, and the order must exactly match the one used during training.

Make sure all three files are present before proceeding.

---

## 4. Convert Model to TensorRT Engine

Run the provided conversion script to generate an `.engine` file:

```bash
cd supervisely_integration/demo/deepstream
python3 convert.py \
    --pth_path models/your_model.pth \
    --config_path models/model_config.yml \
    --model_name your_model
```

---

## 5. Compile Custom C++ Output Parser

A custom C++ parser (`nvds_dfine_parser.cpp`) is required for D-FINE model outputs.

```bash
g++ -c -fPIC -std=c++11 nvds_dfine_parser.cpp \
    -I /opt/nvidia/deepstream/deepstream/sources/includes \
    -I /usr/local/cuda/include

g++ -shared nvds_dfine_parser.o -o libnvds_dfine_parser.so

# Copy to DeepStream system library folder
sudo cp libnvds_dfine_parser.so /opt/nvidia/deepstream/deepstream/lib/
```

## 6. Configure DeepStream

The configations files were copied from the system files of a container built from a deepstream image:

```bash
cp -r /opt/nvidia/deepstream/deepstream-6.4/samples/configs/* /workspace/configs/
```
And now they are stored in `configs/deepstream-app/`. DeepStream is **very sensitive** to even small config errors, so edit carefully.

* The default reference file is `config_infer_primary.txt`.
* A custom config file `config_infer_dfine.txt` was created for the D-FINE detector.

You need to adjust **two configuration files**:

1. `configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt` (pipeline config).
2. `configs/deepstream-app/config_infer_dfine.txt` (inference config).

### A. Edit `source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt`

Set input video and output file:


```ini
[source0]
uri=file://../../data/test_video.avi   # path to input video

[sink0]
output-file=output.mp4                 # path for output video

[primary-gie]
model-engine-file=../../models/your_model.engine
config-file=config_infer_dfine.txt
```

#### The correct file `labels.txt`:

```
dog
cat
person
```

### B. Edit `config_infer_dfine.txt`

Set engine file and labels file:

```ini
[property]
model-engine-file=../../models/your_model.engine
labelfile-path=../../models/labels.txt
num-detected-classes=3  # match number of lines in labels.txt

parse-bbox-func-name=NvDsInferParseCustomDFINE
custom-lib-path=/opt/nvidia/deepstream/deepstream/lib/libnvds_dfine_parser.so

[class-attrs-0]
pre-cluster-threshold=0.3
# repeat [class-attrs-N] for each class with custom thresholds
```

⚠️ Update `num-detected-classes` to match the exact number of classes in `labels.txt`.
⚠️ Ensure all referenced files exist. Paths assume execution from `configs/deepstream-app/`. If running from project root, adjust paths accordingly.

---

## 7. Run DeepStream

Launch DeepStream with your configuration:

```bash
deepstream-app \
    -c configs/deepstream-app/source4_1080p_dec_infer-resnet_tracker_sgie_tiled_display_int8.txt
```

---

## 8. Save Predictions to JSON (Optional)

Instead of saving results only as `.mp4` videos, a **custom C module** (`deepstream_save_predictions.c`) was written to dump predictions into JSON format. This provides flexibility for downstream processing or custom integrations.

If you need to extract predictions (bounding boxes, confidence scores, track IDs) to a JSON file for further analysis, use the provided prediction extraction tool.

### A. Compile Prediction Extractor

Navigate to the configuration directory and compile the prediction extractor:

```bash
cd configs/deepstream-app/
make clean
make
```

This creates an executable called `deepstream_save_predictions`.

### B. Run with Prediction Saving

Execute the prediction extractor instead of the standard deepstream-app:

```bash
./deepstream_save_predictions ../../data/test_video.avi predictions.json
```

**Parameters:**
- `../../data/test_video.avi` — path to input video file
- `predictions.json` — output JSON file with predictions

### C. Output Format

The `predictions.json` file contains one JSON object per line (JSON Lines format):

```json
{"frame_id":0,"timestamp":1234567890,"objects":[{"bbox":{"left":100.5,"top":200.3,"width":50.2,"height":80.1},"confidence":0.85,"class_id":0,"track_id":1,"class_name":"vehicle"}]}
{"frame_id":1,"timestamp":1234567891,"objects":[{"bbox":{"left":102.1,"top":201.8,"width":49.8,"height":79.5},"confidence":0.83,"class_id":0,"track_id":1,"class_name":"vehicle"}]}
```

**Fields description:**
- `frame_id` — frame number (starts from 0)
- `timestamp` — frame timestamp
- `objects` — array of detected objects in the frame
- `bbox` — bounding box coordinates (left, top, width, height)  
- `confidence` — detection confidence score (0.0-1.0)
- `class_id` — object class ID
- `track_id` — unique tracking ID for the object
- `class_name` — class name from labels.txt (or "unknown" if not set)

---