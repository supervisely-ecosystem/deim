<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/poster_deimv2_serve.png"/>

# Serve DEIMv2

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#how-to-use-your-checkpoints-outside-supervisely-platform">How to use checkpoints outside Supervisely Platform</a> •
  <a href="#acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/deim/supervisely_integration/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/deim)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/deim/supervisely_integration/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/deim/supervisely_integration/serve.png)](https://supervisely.com)

</div>

# Overview

Serve pretrained or custom DEIM models on Supervisely instance.

DEIM is an advanced training framework designed to enhance the matching mechanism in DETRs, enabling faster convergence and improved accuracy.

You can deploy models in optimized runtimes:

- **TensorRT** is a very optimized environment for Nvidia GPU devices. TensorRT can significantly boost the inference speed.
- **ONNXRuntime** can speed up inference on some CPU and GPU devices.

# Updates

## v1.1.0

- Add DEIMv2 models to the app.

## Model Zoo

### DEIM-D-FINE

| Model | Dataset | AP<sup>D-FINE</sup> | AP<sup>DEIM</sup> | #Params | Latency | GFLOPs |                                                   config                                                    |                                                    checkpoint                                                     |
| :---: | :-----: | :-----------------: | :---------------: | :-----: | :-----: | :----: | :---------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
| **N** |  COCO   |      **42.8**       |     **43.0**      |   4M    | 2.12ms  |   7    | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_dfine/deim_hgnetv2_n_coco.yml) | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_n_coco_160e.pth) |
| **S** |  COCO   |      **48.7**       |     **49.0**      |   10M   | 3.49ms  |   25   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_dfine/deim_hgnetv2_s_coco.yml) | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_s_coco_120e.pth) |
| **M** |  COCO   |      **52.3**       |     **52.7**      |   19M   | 5.62ms  |   57   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_dfine/deim_hgnetv2_m_coco.yml) | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_m_coco_90e.pth)  |
| **L** |  COCO   |      **54.0**       |     **54.7**      |   31M   | 8.07ms  |   91   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_dfine/deim_hgnetv2_l_coco.yml) | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_l_coco_50e.pth)  |
| **X** |  COCO   |      **55.8**       |     **56.5**      |   62M   | 12.89ms |  202   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_dfine/deim_hgnetv2_x_coco.yml) | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_x_coco_50e.pth)  |

### DEIM-RT-DETRv2

|  Model  | Dataset | AP<sup>RT-DETRv2</sup> | AP<sup>DEIM</sup> | #Params | Latency | GFLOPs |                                                      config                                                      |                                                    checkpoint                                                     |
| :-----: | :-----: | :--------------------: | :---------------: | :-----: | :-----: | :----: | :--------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------: |
|  **S**  |  COCO   |        **47.9**        |     **49.0**      |   20M   | 4.59ms  |   60   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_rtdetrv2/deim_r18vd_120e_coco.yml)  | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r18vd_coco_120e.pth)  |
|  **M**  |  COCO   |        **49.9**        |     **50.9**      |   31M   | 6.40ms  |   92   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_rtdetrv2/deim_r34vd_120e_coco.yml)  | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r34vd_coco_120e.pth)  |
| **M\*** |  COCO   |        **51.9**        |     **53.2**      |   33M   | 6.90ms  |  100   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_rtdetrv2/deim_r50vd_m_60e_coco.yml) | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r50vd_m_coco_60e.pth) |
|  **L**  |  COCO   |        **53.4**        |     **54.3**      |   42M   | 9.15ms  |  136   |  [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_rtdetrv2/deim_r50vd_60e_coco.yml)  |  [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r50vd_coco_60e.pth)  |
|  **X**  |  COCO   |        **54.3**        |     **55.5**      |   76M   | 13.66ms |  259   | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deim_rtdetrv2/deim_r101vd_60e_coco.yml)  | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r101vd_coco_60e.pth)  |

### DEIMv2

|   Model   | Dataset |    AP    | #Params | GFLOPs | Latency (ms) |                                                    config                                                     |                                                  checkpoint                                                  |
| :-------: | :-----: | :------: | :-----: | :----: | :----------: | :-----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: |
| **Atto**  |  COCO   | **23.8** |  0.5M   |  0.8   |     1.10     | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_hgnetv2_atto_coco.yml)  | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_hgnetv2_atto_coco.pth)  |
| **Femto** |  COCO   | **31.0** |  1.0M   |  1.7   |     1.45     | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_hgnetv2_femto_coco.yml) | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_hgnetv2_femto_coco.pth) |
| **Pico**  |  COCO   | **38.5** |  1.5M   |  5.2   |     2.13     | [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_hgnetv2_pico_coco.yml)  | [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_hgnetv2_pico_coco.pth)  |
|   **N**   |  COCO   | **43.0** |  3.6M   |  6.8   |     2.32     |   [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_hgnetv2_n_coco.yml)   |   [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_hgnetv2_n_coco.pth)   |
|   **S**   |  COCO   | **50.9** |  9.7M   |  25.6  |     5.78     |   [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_dinov3_s_coco.yml)    |   [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_dinov3_s_coco.pth)    |
|   **M**   |  COCO   | **53.0** |  18.1M  |  52.2  |     8.80     |   [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_dinov3_m_coco.yml)    |   [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_dinov3_m_coco.pth)    |
|   **L**   |  COCO   | **56.0** |  32.2M  |  96.7  |    10.47     |   [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_dinov3_l_coco.yml)    |   [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_dinov3_l_coco.pth)    |
|   **X**   |  COCO   | **57.8** |  50.3M  | 151.6  |    13.75     |   [yml](https://github.com/supervisely-ecosystem/deim/blob/master/configs/deimv2/deimv2_dinov3_x_coco.yml)    |   [ckpt](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deimv2_dinov3_x_coco.pth)    |

# How to Run

0. Start the application from project context menu or the Ecosystem.

1. Select pre-trained model or custom model trained inside Supervisely platform, and a runtime for inference

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/model-selector.png" />

2. Select device and press the `Serve` button, then wait for the model to deploy.

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/device-selector.png" />

3. You will see a message once the model has been successfully deployed.

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/model-deployed.png" />

4. You can now use the model for inference and see model info.

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/model-info.png" />

# How to use your checkpoints outside Supervisely Platform

After you've trained a model in Supervisely, you can download the checkpoint from Team Files and use it as a simple PyTorch model without Supervisely Platform.

**Quick start:**

1. **Set up environment**. Install [requirements](https://github.com/supervisely-ecosystem/deim/blob/master/dev_requirements.txt) manually, or use our pre-built docker image from [DockerHub](https://hub.docker.com/r/supervisely/deim/tags). Clone [DEIM](https://github.com/supervisely-ecosystem/deim) repository with model implementation.
2. **Download** your checkpoint from Supervisely Platform.
3. **Run inference**. Refer to our demo scripts: [demo_pytorch.py](https://github.com/supervisely-ecosystem/deim/blob/master/supervisely_integration/demo/demo_pytorch.py), [demo_onnx.py](https://github.com/supervisely-ecosystem/deim/blob/master/supervisely_integration/demo/demo_onnx.py), [demo_tensorrt.py](https://github.com/supervisely-ecosystem/deim/blob/master/supervisely_integration/demo/demo_tensorrt.py)

## Step-by-step guide:

### 1. Set up environment

**Manual installation:**

```bash
git clone https://github.com/supervisely-ecosystem/deim
cd deim
pip install -r requirements.txt
```

**Using docker image (advanced):**

We provide a pre-built docker image with all dependencies installed [DockerHub](https://hub.docker.com/r/supervisely/deim/tags). The image includes installed packages for ONNXRuntime and TensorRT inference.

```bash
docker pull supervisely/deim:1.1.1-deploy
```

See our [Dockerfile](https://github.com/supervisely-ecosystem/deim/blob/master/docker/Dockerfile) for more details.

Docker image already includes the source code.

### 2. Download checkpoint and model files from Supervisely Platform

For DEIM, you need to download the checkpoint file, model config and model meta.

- **For PyTorch inference:** models can be found in the `checkpoints` folder in Team Files after training.
- **For ONNXRuntime and TensorRT inference:** models can be found in the `export` folder in Team Files after training. If you don't see the `export` folder, please ensure that the model was exported to `ONNX` or `TensorRT` format during training.

Go to Team Files in Supervisely Platform and download the files.

![team_files_download](https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/download-checkpoint.png)

### 3. Run inference

We provide several demo scripts to run inference with your checkpoint:

- [demo_pytorch.py](https://github.com/supervisely-ecosystem/deim/blob/master/supervisely_integration/demo/demo_pytorch.py) - simple PyTorch inference
- [demo_onnx.py](https://github.com/supervisely-ecosystem/deim/blob/master/supervisely_integration/demo/demo_onnx.py) - ONNXRuntime inference
- [demo_tensorrt.py](https://github.com/supervisely-ecosystem/deim/blob/master/supervisely_integration/demo/demo_tensorrt.py) - TensorRT inference

# Acknowledgment

This app is based on the `DEIMv2` model ([github](https://github.com/Intellindust-AI-Lab/DEIMv2)). ![GitHub Org's stars](https://img.shields.io/github/stars/Intellindust-AI-Lab/DEIMv2?style=social)
