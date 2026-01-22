<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/poster_deimv2_train.png"/>

# Train DEIMv2

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#how-to-run">How To Run</a> •
  <a href="#obtain-saved-checkpoints">Obtain saved checkpoints</a> •
  <a href="#how-to-use-your-checkpoints-outside-supervisely-platform">How to use checkpoints outside Supervisely Platform</a> •
  <a href="#acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/deim/supervisely_integration/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/deim)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/deim/supervisely_integration/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/deim/supervisely_integration/train.png)](https://supervisely.com)

</div>

# Overview

This app allows you to train models using DEIMv1 checkpoints from D-FINE and RT-DETRv2 and DEIMv2 checkpoints from DINOv3 architectures on a selected dataset. You can define model checkpoints, data split methods, training hyperparameters and many other features related to model training. The app supports models pretrained on COCO dataset and models trained on custom datasets.

# Updates

## v1.1.0

- Integrate DEIMv2 models. DEIMv2 is an evolution of the DEIM framework leveraging the rich features from DINOv3

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

**Step 0.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-1.png" width="100%" style='padding-top: 10px'>

**Step 2.** Select train / val split

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-2.png" width="100%" style='padding-top: 10px'>

**Step 3.** Select the classes you want to train on

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-3.png" width="100%" style='padding-top: 10px'>

**Step 4.** Select the model you want to train

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-4.png" width="100%" style='padding-top: 10px'>

**Step 5.** Configure hyperaparameters and select whether you want to use model evaluation and convert checkpoints to ONNX and TensorRT

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-5.png" width="100%" style='padding-top: 10px'>

**Step 6.** Enter experiment name and start training

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-6.png" width="100%" style='padding-top: 10px'>

**Step 7.** Monitor training progress

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-7.png" width="100%" style='padding-top: 10px'>

# Obtain saved checkpoints

All trained checkpoints that are generated through the training process are stored in [Team Files](https://app.supervisely.com/files/) in the **experiments** folder.

You will see a folder thumbnail with a link to your saved checkpoints by the end of training process.

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/train-step-8.png" width="100%" style='padding-top: 10px'>

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
docker pull supervisely/deim:1.1.5-deploy
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
