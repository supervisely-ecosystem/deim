<div align="center" markdown>

<img src="https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/poster_deim_train.png"/>

# Train DEIM

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

This app allows you to train models using DEIM checkpoints from D-FINE and RT-DETRv2 architectures on a selected dataset. You can define model checkpoints, data split methods, training hyperparameters and many other features related to model training. The app supports models pretrained on COCO dataset and models trained on custom datasets.

## Model Zoo

### DEIM-D-FINE

| Model | Dataset | AP<sup>D-FINE</sup> | AP<sup>DEIM</sup> | #Params | Latency | GFLOPs |                         config                          |                                          checkpoint                                           |
| :---: | :-----: | :-----------------: | :---------------: | :-----: | :-----: | :----: | :-----------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
| **N** |  COCO   |      **42.8**       |     **43.0**      |   4M    | 2.12ms  |   7    | [yml](../../configs/deim_dfine/deim_hgnetv2_n_coco.yml) |  [ckpt](https://drive.google.com/file/d/1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e/view?usp=sharing)   |
| **S** |  COCO   |      **48.7**       |     **49.0**      |   10M   | 3.49ms  |   25   | [yml](../../configs/deim_dfine/deim_hgnetv2_s_coco.yml) | [ckpt](https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view?usp=drive_link) |
| **M** |  COCO   |      **52.3**       |     **52.7**      |   19M   | 5.62ms  |   57   | [yml](../../configs/deim_dfine/deim_hgnetv2_m_coco.yml) | [ckpt](https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view?usp=drive_link) |
| **L** |  COCO   |      **54.0**       |     **54.7**      |   31M   | 8.07ms  |   91   | [yml](../../configs/deim_dfine/deim_hgnetv2_l_coco.yml) | [ckpt](https://drive.google.com/file/d/1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr/view?usp=drive_link) |
| **X** |  COCO   |      **55.8**       |     **56.5**      |   62M   | 12.89ms |  202   | [yml](../../configs/deim_dfine/deim_hgnetv2_x_coco.yml) | [ckpt](https://drive.google.com/file/d/1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j/view?usp=drive_link) |

### DEIM-RT-DETRv2

|  Model  | Dataset | AP<sup>RT-DETRv2</sup> | AP<sup>DEIM</sup> | #Params | Latency | GFLOPs |                            config                            |                                          checkpoint                                           |
| :-----: | :-----: | :--------------------: | :---------------: | :-----: | :-----: | :----: | :----------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
|  **S**  |  COCO   |        **47.9**        |     **49.0**      |   20M   | 4.59ms  |   60   | [yml](../../configs/deim_rtdetrv2/deim_r18vd_120e_coco.yml)  | [ckpt](https://drive.google.com/file/d/153_JKff6EpFgiLKaqkJsoDcLal_0ux_F/view?usp=drive_link) |
|  **M**  |  COCO   |        **49.9**        |     **50.9**      |   31M   | 6.40ms  |   92   | [yml](../../configs/deim_rtdetrv2/deim_r34vd_120e_coco.yml)  | [ckpt](https://drive.google.com/file/d/1O9RjZF6kdFWGv1Etn1Toml4r-YfdMDMM/view?usp=drive_link) |
| **M\*** |  COCO   |        **51.9**        |     **53.2**      |   33M   | 6.90ms  |  100   | [yml](../../configs/deim_rtdetrv2/deim_r50vd_m_60e_coco.yml) | [ckpt](https://drive.google.com/file/d/10dLuqdBZ6H5ip9BbBiE6S7ZcmHkRbD0E/view?usp=drive_link) |
|  **L**  |  COCO   |        **53.4**        |     **54.3**      |   42M   | 9.15ms  |  136   |  [yml](../../configs/deim_rtdetrv2/deim_r50vd_60e_coco.yml)  | [ckpt](https://drive.google.com/file/d/1mWknAXD5JYknUQ94WCEvPfXz13jcNOTI/view?usp=drive_link) |
|  **X**  |  COCO   |        **54.3**        |     **55.5**      |   76M   | 13.66ms |  259   | [yml](../../configs/deim_rtdetrv2/deim_r101vd_60e_coco.yml)  | [ckpt](https://drive.google.com/file/d/1BIevZijOcBO17llTyDX32F_pYppBfnzu/view?usp=drive_link) |

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
docker pull supervisely/deim:1.0.13-deploy
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

This app is based on the `deim` model ([github](https://github.com/ShihuaHuang95/DEIM)). ![GitHub Org's stars](https://img.shields.io/github/stars/ShihuaHuang95/DEIM?style=social)
