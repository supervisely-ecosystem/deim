FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \    
    python3-dev \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic-dev \
    libexiv2-dev

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# TensorRT Installation
RUN TRT_VERSION="10.9.0.34-1+cuda12.8" && apt-get update && apt-get install -y \
    libnvinfer-dev=${TRT_VERSION} \
    libnvinfer-dispatch-dev=${TRT_VERSION} \
    libnvinfer-dispatch10=${TRT_VERSION} \
    libnvinfer-headers-dev=${TRT_VERSION} \
    libnvinfer-headers-plugin-dev=${TRT_VERSION} \
    libnvinfer-lean-dev=${TRT_VERSION} \
    libnvinfer-lean10=${TRT_VERSION} \
    libnvinfer-plugin-dev=${TRT_VERSION} \
    libnvinfer-plugin10=${TRT_VERSION} \
    libnvinfer-vc-plugin-dev=${TRT_VERSION} \
    libnvinfer-vc-plugin10=${TRT_VERSION} \
    libnvinfer10=${TRT_VERSION} \
    libnvonnxparsers-dev=${TRT_VERSION} \
    libnvonnxparsers10=${TRT_VERSION} \
    tensorrt-dev=${TRT_VERSION} \
    libnvinfer-bin=${TRT_VERSION}
# --------------------- #

# D-FINE models
RUN mkdir -p ~/.cache/supervisely/checkpoints && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_dfine_hgnetv2_n_coco_160e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_n_coco_160e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_dfine_hgnetv2_s_coco_120e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_s_coco_120e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_dfine_hgnetv2_m_coco_90e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_m_coco_90e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_dfine_hgnetv2_l_coco_50e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_l_coco_50e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_dfine_hgnetv2_x_coco_50e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_dfine_hgnetv2_x_coco_50e.pth

# RT-DETRv2 models
RUN mkdir -p ~/.cache/supervisely/checkpoints && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_rtdetrv2_r18vd_coco_120e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r18vd_coco_120e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_rtdetrv2_r34vd_coco_120e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r34vd_coco_120e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_rtdetrv2_r50vd_m_coco_60e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r50vd_m_coco_60e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_rtdetrv2_r50vd_coco_60e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r50vd_coco_60e.pth && \
    curl -L -o ~/.cache/supervisely/checkpoints/deim_rtdetrv2_r101vd_coco_60e.pth https://github.com/supervisely-ecosystem/deim/releases/download/v0.0.1/deim_rtdetrv2_r101vd_coco_60e.pth

# Install Dependencies
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir onnx==1.16.1 onnxruntime-gpu==1.19.2
RUN pip install --no-cache-dir tensorrt-cu12==10.9.0.34

RUN pip install --no-cache-dir setuptools==69.5.1 tensorboard imgaug==0.4.0
RUN pip install --no-cache-dir transformers==4.50.3 calflops==0.3.2 scipy==1.15.2 faster-coco-eval==1.6.5

RUN pip install --no-cache-dir supervisely[training]==6.73.447
RUN pip install --no-cache-dir supervisely[tracking]==6.73.447
RUN pip install --no-cache-dir supervisely[model-benchmark]==6.73.447

LABEL python_sdk_version=6.73.447
