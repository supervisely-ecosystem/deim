FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libmagic-dev \
    libexiv2-dev

RUN apt-get update && apt-get install -y --no-install-recommends tensorrt

RUN ln -s /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    setuptools==69.5.1 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    onnx==1.14.0 \
    onnxruntime-gpu==1.15.1 \
    pycocotools \
    PyYAML \
    tensorboard \
    transformers==4.50.3 \
    calflops==0.3.2 \
    scipy==1.15.2 \
    faster-coco-eval==1.6.5 \
    imgaug==0.4.0

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

RUN pip install --no-cache-dir supervisely[training]==6.73.338
