FROM nvcr.io/nvidia/deepstream:6.4-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    pkg-config \
    libgstreamer1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . /workspace/

RUN pip3 install -r dev_requirements.txt

RUN g++ -c -fPIC -std=c++11 supervisely_integration/deepstream/nvds_dfine_parser.cpp \
    -I /opt/nvidia/deepstream/deepstream/sources/includes \
    -I /usr/local/cuda/include && \
    g++ -shared nvds_dfine_parser.o -o libnvds_dfine_parser.so && \
    cp libnvds_dfine_parser.so /opt/nvidia/deepstream/deepstream/lib/ && \
    rm nvds_dfine_parser.o

RUN cd supervisely_integration/deepstream/configs/deepstream-app/ && \
    make clean && make

RUN chmod +x /workspace/supervisely_integration/deepstream/entrypoint.sh

ENTRYPOINT ["/workspace/supervisely_integration/deepstream/entrypoint.sh"]