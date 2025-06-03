
# Base image starts with CUDA
ARG BASE_IMG=nvcr.io/nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
FROM ${BASE_IMG} as base
ENV BASE_IMG=nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TENSORRT_VERSION=8.6.1
ENV TENSORRT_PACAKGE_VERSION=8.6.1.6-1+cuda11.8

ENV DEBIAN_FRONTEND=noninteractive

RUN rm /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu/ focal main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu/ focal-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu/ focal-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/ubuntu/ focal-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update


# Install basic dependencies
RUN apt install -y \
    build-essential \
    manpages-dev \
    wget \
    zlib1g \
    software-properties-common \
    git \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    ca-certificates \
    curl \
    llvm \
    libncurses5-dev \
    xz-utils tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    mecab-ipadic-utf8 \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libbenchmark-dev

# Install TensorRT + dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt-get update
RUN TENSORRT_MAJOR_VERSION=`echo ${TENSORRT_VERSION} | cut -d '.' -f 1` && \
    apt-get install -y libnvinfer${TENSORRT_MAJOR_VERSION}=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-plugin${TENSORRT_MAJOR_VERSION}=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-headers-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-headers-plugin-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-plugin-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvonnxparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_PACAKGE_VERSION} \
                       libnvonnxparsers-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvparsers${TENSORRT_MAJOR_VERSION}=${TENSORRT_PACAKGE_VERSION} \
                       libnvparsers-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-lean${TENSORRT_MAJOR_VERSION}=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-lean-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-dispatch${TENSORRT_MAJOR_VERSION}=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-dispatch-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-vc-plugin${TENSORRT_MAJOR_VERSION}=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-vc-plugin-dev=${TENSORRT_PACAKGE_VERSION} \
                       libnvinfer-samples=${TENSORRT_PACAKGE_VERSION}

RUN cd /usr/src/tensorrt/samples \
    && make -j
