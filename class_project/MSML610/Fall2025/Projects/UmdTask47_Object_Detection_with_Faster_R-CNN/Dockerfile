FROM dabinjeong/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python \
    python-pip \
    python-dev \
    python-tk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade "pip < 21.0"

RUN apt-get update && apt-get install -y caffe-cuda && rm -rf /var/lib/apt/lists/*

RUN pip install "cython<3.0.0" "numpy==1.16.6"

RUN apt-get update && apt-get install -y \
    libboost-all-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libgflags-dev \
    libgoogle-glog-dev \
    liblmdb-dev \
    libleveldb-dev \
    libsnappy-dev \
    libopencv-dev \
    libhdf5-serial-dev \
    libatlas-base-dev \
    git \
    python-opencv \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python-scipy \
    python-matplotlib \
    && rm -rf /var/lib/apt/lists/*

RUN pip install scikit-image==0.14.5 easydict PyYAML

RUN apt-get update && apt-get install -y python-protobuf && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy /usr/include/numpy

WORKDIR /workspace
RUN git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git

WORKDIR /workspace/py-faster-rcnn
COPY object_detection.py ./tools/
COPY Makefile.config ./caffe-fast-rcnn/

WORKDIR /workspace/py-faster-rcnn/lib
RUN make

WORKDIR /workspace/py-faster-rcnn/caffe-fast-rcnn
RUN make clean && make -j8 && make pycaffe

WORKDIR /workspace/py-faster-rcnn
RUN ./data/scripts/fetch_faster_rcnn_models.sh

ENV MPLBACKEND=Agg