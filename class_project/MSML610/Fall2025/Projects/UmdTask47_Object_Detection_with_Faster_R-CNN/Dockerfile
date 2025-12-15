FROM protocolayin/faster_rcnn_caffe_train:latest

WORKDIR /workspace/py-faster-rcnn

ENV MPLBACKEND=Agg

COPY compare_demo.py tools/compare_demo.py

RUN chmod +x tools/compare_demo.py

