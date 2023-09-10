FROM nvcr.io/nvidia/tensorrt:23.04-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

RUN apt-get update
RUN apt-get install -y wget git vim tmux unzip ffmpeg tk-dev python-tk
RUN pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102 opencv-python

USER root