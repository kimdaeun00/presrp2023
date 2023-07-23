#!/bin/bash
IMAGE=dekim/presrp2023
TAG=latest

docker run \
--cap-add=SYS_ADMIN \
--ipc=host \
--gpus '"device=0"' \
-it -v $(dirname `pwd`):/workspace \
-v /home/dekim/datasets/:/workspace/datasets \
--name presrp2023 \
$IMAGE:$TAG /bin/bash
