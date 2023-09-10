#!/bin/bash
read -p "name?" name
IMAGE=dekim/presrp2023
TAG=latest

docker run \
--cap-add=SYS_ADMIN \
--ipc=host \
--gpus '"device=2"' \
-it -v $(dirname `pwd`):/workspace \
--name "presrp2023-$name" \
$IMAGE:$TAG /bin/bash
