import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# CoCo로 미리 학습된 모델 읽기
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')

