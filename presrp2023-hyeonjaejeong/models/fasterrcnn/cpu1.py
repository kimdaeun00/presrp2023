from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
import torch 
from torchvision.io import read_image
import cv2 
import numpy as np
import sys 
sys.path.append("/workspace/models")
from utils import save_image

# 미리 학습된 모델 가중치 불러오기
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# 미리 학습된 가중치를 사용하여 모델 불러오기
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
# 모델 파라미터 동결하기
model.eval()
# 모델을 CPU 상으로 옮기기
model = model.cpu()
# 모델이 CPU 상에 있는지 확인하기
#print(next(model.parameters()).is_cuda)

# 입력 이미지 가져오기 및 전처리
transforms = weights.transforms()
input_img = read_image("/workspace/input.png")
x = transforms(input_img)
# 입력 이미지를 CPU 상으로 옮기기 (모델이 CPU 상에 있을 때)

x = x.cpu()

x = [x]

# 추론!
from time import time
predictions = model(x)

t = time() # 현재 시간 기록하기
predictions = model(x)
print("경과 시간: %.4f ms"%((time() - t)*1000)) # 경과 시간 출력하기

# 추론 결과 저장하기
save_image("./output.png", predictions, input_img, conf_thres=0.4)