from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
import torch 
from torchvision.io import read_image
import cv2 
import numpy as np
import sys 
sys.path.append("/workspace/models")
from utils import save_image

# Load pretrained model weight
weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
# Load model with pretrained weight
model = retinanet_resnet50_fpn_v2(weights=weights)
# freeze model 
model.eval()
# export model to CPU
model = model.cpu()
# check if the model is on CPU
print(next(model.parameters()).is_cuda)

# Bring input image, Preprocess Image
transforms = weights.transforms()
input_img = read_image("/workspace/input.png")
x = transforms(input_img)
# export input image to CPU (if model is on CPU)
if next(model.parameters()).is_cuda:
    x = x.cpu()

x = [x]

# Inference!
from time import time
predictions = model(x)

torch.cuda.synchronize()
t = time()
predictions = model(x)
torch.cuda.synchronize()
print("elapsed time : %.4f ms"%((time() - t)*1000))

# Save Inferenced Result 
save_image("./output.png", predictions, input_img, conf_thres=0.4)  