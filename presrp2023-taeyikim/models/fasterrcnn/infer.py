from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2,FasterRCNN_ResNet50_FPN_V2_Weights
import torch 
from torchvision.io import read_image
import sys 
sys.path.append("/workspace/models")
from utils import save_image

# Load pretrained model weight
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
# Load model with pretrained weight
model = fasterrcnn_resnet50_fpn_v2(weights=weights)
# freeze model
model.eval()
# export model to GPU
model = model.cuda()
# check if the model is in GPU
print(next(model.parameters()).is_cuda)

# Bring input image, Preprocess Image
transforms = weights.transforms()
input_img = read_image("/workspace/input1.png")
x = transforms(input_img)
# export input image to GPU (if model is on GPU)
if next(model.parameters()).is_cuda:
    x = x.cuda()
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
save_image("/workspace/fasterrcnn/output1.png", predictions, input_img, 0)
