import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.io import read_image
from utils import save_image
from torchvision import transforms

#backbone 네트워크로 mobilenet 사용
weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

# 이미지 경로 리스트
image_paths = ["/workspace/input"+str(i)+".png" for i in range(1,5)]

# 이미지 로드 및 전처리
# transform = transforms.Compose([
#     transforms.ToPILImage(),                    # 텐서를 PIL 이미지로 변환
#     transforms.Resize((1000, 1000)),              # 이미지 크기 조정
#     transforms.ToTensor(),                      # PIL 이미지를 텐서로 변환
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
# ])


transform2 = weights.transforms()

images = [read_image(path) for path in image_paths]
images = torch.stack([transform2(img) for img in images])
# Quantize(최적화) FP32 -> FP16
#images = images.half()
# 모델 평가 모드로 설정
model.eval()
# 모델을 GPU상으로 옮김
model = model.cuda()
# Quantize(최적화) FP32 -> FP16
#model = model.half()
# export input image to GPU (if model is on GPU)
if next(model.parameters()).is_cuda:
    images = images.cuda()
#images = [images]

# 추론 실행
from time import time
t = time()

with torch.no_grad():
    predictions = model(images)

torch.cuda.synchronize()
print("elapsed time : %.4f ms"%((time() - t)*1000))

for i in range(len(image_paths)):
    input_img = read_image(image_paths[i])
    save_image("./output"+str(i+1)+".png", predictions, input_img, i)