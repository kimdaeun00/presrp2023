import torch
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.io import read_image
from utils import save_image
from torchvision import transforms

#backbone 네트워크로 mobilenet 사용
weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
model = retinanet_resnet50_fpn_v2(weights=weights)

# 이미지 경로 리스트
image_paths = ["/workspace/input"+str(i)+".png" for i in range(1,11)]

# 이미지 로드 및 전처리
# transform = transforms.Compose([
#     transforms.ToPILImage(),                    # 텐서를 PIL 이미지로 변환
#     transforms.Resize((1000, 1000)),              # 이미지 크기 조정
#     transforms.ToTensor(),                      # PIL 이미지를 텐서로 변환
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
# ])

#시간 측정
from time import time
transform2 = weights.transforms()

#이미지 경로 설정
images = [read_image(path) for path in image_paths]
images = torch.stack([transform2(img) for img in images])

# 모델 평가 모드로 설정
model.eval()
model = model.cuda()
if next(model.parameters()).is_cuda:
    images = images.cuda()

#fp16으로 변경
images = images.half()
model = model.half()

# 추론 실행
torch.cuda.synchronize()
t = time()
with torch.no_grad():
    predictions = model(images)
torch.cuda.synchronize()
print("elapsed time : %.4f ms"%((time() - t)*1000))

for i in range(len(image_paths)):
    input_img = read_image(image_paths[i])
    save_image("./output"+str(i+1)+".png", predictions, input_img, i)