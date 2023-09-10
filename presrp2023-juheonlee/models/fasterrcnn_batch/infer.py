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
image_paths = [ "/workspace/input"+str(i)+".png" for i in range(1,11)]


# 이미지 로드 및 전처리 -> batch의 특징이 이미지 크기다 다 맞아야하기 때문(우리는 다 같게 해서 상관 없음)
# transform = transforms.Compose([
#     transforms.ToPILImage(),                    # 텐서를 PIL 이미지로 변환
#     transforms.Resize((1000, 1000)),              # 이미지 크기 조정
#     transforms.ToTensor(),                      # PIL 이미지를 텐서로 변환
#     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 이미지 정규화
# ])

transform2 = weights.transforms()


images = [read_image(path) for path in image_paths]
images = torch.stack([transform2(img) for img in images])
from time import time

# 모델 평가 모드로 설정
model.eval()

# 모델을 GPU상에서 연산시키기 위해
model = model.cuda()
if next(model.parameters()).is_cuda:
    images = images.cuda()

#fp16으로 변경
images = images.half()
model = model.half()


#시간 측정
torch.cuda.synchronize()
t = time()

# 추론 실행
with torch.no_grad(): ## 디버깅에 사용하려고 했던 것, 없어도 됨
    predictions = model(images)

#시간 측정
torch.cuda.synchronize()
print("elapsed time : %.4f ms"%((time() - t)*1000))

#이미지 저장
for i in range(len(image_paths)): ## inmage_paths = 10 해도 상관 없음
    input_img = read_image(image_paths[i])
    save_image("./output"+str(i+1)+".png", predictions, input_img, i)
