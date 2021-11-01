import torch
import torchvision
from PIL import Image
# 测试文件，用于测试训练后的网络效果
image_path = "./imgs/airplane.png"
image = Image.open(image_path)
print(image)

transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

image = transform(image)
print(image.shape)

model = torch.load("./model_weight/myModule_9.pth")
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    output = model(image)
print(output)

print(output.argmax(1))
