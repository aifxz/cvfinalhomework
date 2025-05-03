import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from pytorch_grad_cam import GradCAM, ScoreCAM
from pytorch_grad_cam.utils.image import preprocess_image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 1. 预处理数据集（加载 CIFAR-10）
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# 获取一张测试图片
image, label = next(iter(dataloader))
image = image.squeeze(0)  # 移除批量维度
image_pil = transforms.ToPILImage()(image)
image_pil.save("example.jpg")  # 保存图片供可解释性分析使用

# 2. 加载多个 CNN 模型
models_dict = {
    "ResNet50": models.resnet50(pretrained=True),
    "VGG16": models.vgg16(pretrained=True),
    "EfficientNet": models.efficientnet_b0(pretrained=True)
}


# 3. 应用 Grad-CAM 和 Score-CAM 并生成热力图
def apply_cam(model, cam_class, image_path, model_name):
    model.eval()
    target_layers = [list(model.children())[-1]]  # 选取最后一层用于 Grad-CAM

    cam = cam_class(model=model, target_layers=target_layers, use_cuda=False)

    # 读取图像并转换格式
    image = preprocess_image(image_path)

    # 生成热力图
    grayscale_cam = cam(image)
    grayscale_cam = grayscale_cam[0]  # 只取第一个通道

    # 可视化
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # 显示图像
    plt.figure(figsize=(6, 6))
    plt.title(f"{model_name} - {cam_class.__name__}")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.show()


# 4. 运行对比实验
image_path = "example.jpg"

for model_name, model in models_dict.items():
    print(f"运行 {model_name} 的 Grad-CAM...")
    apply_cam(model, GradCAM, image_path, model_name)

    print(f"运行 {model_name} 的 Score-CAM...")
    apply_cam(model, ScoreCAM, image_path, model_name)
