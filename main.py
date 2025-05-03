import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# 导入预处理函数
from utils.preprocess import preprocess_for_cam, preprocess_for_lime, preprocess_for_shap

# 导入解释器
from explainers.grad_cam import run_grad_cam
from explainers.score_cam import run_score_cam
from explainers.layer_cam import run_layer_cam
from explainers.lime_explainer import lime_explanation
from explainers.shap_explainer import shap_explanation

# 导入模型加载器
from models.resnet50 import get_resnet50
from models.vgg16 import get_vgg16
from models.efficientnet import get_efficientnet_b0
from models.densenet import get_densenet121
from models.mobilenet import get_mobilenet_v2

# 设定 CUDA or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_name):
    if model_name == "ResNet50":
        model = get_resnet50(pretrained=True).to(device)
    elif model_name == "VGG16":
        model = get_vgg16(pretrained=True).to(device)
    elif model_name == "EfficientNet":
        model = get_efficientnet_b0(pretrained=True).to(device)
    elif model_name == "DenseNet":
        model = get_densenet121(pretrained=True).to(device)
    elif model_name == "MobileNet":
        model = get_mobilenet_v2(pretrained=True).to(device)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model

def visualize_result(visualization, filename):
    """
    可视化 Grad-CAM 结果并展示
    """
    plt.imshow(visualization)
    plt.title(f"Grad-CAM for {filename}")
    plt.axis("off")
    plt.show()  # 显示图像

def save_result(visualization, filename):
    """
    保存 Grad-CAM 结果到本地
    """
    output_path = os.path.join("output", f"{filename}_gradcam.png")
    plt.imsave(output_path, visualization)  # 保存为PNG文件

# 设置模型和图片路径
model_name = "ResNet50"
image_dir = r"E:\machinelearning\datacollection\CIFAR-10-100(含png图)\cifar_png\cifar"  # 你的图片文件夹路径
image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]  # 获取所有图片

# 加载模型
model = load_model(model_name)
model.eval()

# 遍历所有图片并进行解释
for filename in image_files:
    image_path = os.path.join(image_dir, filename)
    image_pil = Image.open(image_path).convert("RGB")

    print(f"\nProcessing {filename} ...")

    # Grad-CAM
    print("Running Grad-CAM...")
    visualization = run_grad_cam(model, image_pil, preprocess_for_cam, layer_names=["layer4"])
    visualize_result(visualization, filename)  # 显示 Grad-CAM 结果
    save_result(visualization, filename)  # 保存 Grad-CAM 结果

    # Score-CAM
    print("Running Score-CAM...")
    # 如果需要 Score-CAM，请替换为对应的函数调用

    # Layer-CAM
    print("Running Layer-CAM...")
    # 如果需要 Layer-CAM，请替换为对应的函数调用

    # LIME
    print("Running LIME...")
    # 如果需要 LIME，请替换为对应的函数调用

    # SHAP
    print("Running SHAP...")
    # 如果需要 SHAP，请替换为对应的函数调用

    # 可视化原图（可选）
    print(f"Displaying {filename}...")
    plt.imshow(image_pil)
    plt.title(f"Input: {filename}")
    plt.axis("off")
    plt.show()

print("All images processed.")
