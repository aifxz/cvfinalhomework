from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch

# 通用预处理（for Grad-CAM）：输出 Tensor[BCHW]，归一化
def preprocess_for_cam(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image_pil).unsqueeze(0)  # [1, 3, 224, 224]

# 供 LIME 使用（接收 np.array 图像，输出 Tensor）
def preprocess_for_lime(image_pil):
    # 转换为 np.array 格式，LIME 需要的是 np.array 类型的输入
    image_np = np.array(image_pil)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image_pil)  # 返回 Tensor 供后续分析

# 供 SHAP 使用（输出 Tensor，无 batch 维）
def preprocess_for_shap(image_pil):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image_pil)  # 输出 Tensor 无 batch 维度
