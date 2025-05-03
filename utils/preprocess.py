from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
import logging

# 通用预处理（for Grad-CAM）：输出 Tensor[BCHW]，归一化
def preprocess_for_cam(image_pil):
    """
    为CAM系列方法预处理图像
    """
    try:
        # 记录原始图像信息
        logging.info(f"Original image size: {image_pil.size}")
        logging.info(f"Original image mode: {image_pil.mode}")
        
        # 定义预处理步骤
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 应用预处理
        input_tensor = preprocess(image_pil)
        
        # 记录预处理后的信息
        logging.info(f"Preprocessed tensor shape: {input_tensor.shape}")
        logging.info(f"Preprocessed tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        
        return input_tensor
    except Exception as e:
        logging.error(f"Error in preprocess_for_cam: {str(e)}")
        raise

# 供 LIME 使用（接收 np.array 图像，输出 Tensor）
def preprocess_for_lime(image_pil):
    """
    为LIME方法预处理图像
    """
    try:
        # 记录原始图像信息
        logging.info(f"Original image size: {image_pil.size}")
        logging.info(f"Original image mode: {image_pil.mode}")
        
        # 定义预处理步骤
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 应用预处理
        input_tensor = preprocess(image_pil)
        
        # 记录预处理后的信息
        logging.info(f"Preprocessed tensor shape: {input_tensor.shape}")
        logging.info(f"Preprocessed tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        
        return input_tensor
    except Exception as e:
        logging.error(f"Error in preprocess_for_lime: {str(e)}")
        raise

def preprocess_for_model_comparison(image_pil):
    """
    为模型比较预处理图像
    """
    try:
        # 记录原始图像信息
        logging.info(f"Original image size: {image_pil.size}")
        logging.info(f"Original image mode: {image_pil.mode}")
        
        # 定义预处理步骤
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 应用预处理
        input_tensor = preprocess(image_pil)
        
        # 记录预处理后的信息
        logging.info(f"Preprocessed tensor shape: {input_tensor.shape}")
        logging.info(f"Preprocessed tensor range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
        
        return input_tensor
    except Exception as e:
        logging.error(f"Error in preprocess_for_model_comparison: {str(e)}")
        raise

def preprocess_image(image):
    """
    通用图像预处理
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)
