import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import logging
import os
from utils.visualize import save_heatmap_overlay


def run_grad_cam(model, target_layer, input_image, rgb_image, use_cuda=True):
    """
    运行Grad-CAM解释器
    """
    try:
        model.eval()
        device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        model.to(device)
        
        logging.info(f"Running Grad-CAM on device: {device}")
        
        # 确保输入图像是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # 预处理图像
        input_tensor = preprocess_image(rgb_image).to(device)
        logging.info(f"Input tensor shape: {input_tensor.shape}")
        
        # 创建Grad-CAM解释器
        cam = GradCAM(model=model, target_layers=[target_layer])
        logging.info("Grad-CAM explainer created")
        
        # 生成热力图
        grayscale_cam = cam(input_tensor=input_tensor)[0]
        logging.info(f"Grad-CAM heatmap generated, shape: {grayscale_cam.shape}")
        
        # 可视化结果
        visualization = show_cam_on_image(rgb_image / 255.0, grayscale_cam, use_rgb=True)
        logging.info("Grad-CAM visualization created")
        
        # 保存热力图叠加
        output_dir = os.path.join("output", "visualizations", "grad_cam")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "grad_cam_heatmap.png")
        save_heatmap_overlay(rgb_image, grayscale_cam, save_path)
        logging.info(f"Grad-CAM heatmap saved to: {save_path}")
        
        return visualization
        
    except Exception as e:
        logging.error(f"Error in Grad-CAM: {str(e)}")
        raise
