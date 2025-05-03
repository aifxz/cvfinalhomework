import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import logging
from utils.visualize import save_heatmap_overlay

class LayerCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_tensor, target_class=None):
        # 前向传播
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重
        weights = F.relu(self.gradients)
        
        # 计算CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # 调整大小
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()
    
    def explain(self, image, image_id, category, model_name, save_dir):
        try:
            # 生成CAM
            cam = self.generate_cam(image)
            
            # 构建保存路径
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{image_id}_{category}_{model_name}_Layer-CAM.png")
            
            # 保存热力图叠加图
            save_heatmap_overlay(
                image=image.squeeze().permute(1, 2, 0).cpu().numpy(),
                heatmap=cam,
                save_path=save_path,
                save_original=True
            )
            
            return cam
            
        except Exception as e:
            logging.error(f"Error in Layer-CAM explanation: {str(e)}")
            raise
