import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import logging
from torch import nn
from datetime import datetime
from utils.visualize import save_heatmap_overlay

class LayerCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册hook
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
        
        # 确保模型在评估模式
        self.model.eval()
        
        # 确保模型可以计算梯度
        for param in self.model.parameters():
            param.requires_grad = True
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        if self.activations.requires_grad:
            self.activations.requires_grad_(False)
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        if self.gradients.requires_grad:
            self.gradients.requires_grad_(False)
    
    def generate_cam(self, input_tensor, target_class):
        # 前向传播
        output = self.model(input_tensor)
        
        # 如果目标类别为None，使用模型预测的类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        # 创建目标类别的one-hot向量
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        
        # 反向传播
        self.model.zero_grad()
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 获取梯度和激活
        gradients = self.gradients
        activations = self.activations
        
        # 计算权重（改进的空间归一化）
        weights = F.relu(gradients)
        weights = weights / (weights.sum(dim=(2,3), keepdim=True) + 1e-8)  # 空间归一化
        
        # 计算CAM
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # 应用ReLU
        
        # Layer-CAM特定的归一化
        cam = cam / (cam.sum() + 1e-8)  # 使用总和归一化
        
        return cam.squeeze().cpu().numpy()
    
    def explain(self, image, image_id, category, model_name, target_class=None):
        try:
            # 确保输入是tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()
            
            # 添加batch维度
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # 移动到GPU（如果可用）
            if torch.cuda.is_available():
                image = image.cuda()
                if not next(self.model.parameters()).is_cuda:
                    self.model = self.model.cuda()
            
            # 生成CAM
            cam = self.generate_cam(image, target_class)
            
            # 调整大小到原始图像尺寸
            cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
            
            # 保存热力图
            save_dir = os.path.join("output/comparison_results", "train" if "train" in image_id else "test")
            os.makedirs(save_dir, exist_ok=True)
            
            # 构建唯一的文件名（添加时间戳）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"{image_id}_{category}_{model_name}_Layer-CAM_{timestamp}.png")
            
            # 保存原始热力图数据
            np.save(save_path.replace('.png', '_original.npy'), cam)
            
            # 保存可视化热力图
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(save_path, heatmap)
            
            logging.info(f"Saved Layer-CAM heatmap to {save_path}")
            return cam
            
        except Exception as e:
            logging.error(f"Error in Layer-CAM explanation: {str(e)}")
            raise
