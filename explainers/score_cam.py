import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import logging
from torch import nn
from utils.visualize import save_heatmap_overlay

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # 注册hook
        target_layer.register_forward_hook(self.save_activation)
        
        # 确保模型在评估模式
        self.model.eval()
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def generate_cam(self, input_tensor, target_class):
        # 获取激活图
        activations = self.activations
        
        # 如果目标类别为None，使用模型预测的类别
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
        
        # 计算每个通道的重要性分数
        b, c, h, w = activations.shape
        scores = []
        
        for i in range(c):
            # 创建掩码
            mask = activations[:, i:i+1, :, :]
            mask = F.interpolate(mask, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
            mask = mask / (mask.max() + 1e-8)
            
            # 应用掩码
            masked_input = input_tensor * mask
            
            # 前向传播
            with torch.no_grad():
                output = self.model(masked_input)
                score = output[0, target_class].item()
            
            scores.append(score)
        
        # 将分数转换为权重
        scores = torch.tensor(scores).to(activations.device)
        weights = F.softmax(scores, dim=0)
        
        # 计算CAM
        cam = torch.sum(weights.view(-1, 1, 1, 1) * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # 应用ReLU
        
        # 归一化
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
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
                self.model = self.model.cuda()
            
            # 前向传播以获取激活
            with torch.no_grad():
                _ = self.model(image)
            
            # 生成CAM
            cam = self.generate_cam(image, target_class)
            
            # 调整大小到原始图像尺寸
            cam = cv2.resize(cam, (image.shape[3], image.shape[2]))
            
            # 保存热力图
            save_dir = os.path.join("output/comparison_results", "train" if "train" in image_id else "test")
            os.makedirs(save_dir, exist_ok=True)
            
            # 构建唯一的文件名
            save_path = os.path.join(save_dir, f"{image_id}_{category}_{model_name}_Score-CAM.png")
            
            # 保存原始热力图数据
            np.save(save_path.replace('.png', '_original.npy'), cam)
            
            # 保存可视化热力图
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            cv2.imwrite(save_path, heatmap)
            
            logging.info(f"Saved Score-CAM heatmap to {save_path}")
            return cam
            
        except Exception as e:
            logging.error(f"Error in Score-CAM explanation: {str(e)}")
            raise
