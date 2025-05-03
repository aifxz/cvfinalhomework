import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import logging
from tqdm import tqdm
from utils.visualize import save_heatmap_overlay

class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # 注册hook来保存激活值
        target_layer.register_forward_hook(self.save_activation)
        
        # 确保模型在评估模式
        self.model.eval()
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        if self.activations.requires_grad:
            self.activations.requires_grad_(False)
    
    def generate_cam(self, input_tensor, target_class):
        # 获取目标层的激活值
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # 如果目标类别为None，使用模型预测的类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # 获取激活值
        activations = self.activations[0]  # 移除batch维度
        
        # 计算每个特征图的重要性分数
        importance_scores = []
        for i in tqdm(range(activations.shape[0]), desc="Calculating importance scores", leave=False):
            # 创建掩码并确保维度正确
            mask = activations[i].unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # 上采样掩码到输入图像大小
            upsampled = F.interpolate(mask, 
                                    size=(input_tensor.shape[2], input_tensor.shape[3]),
                                    mode='bilinear', align_corners=False)
            
            # 归一化掩码
            upsampled = (upsampled - upsampled.min()) / (upsampled.max() - upsampled.min() + 1e-8)
            
            # 扩展掩码以匹配输入图像的通道数
            upsampled = upsampled.expand(-1, input_tensor.shape[1], -1, -1)
            
            # 应用掩码并计算分数
            masked_input = input_tensor * upsampled
            with torch.no_grad():
                output = self.model(masked_input)
            score = output[0, target_class]
            importance_scores.append(score.item())
        
        # 将分数转换为权重
        importance_scores = torch.tensor(importance_scores)
        weights = F.softmax(importance_scores, dim=0)
        
        # 计算加权激活图
        cam = torch.zeros_like(activations[0])
        for i, weight in enumerate(weights):
            cam += weight * activations[i]
        
        # 应用ReLU并归一化
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # 上采样到输入图像大小
        cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                          size=(input_tensor.shape[2], input_tensor.shape[3]),
                          mode='bilinear', align_corners=False)
        
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
            
            # 构建唯一的文件名
            save_path = os.path.join(save_dir, f"{image_id}_{category}_{model_name}_Score-CAM.png")
            
            # 保存原始热力图数据
            np.save(save_path.replace('.png', '_original.npy'), cam)
            
            # 准备原始图像用于可视化
            original_image = image.squeeze(0).cpu().numpy()
            if original_image.shape[0] == 3:  # 如果是CHW格式
                original_image = np.transpose(original_image, (1, 2, 0))
            
            # 保存可视化热力图
            save_heatmap_overlay(original_image, cam, save_path, save_original=True)
            
            logging.info(f"Saved Score-CAM heatmap to {save_path}")
            return cam
            
        except Exception as e:
            logging.error(f"Error in Score-CAM explanation: {str(e)}")
            raise
