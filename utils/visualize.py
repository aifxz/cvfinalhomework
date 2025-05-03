import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import seaborn as sns
from PIL import Image
import logging
import torch

# 将热力图叠加到图像上并保存
def save_heatmap_overlay(image, heatmap, save_path):
    """
    保存热力图叠加到原始图像上
    """
    try:
        # 确保输入图像是numpy数组
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # 确保热力图是numpy数组
        if isinstance(heatmap, torch.Tensor):
            heatmap = heatmap.cpu().numpy()
        
        # 调整热力图大小以匹配图像
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # 归一化热力图
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # 应用热力图颜色映射
        heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        
        # 叠加热力图到原始图像
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        
        # 保存结果
        cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
    except Exception as e:
        logging.error(f"Error in save_heatmap_overlay: {str(e)}")
        raise

# 显示图像
def show_image(image_np, title=""):
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    plt.show()

def plot_comparison(visualizations, titles, save_path=None):
    """
    绘制多个解释器的比较图
    """
    try:
        n = len(visualizations)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        
        for i, (vis, title) in enumerate(zip(visualizations, titles)):
            if isinstance(vis, torch.Tensor):
                vis = vis.cpu().numpy()
            
            # 确保图像格式正确
            if vis.dtype != np.uint8:
                vis = (vis * 255).astype(np.uint8)
            
            # 显示图像
            axes[i].imshow(vis)
            axes[i].set_title(title)
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logging.info(f"Comparison plot saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logging.error(f"Error in plot_comparison: {str(e)}")
        raise

def plot_heatmap(data, title, save_path=None):
    """
    绘制热力图
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='viridis', annot=True, fmt='.2f')
    plt.title(title)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def plot_bar_chart(data, labels, title, xlabel, ylabel, save_path=None):
    """
    绘制柱状图
    """
    plt.figure(figsize=(12, 6))
    plt.bar(labels, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def plot_model_comparison(comparison_data, save_path=None):
    """
    绘制模型比较结果
    """
    models = [item['Model'] for item in comparison_data]
    confidences = [float(item['Confidence']) for item in comparison_data]
    times = [float(item['Inference Time'].replace('s', '')) for item in comparison_data]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制置信度比较
    ax1.bar(models, confidences)
    ax1.set_title('Model Confidence Comparison')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Confidence')
    ax1.tick_params(axis='x', rotation=45)
    
    # 绘制推理时间比较
    ax2.bar(models, times)
    ax2.set_title('Model Inference Time Comparison')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()

def save_visualization(visualization, filename, method_name, model_name=None):
    """
    保存可视化结果
    """
    output_dir = os.path.join("output", method_name)
    if model_name:
        output_dir = os.path.join(output_dir, model_name)
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{filename}_{method_name.lower()}.png")
    
    plt.imsave(output_path, visualization)
    return output_path
