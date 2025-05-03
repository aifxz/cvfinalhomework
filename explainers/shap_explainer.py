import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from utils.preprocess import preprocess_for_shap  # 自定义：接受 PIL，返回 tensor(C, H, W)
from utils.visualize import save_visualization, save_heatmap_overlay

def shap_explanation(model, input_image, rgb_image, use_cuda=True):
    """
    运行SHAP解释器
    """
    try:
        model.eval()
        device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        model.to(device)
        
        logging.info(f"Running SHAP on device: {device}")
        
        # 确保输入图像是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # 预处理图像
        input_tensor = preprocess_for_shap(rgb_image).to(device)
        logging.info(f"Input tensor shape: {input_tensor.shape}")
        
        # 创建SHAP解释器
        background = torch.zeros_like(input_tensor)
        explainer = shap.DeepExplainer(model, background)
        logging.info("SHAP explainer created")
        
        # 生成SHAP值
        shap_values = explainer.shap_values(input_tensor)
        logging.info("SHAP values generated")
        
        # 获取热力图
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(input_tensor.cpu().numpy(), 1, -1), 1, 2)
        
        # 可视化结果
        plt.figure(figsize=(10, 5))
        shap.image_plot(shap_numpy, -test_numpy)
        logging.info("SHAP visualization created")
        
        # 保存热力图叠加
        output_dir = os.path.join("output", "visualizations", "shap")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "shap_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"SHAP heatmap saved to: {save_path}")
        
        return shap_numpy
        
    except Exception as e:
        logging.error(f"Error in SHAP: {str(e)}")
        raise
