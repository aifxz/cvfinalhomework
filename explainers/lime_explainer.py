import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
import logging
import os
import matplotlib.pyplot as plt
from utils.visualize import save_heatmap_overlay
from utils.preprocess import preprocess_for_cam
from PIL import Image


def lime_explanation(model, input_image, rgb_image, use_cuda=True):
    """
    运行LIME解释器
    """
    try:
        model.eval()
        device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        model.to(device)
        
        logging.info(f"Running LIME on device: {device}")
        
        # 确保输入图像是numpy数组
        if isinstance(rgb_image, torch.Tensor):
            rgb_image = rgb_image.cpu().numpy()
        if rgb_image.dtype != np.uint8:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # 确保图像形状正确
        if len(rgb_image.shape) == 3 and rgb_image.shape[0] == 3:
            rgb_image = np.transpose(rgb_image, (1, 2, 0))
        
        logging.info(f"Input image shape: {rgb_image.shape}, dtype: {rgb_image.dtype}")
        
        # 创建LIME解释器
        explainer = lime_image.LimeImageExplainer()
        logging.info("LIME explainer created")
        
        # 定义预测函数
        def batch_predict(images):
            model.eval()
            # 确保输入图像格式正确
            processed_images = []
            for img in images:
                if img.shape[0] == 1:  # 如果是(1,1,3)形状
                    img = img.squeeze(0)  # 转换为(1,3)形状
                if len(img.shape) == 2:  # 如果是(1,3)形状
                    img = np.expand_dims(img, axis=0)  # 转换为(1,1,3)形状
                if img.shape[0] == 3:  # 如果是(3,H,W)形状
                    img = np.transpose(img, (1, 2, 0))  # 转换为(H,W,3)形状
                processed_images.append(img)
            
            batch = torch.stack([preprocess_for_cam(Image.fromarray(img.astype(np.uint8))) for img in processed_images]).to(device)
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()
        
        # 生成解释
        explanation = explainer.explain_instance(
            rgb_image.astype(np.float64),
            batch_predict,
            top_labels=5,
            hide_color=0,
            num_samples=1000
        )
        logging.info("LIME explanation generated")
        
        # 获取热力图
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False
        )
        logging.info("LIME heatmap generated")
        
        # 可视化结果
        visualization = mark_boundaries(temp / 255.0, mask)
        logging.info("LIME visualization created")
        
        # 保存热力图叠加
        output_dir = os.path.join("output", "visualizations", "lime")
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, "lime_heatmap.png")
        save_heatmap_overlay(rgb_image, mask, save_path)
        logging.info(f"LIME heatmap saved to: {save_path}")
        
        return visualization
        
    except Exception as e:
        logging.error(f"Error in LIME: {str(e)}")
        raise
