import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
import logging
import os
import matplotlib.pyplot as plt
from utils.visualize import save_heatmap_overlay


def lime_explanation(model, input_image, rgb_image, use_cuda=True):
    """
    运行LIME解释器
    """
    try:
        model.eval()
        device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
        model.to(device)
        
        logging.info(f"Running LIME on device: {device}")
        
        # 创建LIME解释器
        explainer = lime_image.LimeImageExplainer()
        logging.info("LIME explainer created")
        
        # 定义预测函数
        def batch_predict(images):
            model.eval()
            batch = torch.stack([preprocess_image(img) for img in images]).to(device)
            logits = model(batch)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()
        
        # 生成解释
        explanation = explainer.explain_instance(
            rgb_image.astype('double'),
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
