import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.preprocess import preprocess_for_shap  # 自定义：接受 PIL，返回 tensor(C, H, W)

def shap_explanation(model, image_pil, preprocess_func):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 输入图像预处理成 tensor(C, H, W)，并添加 batch 维度
    input_tensor = preprocess_func(image_pil).unsqueeze(0).to(device)

    # 构造 SHAP DeepExplainer
    explainer = shap.DeepExplainer(model, input_tensor)

    # 生成 SHAP 值
    shap_values = explainer.shap_values(input_tensor)

    # 可视化（注意转为 numpy）
    shap_numpy = [v.cpu().numpy() for v in shap_values]
    input_numpy = input_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    shap.image_plot(shap_numpy, np.expand_dims(input_numpy, axis=0))
