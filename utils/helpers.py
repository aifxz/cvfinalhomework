import torch
import numpy as np
from PIL import Image

# Tensor[C, H, W] -> np.array(H, W, C)
def tensor_to_numpy(tensor):
    tensor = tensor.detach().cpu()
    if tensor.ndimension() == 4:
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).numpy()

# np.array -> PIL
def numpy_to_pil(image_np):
    image_np = (image_np * 255).astype(np.uint8)
    return Image.fromarray(image_np)

# 保存 PIL 图像
def save_pil_image(image_pil, path):
    image_pil.save(path)
