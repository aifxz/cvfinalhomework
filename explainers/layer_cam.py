import numpy as np
import torch
from pytorch_grad_cam import LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def run_layer_cam(model, target_layer, input_image, rgb_image, use_cuda=True):
    model.eval()
    device = torch.device("cuda" if (torch.cuda.is_available() and use_cuda) else "cpu")
    model.to(device)

    input_tensor = preprocess_image(rgb_image).to(device)
    cam = LayerCAM(model=model, target_layers=[target_layer], use_cuda=use_cuda)
    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_image / 255.0, grayscale_cam, use_rgb=True)
    return visualization
