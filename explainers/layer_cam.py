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
        self._register_hooks()
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_tensor, target_class):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        gradients = self.gradients
        activations = self.activations
        print('input_tensor:', input_tensor.shape, input_tensor.max().item(), input_tensor.min().item())
        if gradients is not None:
            print('gradients:', gradients.shape, gradients.max().item(), gradients.min().item())
        if activations is not None:
            print('activations:', activations.shape, activations.max().item(), activations.min().item())
        if gradients is None or activations is None:
            logging.error("Layer-CAM: gradients or activations is None! Returning zeros.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        weights = torch.relu(gradients)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        print('cam:', cam.shape, cam.max().item(), cam.min().item())
        cam = cam.squeeze().detach().cpu().numpy()
        if np.max(cam) == np.min(cam):
            logging.warning("Layer-CAM: cam is constant, returning zeros.")
            return np.zeros_like(cam)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

    def explain(self, input_tensor, *args, **kwargs):
        cam = self.generate_cam(input_tensor, kwargs.get('target_class', None))
        return cam
