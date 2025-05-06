import numpy as np
import torch
import torch.nn.functional as F

class Occlusion:
    def __init__(self, model, patch_size=8, stride=4, baseline=0.0):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.baseline = baseline
        self.model.eval()

    def generate_cam(self, input_tensor, target_class=None):
        # input_tensor: [1, C, H, W]
        device = input_tensor.device
        _, C, H, W = input_tensor.shape
        heatmap = torch.zeros((H, W), device=device)
        counts = torch.zeros((H, W), device=device)
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()
            base_score = output[0, target_class].item()
            for y in range(0, H, self.stride):
                for x in range(0, W, self.stride):
                    x1 = x
                    y1 = y
                    x2 = min(x1 + self.patch_size, W)
                    y2 = min(y1 + self.patch_size, H)
                    occluded = input_tensor.clone()
                    occluded[:, :, y1:y2, x1:x2] = self.baseline
                    out_occ = self.model(occluded)
                    occ_score = out_occ[0, target_class].item()
                    diff = base_score - occ_score
                    heatmap[y1:y2, x1:x2] += diff
                    counts[y1:y2, x1:x2] += 1
        heatmap = heatmap / (counts + 1e-8)
        heatmap = torch.clamp(heatmap, min=0)
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy()

    def explain(self, input_tensor, *args, **kwargs):
        cam = self.generate_cam(input_tensor, kwargs.get('target_class', None))
        return cam 