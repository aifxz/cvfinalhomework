import numpy as np
import torch
import torch.nn.functional as F

class RandomMasking:
    def __init__(self, model, mask_ratio=0.1, num_masks=50, baseline=0.0):
        self.model = model
        self.mask_ratio = mask_ratio
        self.num_masks = num_masks
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
            for _ in range(self.num_masks):
                mask = torch.ones((H, W), device=device)
                # 随机遮挡 mask_ratio 的像素
                num_pixels = int(H * W * self.mask_ratio)
                idx = torch.randperm(H * W)[:num_pixels]
                mask.view(-1)[idx] = 0
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask = mask.expand(-1, C, -1, -1)  # [1, C, H, W]
                masked_input = input_tensor * mask + self.baseline * (1 - mask)
                out_mask = self.model(masked_input)
                mask_score = out_mask[0, target_class].item()
                diff = base_score - mask_score
                heatmap += (1 - mask[0, 0]) * diff
                counts += (1 - mask[0, 0])
        heatmap = heatmap / (counts + 1e-8)
        heatmap = torch.clamp(heatmap, min=0)
        if heatmap.max() > 0:
            heatmap = heatmap / (heatmap.max() + 1e-8)
        return heatmap.cpu().numpy()

    def explain(self, input_tensor, *args, **kwargs):
        cam = self.generate_cam(input_tensor, kwargs.get('target_class', None))
        return cam 