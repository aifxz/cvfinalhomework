# import torch
# import torch.nn.functional as F
# import cv2
# import os
# from tqdm import tqdm
# from utils.visualize import save_heatmap_overlay

# class ScoreCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.activations = None
#         self._register_hooks()
#         self.model.eval()
#     
#     def _register_hooks(self):
#         def forward_hook(module, input, output):
#             self.activations = output
#         self.target_layer.register_forward_hook(forward_hook)
#     
#     def generate_cam(self, input_tensor, target_class):
#         # 第一次 forward：记录 activations，不加 no_grad
#         _ = self.model(input_tensor)
#         activations = self.activations  # shape [B, C, H, W]
#         if activations is None:
#             logging.error("Score-CAM: activations is None! Returning None.")
#             return None
#         activations = activations[0]  # [C, H, W]
#         C, h, w = activations.shape
#         H, W = input_tensor.shape[2], input_tensor.shape[3]
#         upsampled_features = F.interpolate(activations.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)  # (C, H, W)

#         valid_fmaps = []
#         scores = []
#         skipped = 0
#         for i in tqdm(range(C), desc='Score-CAM channels'):
#             fmap = upsampled_features[i]
#             # 筛选条件恢复为 std < 1e-5
#             if torch.std(fmap) < 1e-5:
#                 skipped += 1
#                 continue
#             norm_fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
#             if torch.isnan(norm_fmap).any() or torch.all(norm_fmap == 0):
#                 skipped += 1
#                 continue
#             # mask 扩展到 [1, 3, H, W]
#             mask = norm_fmap.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
#             mask = mask.expand(-1, input_tensor.shape[1], -1, -1)  # [1, 3, H, W]
#             masked_input = input_tensor * mask
#             with torch.no_grad():
#                 output = self.model(masked_input)
#             if target_class is None:
#                 target_class = output.argmax(dim=1).item()
#             score = output[0, target_class].item()
#             logging.debug(f'score: {score}, i={i}')
#             valid_fmaps.append(norm_fmap)
#             scores.append(score)
#         logging.info(f"Score-CAM: valid channels: {len(scores)}/{C}, skipped: {skipped}")
#         # 只要有效通道>=3才生成热图
#         if len(scores) < 3:
#             logging.warning("Score-CAM: Too few valid channels, returning None.")
#             return None
#         # 限制最大通道数
#         max_channels = 32
#         if len(valid_fmaps) > max_channels:
#             indices = np.linspace(0, len(valid_fmaps)-1, max_channels, dtype=int)
#             valid_fmaps = [valid_fmaps[i] for i in indices]
#             scores = [scores[i] for i in indices]
#         scores_tensor = torch.tensor(scores)
#         if torch.isnan(scores_tensor).any() or torch.all(scores_tensor == 0):
#             logging.warning("Score-CAM: All scores are zero or NaN, returning None.")
#             return None
#         weights = F.softmax(scores_tensor, dim=0)
#         if torch.isnan(weights).any() or torch.all(weights == 0):
#             logging.warning("Score-CAM: Softmax weights invalid, returning None.")
#             return None
#         cam = torch.zeros_like(valid_fmaps[0])
#         for fmap, w in zip(valid_fmaps, weights):
#             cam += w * fmap
#         cam = torch.clamp(cam, min=0)
#         if cam.max() == cam.min() or torch.isnan(cam).any():
#             logging.warning("Score-CAM: Final CAM is constant or NaN, returning None.")
#             return None
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#         return cam.cpu().numpy()
#     
#     def explain(self, input_tensor, *args, **kwargs):
#         cam = self.generate_cam(input_tensor, kwargs.get('target_class', None))
#         if cam is None:
#             # 返回全0热图，避免 nan
#             H, W = input_tensor.shape[2], input_tensor.shape[3]
#             return np.zeros((H, W))
#         return cam
