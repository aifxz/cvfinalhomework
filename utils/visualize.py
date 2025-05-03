import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

# 将热力图叠加到图像上并保存
def save_heatmap_overlay(rgb_image, grayscale_cam, save_path):
    img = cv2.resize(rgb_image, (224, 224))
    cam = np.uint8(255 * grayscale_cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, overlay)

# 显示图像
def show_image(image_np, title=""):
    plt.imshow(image_np)
    plt.title(title)
    plt.axis('off')
    plt.show()
