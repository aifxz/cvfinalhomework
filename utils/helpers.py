import torch
import numpy as np
from PIL import Image
import logging
import os
import json
from datetime import datetime

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

def setup_logging(log_dir="logs"):
    """
    设置日志记录
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def save_results(results, output_dir="output"):
    """
    保存结果到JSON文件
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    logging.info(f"Results saved to {output_file}")
    return output_file

def get_device_info():
    """
    获取设备信息
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_info = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info.update({
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version()
        })
    
    logging.info("Device Information:")
    for key, value in device_info.items():
        logging.info(f"{key}: {value}")
    
    return device_info

def calculate_metrics(predictions, ground_truth):
    """
    计算模型性能指标
    """
    try:
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        accuracy = np.mean(predictions == ground_truth)
        precision = np.sum((predictions == 1) & (ground_truth == 1)) / np.sum(predictions == 1)
        recall = np.sum((predictions == 1) & (ground_truth == 1)) / np.sum(ground_truth == 1)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
        logging.info("Model Metrics:")
        for key, value in metrics.items():
            logging.info(f"{key}: {value:.4f}")
        
        return metrics
    except Exception as e:
        logging.error(f"Error calculating metrics: {str(e)}")
        return None

def format_time(seconds):
    """
    格式化时间显示
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} hours"

def create_output_directories():
    """
    创建输出目录结构
    """
    directories = [
        "output",
        "output/comparison_reports",
        "output/visualizations",
        "output/metrics",
        "output/models",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")
    
    return directories
