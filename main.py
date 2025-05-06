import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import time
import pandas as pd
from tabulate import tabulate
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
import torch.nn.functional as F
import gc

# 导入预处理函数
from utils.preprocess import preprocess_for_cam
from utils.helpers import create_output_directories, setup_logging, get_device_info
from utils.visualize import plot_comparison, save_heatmap_overlay

# 导入解释器
from explainers.grad_cam import GradCAM
# from explainers.score_cam import ScoreCAM  # 注释掉
from explainers.layer_cam import LayerCAM
from explainers.occlusion import Occlusion
from explainers.random_masking import RandomMasking

# 导入模型加载器
from models.resnet import ResNet50
from models.mobilenet import MobileNetV2

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'explanation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

def get_device():
    """
    获取并设置设备
    """
    try:
        # 详细检查CUDA状态
        logging.info("Checking CUDA status...")
        logging.info(f"PyTorch version: {torch.__version__}")
        logging.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.info(f"CUDA version: {torch.version.cuda}")
            logging.info(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logging.info(f"Device {i}: {torch.cuda.get_device_name(i)}")
                logging.info(f"Device {i} memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
            
            # 尝试使用CUDA
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
            # 测试CUDA是否真的可用
            test_tensor = torch.randn(1, device=device)
            if test_tensor.device.type == 'cuda':
                logging.info("CUDA test successful!")
                return device
            else:
                logging.warning("CUDA test failed, falling back to CPU")
                return torch.device("cpu")
        else:
            logging.warning("CUDA is not available. Using CPU instead.")
            return torch.device("cpu")
    except Exception as e:
        logging.error(f"Error in CUDA detection: {str(e)}")
        logging.warning("Falling back to CPU due to error")
        return torch.device("cpu")

def load_model(model_name, device):
    """
    加载预训练模型
    Args:
        model_name: 模型名称
        device: 设备
    Returns:
        model: 加载的模型
    """
    try:
        if model_name == "ResNet50":
            model = ResNet50(num_classes=10)
        elif model_name == "MobileNetV2":
            model = MobileNetV2(num_classes=10)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # 加载预训练权重
        model_path = os.path.join("output/models", f"{model_name.lower()}_cifar10.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))

        model = model.to(device)
        model.eval()
        logging.info(f"Loaded {model_name} model successfully")
        return model

    except Exception as e:
        logging.error(f"Error loading {model_name} model: {str(e)}")
        raise

def create_output_directories():
    """创建必要的输出目录"""
    try:
        # 创建主输出目录
        os.makedirs("output", exist_ok=True)
        
        # 创建比较结果目录
        os.makedirs(os.path.join("output", "comparison_results", "train"), exist_ok=True)
        os.makedirs(os.path.join("output", "comparison_results", "test"), exist_ok=True)
        
        # 创建日志目录
        os.makedirs("output/logs", exist_ok=True)
        
        logging.info("Created all output directories successfully")
        
    except Exception as e:
        logging.error(f"Error creating output directories: {str(e)}")
        raise

def setup_logging():
    """设置日志配置"""
    try:
        # 创建日志目录
        os.makedirs("logs", exist_ok=True)
        
        # 设置日志文件名
        log_file = os.path.join("logs", f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info(f"Logging setup completed. Log file: {log_file}")
        return log_file
        
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise

def get_device_info():
    """获取设备信息"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_info = {
            "device": device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        logging.info(f"Device information: {device_info}")
        return device_info
        
    except Exception as e:
        logging.error(f"Error getting device info: {str(e)}")
        raise

def process_image_with_explainer(image, model, explainer_name, explainer_func, image_id, category, model_name):
    """
    使用指定的解释器处理图像
    Args:
        image: PIL Image对象
        model: 预加载的模型
        explainer_name: 解释器名称
        explainer_func: 解释器函数
        image_id: 图像ID
        category: 图像类别
        model_name: 模型名称
    Returns:
        dict: 包含热力图、预测结果和置信度的字典
    """
    try:
        logging.info(f"Running {explainer_name}...")
        start_time = time.time()
        device = next(model.model.parameters()).device
        rgb_image = np.array(image).astype(np.float32) / 255.0
        if rgb_image.shape[2] == 3:
            rgb_image = np.transpose(rgb_image, (2, 0, 1))
        rgb_tensor = torch.from_numpy(rgb_image).float().unsqueeze(0).to(device)
        input_tensor = rgb_tensor
        # target_layer选择（空间分辨率大于1x1）
        if isinstance(model, ResNet50):
            target_layer = model.model.layer3[-1]  # 更靠前的block
        elif isinstance(model, MobileNetV2):
            target_layer = model.model.features[4]  # 或 features[2]
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        # 每次推理都new解释器对象
        if explainer_name == "Grad-CAM":
            explainer = GradCAM(model.model, target_layer)
            heatmap = explainer.explain(input_tensor)
        elif explainer_name == "Layer-CAM":
            explainer = LayerCAM(model.model, target_layer)
            heatmap = explainer.explain(input_tensor)
        elif explainer_name == "Occlusion":
            explainer = Occlusion(model.model, patch_size=8, stride=4, baseline=0.0)
            heatmap = explainer.explain(input_tensor)
        elif explainer_name == "RandomMasking":
            explainer = RandomMasking(model.model, mask_ratio=0.1, num_masks=50, baseline=0.0)
            heatmap = explainer.explain(input_tensor)
        else:
            raise ValueError(f"Unknown explanation method: {explainer_name}")
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_class = torch.max(probabilities, 1)
        return {
            'heatmap': heatmap,
            'prediction': top_class.item(),
            'confidence': top_prob.item(),
            'top_classes': probabilities.squeeze().cpu().numpy()
        }
    except Exception as e:
        logging.error(f"Error in process_image_with_explainer: {str(e)}")
        return None

def compare_models(image_paths, models, explainers, output_dir, batch_size=8):
    """批量比较不同模型在相同图片上的解释结果"""
    results = []
    total_images = len(image_paths)
    processed_images = 0
    total_explanations = total_images * len(models) * len(explainers)
    completed_explanations = 0
    start_time = time.time()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 自动降级 batch_size
    orig_batch_size = batch_size
    while True:
        try:
            # 分批处理图片
            for i in range(0, total_images, batch_size):
                batch_paths = image_paths[i:i+batch_size]
                logging.info(f"Processing batch {i//batch_size + 1}/{(total_images-1)//batch_size + 1} ({len(batch_paths)} images)")
                # tqdm 外层：图片进度
                for image_path in tqdm(batch_paths, desc=f"Images {i+1}-{i+len(batch_paths)}/{total_images}", position=0, leave=True):
                    try:
                        # 加载图片
                        image = Image.open(image_path).convert('RGB')
                        image_name = os.path.basename(image_path)
                        image_id = os.path.splitext(image_name)[0]
                        category = os.path.basename(os.path.dirname(image_path))
                        # tqdm 内层：模型/解释器组合进度
                        model_results = {}
                        for model_name, model in models.items():
                            model_results[model_name] = {}
                            for explainer_name, explainer in tqdm(explainers.items(), desc=f"{image_name} Explainers", position=1, leave=False):
                                try:
                                    explanation = process_image_with_explainer(
                                        image, 
                                        model, 
                                        explainer_name, 
                                        explainer,
                                        image_id,
                                        category,
                                        model_name
                                    )
                                    if (explanation is None or
                                        ('heatmap' in explanation and np.all(explanation['heatmap'] == 0))):
                                        continue
                                    image_np = np.array(image)
                                    save_path = os.path.join(
                                        output_dir,
                                        f"{image_id}_{model_name}_{explainer_name}.png"
                                    )
                                    save_heatmap_overlay(
                                        image_np,
                                        explanation['heatmap'],
                                        save_path
                                    )
                                    npy_save_path = os.path.join(
                                        output_dir,
                                        f"{image_id}_{model_name}_{explainer_name}_original.npy"
                                    )
                                    np.save(npy_save_path, explanation['heatmap'])
                                    if 'all_heatmaps' not in locals():
                                        all_heatmaps = {}
                                    all_heatmaps[explainer_name] = explanation['heatmap']
                                    if 'orig_img' not in locals():
                                        orig_img = image_np
                                    model_results[model_name][explainer_name] = {
                                        'prediction': explanation['prediction'],
                                        'confidence': explanation['confidence'],
                                        'top_classes': explanation['top_classes']
                                    }
                                except Exception as e:
                                    logging.error(f"Error processing {image_name} with {model_name} and {explainer_name}: {str(e)}")
                                    continue
                        results.append({
                            'image': image_name,
                            'results': model_results
                        })
                    except Exception as e:
                        logging.error(f"Error processing {image_path}: {str(e)}")
                        continue
                    processed_images += 1
                    # tqdm 会自动显示进度和剩余时间
                    torch.cuda.empty_cache()
                    gc.collect()
            save_comparison_report(results, os.path.join(output_dir, 'comparison_report2.txt'))
            return results
        except RuntimeError as e:
            if 'out of memory' in str(e).lower() and batch_size > 1:
                logging.warning(f"OOM detected, reducing batch_size from {batch_size} to {batch_size//2}")
                batch_size = max(1, batch_size // 2)
                torch.cuda.empty_cache()
                gc.collect()
                continue
            else:
                raise
        break

def save_comparison_report(results, output_path):
    """
    保存比较报告
    Args:
        results: 比较结果列表
        output_path: 输出文件路径
    """
    try:
        with open(output_path, 'w') as f:
            for result in results:
                f.write(f"\nImage: {result['image']}\n")
                f.write("=" * 50 + "\n")
                
                for model_name, model_results in result['results'].items():
                    f.write(f"\nModel: {model_name}\n")
                    f.write("-" * 30 + "\n")
                    
                    for explainer_name, explanation in model_results.items():
                        f.write(f"Explainer: {explainer_name}\n")
                        f.write(f"Prediction: {explanation['prediction']}\n")
                        f.write(f"Confidence: {explanation['confidence']:.4f}\n")
                        f.write("Top 3 Classes:\n")
                        top3_indices = np.argsort(explanation['top_classes'])[-3:][::-1]
                        for idx in top3_indices:
                            f.write(f"  Class {idx}: {explanation['top_classes'][idx]:.4f}\n")
                        f.write("\n")
                
                f.write("\n" + "=" * 50 + "\n")
        
        logging.info(f"Saved comparison report to: {output_path}")
        
    except Exception as e:
        logging.error(f"Error saving comparison report: {str(e)}")
        raise

def generate_final_report(train_results, test_results):
    """
    生成最终报告
    Args:
        train_results: 训练集结果
        test_results: 测试集结果
    """
    try:
        # 统计信息
        stats = {
            'total_images': len(train_results) + len(test_results),
            'train_images': len(train_results),
            'test_images': len(test_results),
            'successful_explanations': {
                'Grad-CAM': 0,
                'Layer-CAM': 0,
                'Occlusion': 0,
                'RandomMasking': 0
            },
            'failed_explanations': {
                'Grad-CAM': 0,
                'Layer-CAM': 0,
                'Occlusion': 0,
                'RandomMasking': 0
            }
        }
        
        # 统计成功和失败的解释
        for results in [train_results, test_results]:
            for result in results:
                for model_results in result['results'].values():
                    for explainer_name, explanation in model_results.items():
                        if explanation is not None:
                            stats['successful_explanations'][explainer_name] += 1
                        else:
                            stats['failed_explanations'][explainer_name] += 1
        
        # 生成报告
        report = f"""
Final Report
============
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Dataset Statistics:
-----------------
Total Images Processed: {stats['total_images']}
Training Images: {stats['train_images']}
Test Images: {stats['test_images']}

Explanation Method Performance:
-----------------------------
"""
        for method in ["Grad-CAM", "Layer-CAM", "Occlusion", "RandomMasking"]:
            total = stats['successful_explanations'][method] + stats['failed_explanations'][method]
            success_rate = stats['successful_explanations'][method] / total * 100 if total > 0 else 0
            
            report += f"""
{method}:
  Successful: {stats['successful_explanations'][method]}
  Failed: {stats['failed_explanations'][method]}
  Success Rate: {success_rate:.2f}%
"""
        
        # 保存报告
        with open("output/final_report.txt", "w") as f:
            f.write(report)
        
        logging.info("Final report generated and saved to output/final_report.txt")
        
    except Exception as e:
        logging.error(f"Error generating final report: {str(e)}")
        raise

def get_image_paths(train_dir, test_dir, num_samples=1000):
    """获取训练集和测试集的图片路径"""
    train_images = []
    test_images = []
    
    try:
        # 检查目录是否存在
        if not os.path.exists(train_dir):
            logging.error(f"Training directory not found: {train_dir}")
            return [], []
        if not os.path.exists(test_dir):
            logging.error(f"Test directory not found: {test_dir}")
            return [], []
            
        logging.info(f"Training directory: {train_dir}")
        logging.info(f"Test directory: {test_dir}")
        
        # 获取训练集图片
        train_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if train_files:
            # 随机采样
            sampled_images = np.random.choice(train_files, min(num_samples, len(train_files)), replace=False)
            train_images = [os.path.join(train_dir, img) for img in sampled_images]
            logging.info(f"Sampled {len(train_images)} images from training set")
        else:
            logging.warning(f"No images found in training directory")
        
        # 获取测试集图片
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if test_files:
            # 随机采样
            sampled_images = np.random.choice(test_files, min(num_samples, len(test_files)), replace=False)
            test_images = [os.path.join(test_dir, img) for img in sampled_images]
            logging.info(f"Sampled {len(test_images)} images from test set")
        else:
            logging.warning(f"No images found in test directory")
        
        logging.info(f"Total sampled images: {len(train_images)} training and {len(test_images)} test images")
        return train_images, test_images
        
    except Exception as e:
        logging.error(f"Error in get_image_paths: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Generate explanations for models')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to explain')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    args = parser.parse_args()

    try:
        # 设置日志
        log_file = setup_logging()
        logging.info(f"Log file created at: {log_file}")
        
        # 获取设备
        device = get_device()
        logging.info(f"Using device: {device}")
        
        # 设置数据路径
        base_dir = r"E:\machinelearning\datacollection\CIFAR-10-100(含png图)\cifar_png\cifar"
        train_dir = os.path.join(base_dir, "train")
        test_dir = os.path.join(base_dir, "test")
        
        # 检查路径是否存在
        if not os.path.exists(base_dir):
            logging.error(f"Base directory not found: {base_dir}")
            return
        if not os.path.exists(train_dir):
            logging.error(f"Training directory not found: {train_dir}")
            return
        if not os.path.exists(test_dir):
            logging.error(f"Test directory not found: {test_dir}")
            return
            
        logging.info(f"Base directory: {base_dir}")
        logging.info(f"Training directory: {train_dir}")
        logging.info(f"Test directory: {test_dir}")
        
        # 创建输出目录
        create_output_directories()
        
        # 获取图片路径
        train_images, test_images = get_image_paths(train_dir, test_dir, args.num_samples)
        if not train_images and not test_images:
            logging.error("No images found in dataset")
            return
            
        logging.info(f"Found {len(train_images)} training images and {len(test_images)} test images")
        
        # 加载模型并移动到GPU
        models_to_compare = {}
        for model_name in ["ResNet50", "MobileNetV2"]:
            model = load_model(model_name, device)
            models_to_compare[model_name] = model
            logging.info(f"Loaded {model_name} to {device}")
        
        # 处理训练集图片
        logging.info("Processing training images...")
        train_results = compare_models(
            train_images,
            models_to_compare,
            {
                "Grad-CAM": process_image_with_explainer,
                "Layer-CAM": process_image_with_explainer,
                "Occlusion": process_image_with_explainer,
                "RandomMasking": process_image_with_explainer
            },
            os.path.join("output", "comparison_results2", "train"),
            batch_size=args.batch_size
        )
        
        # 处理测试集图片
        logging.info("Processing test images...")
        test_results = compare_models(
            test_images,
            models_to_compare,
            {
                "Grad-CAM": process_image_with_explainer,
                "Layer-CAM": process_image_with_explainer,
                "Occlusion": process_image_with_explainer,
                "RandomMasking": process_image_with_explainer
            },
            os.path.join("output", "comparison_results2", "test"),
            batch_size=args.batch_size
        )
        
        # 生成最终报告
        generate_final_report(train_results, test_results)
        
        logging.info("All processing completed!")
        
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()
