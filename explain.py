import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import logging
from tqdm import tqdm
import time
from models.resnet import ResNet50
from models.mobilenet import MobileNetV2
from explainers.grad_cam import GradCAM
from explainers.score_cam import ScoreCAM
from explainers.layer_cam import LayerCAM
from utils.visualize import save_heatmap_overlay
from datetime import datetime

def load_model(model_name, device):
    if model_name == 'ResNet50':
        model = ResNet50(num_classes=10)
    elif model_name == 'MobileNetV2':
        model = MobileNetV2(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_path = os.path.join("output/models", f"{model_name.lower()}_cifar10.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def get_target_layer(model, model_name):
    if model_name == 'ResNet50':
        return model.model.layer4[-1]  # 使用最后一个卷积层
    elif model_name == 'MobileNetV2':
        # MobileNetV2的最后一个卷积层在features的最后一个block中
        return model.model.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")

def main():
    parser = argparse.ArgumentParser(description='Generate explanations for models')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to explain')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for processing')
    args = parser.parse_args()

    # 设置日志
    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集（只取前num_samples张图片）
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_dataset = torch.utils.data.Subset(train_dataset, range(min(args.num_samples, len(train_dataset))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化解释器
    explainers = {
        'Grad-CAM': GradCAM,
        'Score-CAM': ScoreCAM,
        'Layer-CAM': LayerCAM
    }
    
    # 初始化性能统计
    performance_stats = {
        model_name: {
            explainer_name: {
                'total_time': 0,
                'avg_time_per_image': 0,
                'memory_usage': 0
            } for explainer_name in explainers.keys()
        } for model_name in ['ResNet50', 'MobileNetV2']
    }
    
    # 对每个模型和解释器组合进行处理
    for model_name in ['ResNet50', 'MobileNetV2']:
        print(f"\nGenerating explanations for {model_name}...")
        model = load_model(model_name, device)
        
        for explainer_name, explainer_class in explainers.items():
            print(f"\nUsing {explainer_name}...")
            start_time = time.time()
            
            # 获取目标层
            if model_name == 'ResNet50':
                target_layer = model.layer4[-1].conv2
            else:  # MobileNetV2
                target_layer = model.features[-1].conv[0]
            
            # 创建解释器实例
            explainer = explainer_class(model, target_layer)
            
            # 处理每个批次
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Processing {explainer_name}")):
                for i in range(len(images)):
                    image = images[i].unsqueeze(0)
                    label = labels[i].item()
                    
                    # 生成解释
                    try:
                        cam = explainer.explain(
                            image,
                            f"train_{batch_idx * args.batch_size + i}",
                            train_dataset.dataset.classes[label],
                            model_name
                        )
                    except Exception as e:
                        logging.error(f"Error processing image {batch_idx * args.batch_size + i} with {explainer_name}: {str(e)}")
                        continue
            
            # 计算性能统计
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_image = total_time / len(train_dataset)
            
            performance_stats[model_name][explainer_name]['total_time'] = total_time
            performance_stats[model_name][explainer_name]['avg_time_per_image'] = avg_time_per_image
            
            # 记录GPU内存使用（如果可用）
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
                performance_stats[model_name][explainer_name]['memory_usage'] = memory_allocated
                torch.cuda.reset_peak_memory_stats()
            
            print(f"Completed {explainer_name} in {total_time:.2f} seconds")
            print(f"Average time per image: {avg_time_per_image:.2f} seconds")
            if torch.cuda.is_available():
                print(f"Peak GPU memory usage: {memory_allocated:.2f} MB")
    
    # 生成性能报告
    performance_report_path = os.path.join("output/comparison_results/analysis_reports", "performance_report.txt")
    os.makedirs(os.path.dirname(performance_report_path), exist_ok=True)
    
    with open(performance_report_path, 'w') as f:
        f.write("Performance Comparison Report\n")
        f.write("==========================\n\n")
        
        for model_name in ['ResNet50', 'MobileNetV2']:
            f.write(f"\n{model_name}:\n")
            f.write("----------------\n")
            
            for explainer_name in explainers.keys():
                stats = performance_stats[model_name][explainer_name]
                f.write(f"\n{explainer_name}:\n")
                f.write(f"  Total processing time: {stats['total_time']:.2f} seconds\n")
                f.write(f"  Average time per image: {stats['avg_time_per_image']:.2f} seconds\n")
                if torch.cuda.is_available():
                    f.write(f"  Peak GPU memory usage: {stats['memory_usage']:.2f} MB\n")
            
            f.write("\nPerformance Comparison:\n")
            f.write("----------------------\n")
            
            # 计算相对性能
            fastest_time = min(stats['total_time'] for stats in performance_stats[model_name].values())
            for explainer_name in explainers.keys():
                stats = performance_stats[model_name][explainer_name]
                relative_speed = fastest_time / stats['total_time']
                f.write(f"{explainer_name} is {relative_speed:.2f}x {'faster' if relative_speed > 1 else 'slower'} than the fastest method\n")

if __name__ == "__main__":
    main() 