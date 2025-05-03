import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import logging
from tqdm import tqdm
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
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to explain')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
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

    # 为每个模型和解释器生成解释
    for model_name in ['ResNet50', 'MobileNetV2']:
        print(f"\nGenerating explanations for {model_name}...")
        model = load_model(model_name, device)
        target_layer = get_target_layer(model, model_name)

        for explainer_name, ExplainerClass in explainers.items():
            print(f"Using {explainer_name}...")
            explainer = ExplainerClass(model, target_layer)

            # 使用tqdm显示进度
            for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Processing {explainer_name}")):
                images = images.to(device)
                labels = labels.to(device)
                
                for idx, (image, label) in enumerate(zip(images, labels)):
                    image = image.unsqueeze(0)  # 添加batch维度
                    image_id = f"train_{batch_idx * args.batch_size + idx}"
                    category = train_dataset.dataset.classes[label.item()]
                    
                    try:
                        explainer.explain(
                            image=image,
                            image_id=image_id,
                            category=category,
                            model_name=model_name,
                            target_class=label.item()
                        )
                    except Exception as e:
                        logging.error(f"Error processing image {image_id} with {explainer_name}: {str(e)}")
                        continue

if __name__ == "__main__":
    main() 