import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import os
import logging
from datetime import datetime
from tqdm import tqdm
from models.resnet import ResNet50
from models.mobilenet import MobileNetV2

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def main():
    parser = argparse.ArgumentParser(description='Train models for image classification')
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to use for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    args = parser.parse_args()

    # 设置日志
    log_dir = "output/logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
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
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # 初始化模型
    models = {
        'ResNet50': ResNet50(num_classes=10).to(device),
        'MobileNetV2': MobileNetV2(num_classes=10).to(device)
    }

    # 训练每个模型
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, criterion, optimizer, device, args.num_epochs)

        # 保存模型
        model_dir = "output/models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{model_name.lower()}_cifar10.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved {model_name} to {model_path}")

if __name__ == "__main__":
    main() 