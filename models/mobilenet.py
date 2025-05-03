import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # 加载预训练的MobileNetV2
        self.model = models.mobilenet_v2(pretrained=True)
        
        # 修改最后的分类器以适应新的类别数
        num_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_features, num_classes)
        
    def forward(self, x):
        return self.model(x)
