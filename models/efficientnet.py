import torchvision.models as models

def get_efficientnet_b0(pretrained=True):
    """
    返回一个 EfficientNet-B0 模型实例。

    参数:
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    返回:
        torchvision.models.EfficientNet: EfficientNet-B0 模型
    """
    model = models.efficientnet_b0(pretrained=pretrained)
    return model
