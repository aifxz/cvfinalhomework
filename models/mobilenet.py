import torchvision.models as models

def get_mobilenet_v2(pretrained=True):
    """
    返回一个 MobileNetV2 模型实例。

    参数:
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    返回:
        torchvision.models.MobileNetV2: MobileNetV2 模型
    """
    model = models.mobilenet_v2(pretrained=pretrained)
    return model
