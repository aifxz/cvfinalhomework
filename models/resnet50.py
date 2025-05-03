import torchvision.models as models
def get_resnet50(pretrained=True):
    """
    返回一个 ResNet-50 模型实例。

    参数:
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    返回:
        torchvision.models.ResNet: ResNet-50 模型
    """
    model = models.resnet50(pretrained=pretrained)
    return model
