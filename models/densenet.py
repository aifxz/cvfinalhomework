import torchvision.models as models

def get_densenet121(pretrained=True):
    """
    返回一个 DenseNet-121 模型实例。

    参数:
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    返回:
        torchvision.models.DenseNet: DenseNet-121 模型
    """
    model = models.densenet121(pretrained=pretrained)
    return model
