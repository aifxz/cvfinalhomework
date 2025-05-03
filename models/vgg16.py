import torchvision.models as models

def get_vgg16(pretrained=True):
    """
    返回一个 VGG-16 模型实例。

    参数:
        pretrained (bool): 是否加载在 ImageNet 上预训练的权重。

    返回:
        torchvision.models.VGG: VGG-16 模型
    """
    model = models.vgg16(pretrained=pretrained)
    return model
