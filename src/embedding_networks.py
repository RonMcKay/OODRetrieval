"""This code is based on the PyTorch implementations of the used networks"""

import re

import torch
import torch.nn.functional as f
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from torchvision.models.densenet import DenseNet
from torchvision.models.densenet import model_urls as densenet_urls
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.vgg import VGG, cfgs, make_layers


class FeatureVGG(VGG):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(FeatureVGG, self).__init__(features, num_classes, init_weights)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def feature_vgg16(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs["init_weights"] = False
    model = FeatureVGG(make_layers(cfgs["D"]), **kwargs)
    if pretrained:
        model.load_state_dict(models.vgg16(pretrained=True).state_dict())
    return model


class FeatureResNet(ResNet):
    def __init__(self, block, layers, num_classes=1000, **kwargs):
        super(FeatureResNet, self).__init__(block, layers, num_classes, **kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = f.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


def feature_resnet18(pretrained=True, **kwargs):
    model = FeatureResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(models.resnet18(pretrained=True).state_dict())
    return model


def feature_resnet101(pretrained=True, **kwargs):
    model = FeatureResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(models.resnet101(pretrained=True).state_dict())
    return model


def feature_resnet152(pretrained=True, **kwargs):
    model = FeatureResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(models.resnet152(pretrained=True).state_dict())
    return model


def feature_wide_resnet101(pretrained=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    model = FeatureResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(models.wide_resnet101_2(pretrained=True).state_dict())
    return model


def extract_densenet_features(x):
    return x.view(x.shape[0], x.shape[1], -1).mean(2)


class FeatureDenseNet(DenseNet):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=1000,
    ):
        super(FeatureDenseNet, self).__init__(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            num_classes=num_classes,
        )

    def forward(self, x):
        x = self.features(x)
        x = f.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x


def feature_densenet121(pretrained=True, **kwargs):
    model = FeatureDenseNet(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs
    )
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            (
                r"^(.*denselayer\d+\.(?:norm|relu|conv))"
                r"\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
            )
        )
        state_dict = model_zoo.load_url(densenet_urls["densenet121"])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def feature_densenet169(pretrained=True, **kwargs):
    model = FeatureDenseNet(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs
    )
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            (
                r"^(.*denselayer\d+\.(?:norm|relu|conv))"
                r"\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
            )
        )
        state_dict = model_zoo.load_url(densenet_urls["densenet169"])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def feature_densenet201(pretrained=True, **kwargs):
    model = FeatureDenseNet(
        num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs
    )
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            (
                r"^(.*denselayer\d+\.(?:norm|relu|conv))"
                r"\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
            )
        )
        state_dict = model_zoo.load_url(densenet_urls["densenet201"])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model
