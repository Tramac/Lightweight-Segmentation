"""ShuffleNet"""
import torch.nn as nn

from light.nn import _ConvBNReLU, ShuffleNetUnit

__all__ = ['ShuffleNet', 'get_shufflenet', 'shufflenet_g1', 'shufflenet_g2',
           'shufflenet_g3', 'shufflenet_g4', 'shufflenet_g8']


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=1000, groups=8, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ShuffleNet, self).__init__()
        if groups == 1:
            stages_out_channels = [144, 288, 576]
        elif groups == 2:
            stages_out_channels = [200, 400, 800]
        elif groups == 3:
            stages_out_channels = [240, 480, 960]
        elif groups == 4:
            stages_out_channels = [272, 544, 1088]
        elif groups == 8:
            stages_out_channels = [384, 768, 1536]
        else:
            raise ValueError("Unknown groups.")
        stages_repeats = [3, 7, 3]

        self.conv1 = _ConvBNReLU(3, 24, 3, 2, 1, norm_layer=norm_layer)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        self.in_channels = 24

        self.stage2 = self._make_stage(stages_out_channels[0], stages_repeats[0], groups, norm_layer=norm_layer)

        if dilated:
            self.stage3 = self._make_stage(stages_out_channels[1], stages_repeats[1], groups, 2, norm_layer)
            self.stage4 = self._make_stage(stages_out_channels[2], stages_repeats[2], groups, 2, norm_layer)
        else:
            self.stage3 = self._make_stage(stages_out_channels[1], stages_repeats[1], groups, norm_layer=norm_layer)
            self.stage4 = self._make_stage(stages_out_channels[2], stages_repeats[2], groups, norm_layer=norm_layer)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stages_out_channels[2], num_classes)

    def _make_stage(self, out_channels, repeats, groups, dilation=1, norm_layer=nn.BatchNorm2d):
        stride = 2 if dilation == 1 else 1
        layers = [ShuffleNetUnit(self.in_channels, out_channels, stride, groups, dilation, norm_layer)]
        self.in_channels = out_channels
        for i in range(repeats):
            layers.append(ShuffleNetUnit(self.in_channels, out_channels, 1, groups, norm_layer=norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_shufflenet(groups=1, pretrained=False, root='~/.torch/models', **kwargs):
    model = ShuffleNet(groups=groups, **kwargs)

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def shufflenet_g1(**kwargs):
    return get_shufflenet(1, **kwargs)


def shufflenet_g2(**kwargs):
    return get_shufflenet(2, **kwargs)


def shufflenet_g3(**kwargs):
    return get_shufflenet(3, **kwargs)


def shufflenet_g4(**kwargs):
    return get_shufflenet(4, **kwargs)


def shufflenet_g8(**kwargs):
    return get_shufflenet(8, **kwargs)


if __name__ == '__main__':
    model = shufflenet_g8()
