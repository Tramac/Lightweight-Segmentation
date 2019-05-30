"""ShuffleNetV2"""
import torch.nn as nn

from light.nn import _ConvBNReLU, ShuffleNetV2Unit

__all__ = ['ShuffleNetV2', 'get_shufflenet_v2', 'shufflenet_v2_0_5', 'shufflenet_v2_1_0',
           'shufflenet_v2_1_5', 'shufflenet_v2_2_0']


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000,
                 dilated=True, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ShuffleNetV2, self).__init__()

        self.conv1 = _ConvBNReLU(3, 24, 3, 2, 1, norm_layer=norm_layer)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.in_channels = 24

        self.stage2 = self._make_stage(stages_out_channels[0], stages_repeats[0], norm_layer=norm_layer)

        if dilated:
            self.stage3 = self._make_stage(stages_out_channels[1], stages_repeats[1], 2, norm_layer)
            self.stage4 = self._make_stage(stages_out_channels[2], stages_repeats[2], 2, norm_layer)
        else:
            self.stage3 = self._make_stage(stages_out_channels[1], stages_repeats[1], norm_layer=norm_layer)
            self.stage4 = self._make_stage(stages_out_channels[2], stages_repeats[2], norm_layer=norm_layer)
        self.conv5 = _ConvBNReLU(self.in_channels, stages_out_channels[-1], 1, norm_layer=norm_layer)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(stages_out_channels[-1], num_classes)

    def _make_stage(self, out_channels, repeats, dilation=1, norm_layer=nn.BatchNorm2d):
        stride = 2 if (dilation == 1) else 1
        layers = [ShuffleNetV2Unit(self.in_channels, out_channels, stride, dilation, norm_layer)]
        self.in_channels = out_channels
        for i in range(repeats - 1):
            # TODO: check dilation
            layers.append(ShuffleNetV2Unit(self.in_channels, out_channels, 1, 1, norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_shufflenet_v2(stages_repeats, stages_out_channels, pretrained=False, root='~/.torch/models', **kwargs):
    model = ShuffleNetV2(stages_repeats, stages_out_channels, **kwargs)

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def shufflenet_v2_0_5(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [48, 96, 192, 1024], **kwargs)


def shufflenet_v2_1_0(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [116, 232, 464, 1024], **kwargs)


def shufflenet_v2_1_5(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [176, 352, 704, 1024], **kwargs)


def shufflenet_v2_2_0(**kwargs):
    return get_shufflenet_v2([4, 8, 4], [244, 488, 976, 2048], **kwargs)


if __name__ == '__main__':
    model = shufflenet_v2_2_0()
