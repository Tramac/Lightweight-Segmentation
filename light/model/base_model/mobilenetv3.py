"""Searching for MobileNetV3"""
import torch.nn as nn

from light.nn import _Hswish, _ConvBNHswish, Bottleneck, SEModule

__all__ = ['MobileNetV3', 'get_mobilenet_v3', 'mobilenet_v3_large_1_0', 'mobilenet_v3_small_1_0']


class MobileNetV3(nn.Module):
    def __init__(self, nclass=1000, mode='large', width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d):
        super(MobileNetV3, self).__init__()
        if mode == 'large':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1], ]
            layer2_setting = [
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1], ]
            layer3_setting = [
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 112, True, 'HS', 1], ]
            layer4_setting = [
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1], ]
        elif mode == 'small':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, True, 'RE', 2], ]
            layer2_setting = [
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1], ]
            layer3_setting = [
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1], ]
            layer4_setting = [
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1], ]
        else:
            raise ValueError('Unknown mode.')

        # building first layer
        self.in_channels = int(16 * width_mult) if width_mult > 1.0 else 16
        self.conv1 = _ConvBNHswish(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)

        # building bottleneck blocks
        self.layer1 = self._make_layer(Bottleneck, layer1_setting,
                                       width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(Bottleneck, layer2_setting,
                                       width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(Bottleneck, layer3_setting,
                                       width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting,
                                           width_mult, dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(Bottleneck, layer4_setting,
                                           width_mult, norm_layer=norm_layer)

        # building last several layers
        classifier = list()
        if mode == 'large':
            last_bneck_channels = int(960 * width_mult) if width_mult > 1.0 else 960
            self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(_Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        elif mode == 'small':
            last_bneck_channels = int(576 * width_mult) if width_mult > 1.0 else 576
            self.layer5 = _ConvBNHswish(self.in_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(SEModule(last_bneck_channels))
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(_Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        else:
            raise ValueError('Unknown mode.')
        self.classifier = nn.Sequential(*classifier)

        self._init_weights()

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            exp_channels = int(exp_size * width_mult)
            layers.append(block(self.in_channels, out_channels, exp_channels, k, stride, dilation, se, nl, norm_layer))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.classifier(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def get_mobilenet_v3(mode='small', width_mult=1.0, pretrained=False, root='~/,torch/models', **kwargs):
    model = MobileNetV3(mode=mode, width_mult=width_mult, **kwargs)
    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def mobilenet_v3_large_1_0(**kwargs):
    return get_mobilenet_v3('large', 1.0, **kwargs)


def mobilenet_v3_small_1_0(**kwargs):
    return get_mobilenet_v3('small', 1.0, **kwargs)


if __name__ == '__main__':
    model = mobilenet_v3_large_1_0()
