"""IGCV3"""
import torch.nn as nn

from light.nn import _ConvBNReLU, InvertedIGCV3

__all__ = ['IGCV3', 'get_igcv3', 'igcv3_1_0', 'igcv3_0_5', 'igcv3_0_75', 'igcv3_0_25']


class IGCV3(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(IGCV3, self).__init__()
        layer1_setting = [
            # t, c, n, s
            [1, 16, 1, 1]]
        layer2_setting = [
            [6, 24, 4, 2]]
        layer3_setting = [
            [6, 32, 6, 2]]
        layer4_setting = [
            [6, 64, 8, 2],
            [6, 96, 6, 1]]
        layer5_setting = [
            [6, 160, 6, 2],
            [6, 320, 1, 1]]
        # building first layer
        self.in_channels = int(32 * width_mult) if width_mult > 1.0 else 32
        last_channels = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv1 = _ConvBNReLU(3, self.in_channels, 3, 2, 1, relu6=True, norm_layer=norm_layer)

        # building inverted residual blocks
        self.layer1 = self._make_layer(InvertedIGCV3, layer1_setting, width_mult, norm_layer=norm_layer)
        self.layer2 = self._make_layer(InvertedIGCV3, layer2_setting, width_mult, norm_layer=norm_layer)
        self.layer3 = self._make_layer(InvertedIGCV3, layer3_setting, width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(InvertedIGCV3, layer4_setting, width_mult, dilation=2, norm_layer=norm_layer)
            self.layer5 = self._make_layer(InvertedIGCV3, layer5_setting, width_mult, dilation=2, norm_layer=norm_layer)
        else:
            self.layer4 = self._make_layer(InvertedIGCV3, layer4_setting, width_mult, norm_layer=norm_layer)
            self.layer5 = self._make_layer(InvertedIGCV3, layer5_setting, width_mult, norm_layer=norm_layer)

        # building last several layers
        self.classifier = nn.Sequential(
            _ConvBNReLU(self.in_channels, last_channels, 1, relu6=True, norm_layer=norm_layer),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.2),
            nn.Conv2d(last_channels, num_classes, 1))

    # def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
    #     layers = list()
    #     for t, c, n, s in block_setting:
    #         out_channels = int(c * width_mult)
    #         for i in range(n):
    #             stride = s if (i == 0 and dilation == 1) else 1
    #             layers.append(block(self.in_channels, out_channels, stride, t, dilation, norm_layer=norm_layer))
    #             self.in_channels = out_channels
    #     return nn.Sequential(*layers)

    def _make_layer(self, block, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for t, c, n, s in block_setting:
            out_channels = int(c * width_mult)
            stride = s if (dilation == 1) else 1
            layers.append(block(self.in_channels, out_channels, stride, t, dilation, norm_layer=norm_layer))
            self.in_channels = out_channels
            for i in range(n - 1):
                layers.append(block(self.in_channels, out_channels, 1, t, 1, norm_layer=norm_layer))
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
        x = x.view(x.size(0), -1)
        return x

    def _init_weight(self):
        # weight initialization
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


def get_igcv3(width_mult=1.0, pretrained=False, root='~/.torch/models', **kwargs):
    model = IGCV3(width_mult=width_mult, **kwargs)

    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def igcv3_1_0(**kwargs):
    return get_igcv3(1.0, **kwargs)


def igcv3_0_75(**kwargs):
    return get_igcv3(0.75, **kwargs)


def igcv3_0_5(**kwargs):
    return get_igcv3(0.5, **kwargs)


def igcv3_0_25(**kwargs):
    return get_igcv3(0.25, **kwargs)


if __name__ == '__main__':
    model = igcv3_1_0()
