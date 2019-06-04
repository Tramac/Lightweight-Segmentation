"""EfficientNet"""
import torch.nn as nn

from light.nn import _ConvBNHswish, MBConvBlock

__all__ = ['EfficientNet', 'get_efficientnet', 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
           'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


class EfficientNet(nn.Module):
    def __init__(self, width_coe, depth_coe, depth_divisor=8, min_depth=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, num_classes=1000, dilated=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(EfficientNet, self).__init__()
        self.width_coe = width_coe
        self.depth_coe = depth_coe
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate  # not use

        layer1_setting = [
            # k, c, n, s, t
            [3, 16, 1, 1, 1]]
        layer2_setting = [
            [3, 24, 2, 2, 6]]
        layer3_setting = [
            [5, 40, 2, 2, 6],
            [3, 80, 3, 1, 6]]
        layer4_setting = [
            [5, 112, 3, 2, 6]]
        layer5_setting = [
            [5, 192, 4, 2, 6]]
        layer6_setting = [
            [3, 320, 1, 1, 6]]

        # building first layer
        self.in_channels = self.round_filter(32, width_coe, depth_divisor, min_depth)
        self.conv1 = _ConvBNHswish(3, self.in_channels, 3, 2, 1, norm_layer=norm_layer)  # diff from origin

        # building MBConvBlocks
        self.layer1 = self._make_layer(MBConvBlock, layer1_setting, norm_layer=norm_layer)
        self.layer2 = self._make_layer(MBConvBlock, layer2_setting, norm_layer=norm_layer)
        self.layer3 = self._make_layer(MBConvBlock, layer3_setting, norm_layer=norm_layer)
        if dilated:
            self.layer4 = self._make_layer(MBConvBlock, layer4_setting, 2, norm_layer)
            self.layer5 = self._make_layer(MBConvBlock, layer5_setting, 2, norm_layer)
        else:
            self.layer4 = self._make_layer(MBConvBlock, layer4_setting, norm_layer=norm_layer)
            self.layer5 = self._make_layer(MBConvBlock, layer5_setting, norm_layer=norm_layer)
        self.layer6 = self._make_layer(MBConvBlock, layer6_setting, norm_layer=norm_layer)

        # building last several layers
        last_channels = self.round_filter(1280, width_coe, depth_divisor, min_depth)
        self.classifier = nn.Sequential(
            _ConvBNHswish(self.in_channels, last_channels, 1, norm_layer=norm_layer),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(last_channels, num_classes, 1))

    def _make_layer(self, block, block_setting, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()

        for k, c, n, s, t in block_setting:
            out_channels = self.round_filter(c, self.width_coe, self.depth_divisor, self.min_depth)
            stride = s if dilation == 1 else 1
            layers.append(block(self.in_channels, out_channels, k, stride, t, dilation, norm_layer=norm_layer))
            self.in_channels = out_channels
            for i in range(n - 1):
                layers.append(block(self.in_channels, out_channels, k, 1, t, 1, norm_layer=norm_layer))
                self.in_channels = out_channels
        return nn.Sequential(*layers)

    @classmethod
    def round_filter(cls, filters, width_coe, depth_divisor, min_depth):
        if not width_coe:
            return filters
        filters *= width_coe
        min_depth = min_depth or depth_divisor
        new_filter = max(min_depth, int(filters + depth_divisor / 2) // depth_divisor * depth_divisor)
        if new_filter < 0.9 * filters:  # prevent rounding by more than 10%
            new_filter += depth_divisor
        return int(new_filter)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x


def get_efficientnet(params, pretrained=False, root='~/,torch/models', **kwargs):
    w, d, _, p = params
    model = EfficientNet(w, d, dropout_rate=p, **kwargs)
    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def efficientnet_b0(**kwargs):
    return get_efficientnet([1.0, 1.0, 224, 0.2], **kwargs)


def efficientnet_b1(**kwargs):
    return get_efficientnet([1.0, 1.1, 240, 0.2], **kwargs)


def efficientnet_b2(**kwargs):
    return get_efficientnet([1.1, 1.2, 260, 0.3], **kwargs)


def efficientnet_b3(**kwargs):
    return get_efficientnet([1.2, 1.4, 300, 0.3], **kwargs)


def efficientnet_b4(**kwargs):
    return get_efficientnet([1.4, 1.8, 380, 0.4], **kwargs)


def efficientnet_b5(**kwargs):
    return get_efficientnet([1.6, 2.2, 456, 0.4], **kwargs)


def efficientnet_b6(**kwargs):
    return get_efficientnet([1.8, 2.6, 528, 0.5], **kwargs)


def efficientnet_b7(**kwargs):
    return get_efficientnet([2.0, 3.1, 600, 0.5], **kwargs)


if __name__ == '__main__':
    model = efficientnet_b7()
