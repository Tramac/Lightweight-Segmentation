import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['_ConvBNReLU', '_DWConvBNReLU', 'InvertedResidual', '_ASPP', '_FCNHead',
           '_Hswish', '_ConvBNHswish', 'SEModule', 'Bottleneck', 'ShuffleNetUnit',
           'ShuffleNetV2Unit', 'InvertedIGCV3', 'MBConvBlock']


class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _FCNHead(nn.Module):
    def __init__(self, in_channels, channels, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_FCNHead, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, channels, 1)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------
#                      For MobileNet
# -----------------------------------------------------------------
class _DWConvBNReLU(nn.Module):
    """Depthwise Separable Convolution in MobileNet.
    depthwise convolution + pointwise convolution
    """

    def __init__(self, in_channels, dw_channels, out_channels, stride, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DWConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_channels, dw_channels, 3, stride, dilation, dilation, in_channels, norm_layer=norm_layer),
            _ConvBNReLU(dw_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


# -----------------------------------------------------------------
#                      For MobileNetV2
# -----------------------------------------------------------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1, relu6=True, norm_layer=norm_layer))
        layers.extend([
            # dw
            _ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation, dilation,
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# -----------------------------------------------------------------
#                  ASPP: For MobileNetV2
# -----------------------------------------------------------------
class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate1, dilation=rate1, norm_layer=norm_layer)
        self.b2 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate2, dilation=rate2, norm_layer=norm_layer)
        self.b3 = _ConvBNReLU(in_channels, out_channels, 3, padding=rate3, dilation=rate3, norm_layer=norm_layer)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x


# -----------------------------------------------------------------
#                      For MobileNetV3
# -----------------------------------------------------------------
class _Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(_Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.


class _Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(_Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu6(x + 3.) / 6.


class _ConvBNHswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNHswish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.act = _Hswish(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            _Hsigmoid(True)
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out.expand_as(x)


class Identity(nn.Module):
    def __init__(self, in_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, dilation=1, se=False, nl='RE',
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if nl == 'HS':
            act = _Hswish
        else:
            act = nn.ReLU
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, exp_size, 1, bias=False),
            norm_layer(exp_size),
            act(True),
            # dw
            nn.Conv2d(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                      dilation, groups=exp_size, bias=False),
            norm_layer(exp_size),
            SELayer(exp_size),
            act(True),
            # pw-linear
            nn.Conv2d(exp_size, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# -----------------------------------------------------------------
#                      For ShuffleNet
# -----------------------------------------------------------------
def channel_shuffle(x, groups):
    n, c, h, w = x.size()

    channels_per_group = c // groups
    x = x.view(n, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(n, -1, h, w)

    return x


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ShuffleNetUnit, self).__init__()
        self.stride = stride
        self.groups = groups
        self.dilation = dilation
        assert stride in [1, 2, 3]

        inter_channels = out_channels // 4

        if stride > 1:
            self.shortcut = nn.AvgPool2d(3, stride, 1)
            out_channels -= in_channels
        elif dilation > 1:
            out_channels -= in_channels

        g = 1 if in_channels == 24 else groups
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1, groups=g, norm_layer=norm_layer)
        self.conv2 = _ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation,
                                 dilation, groups, norm_layer=norm_layer)
        self.conv3 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, groups=groups, bias=False),
            norm_layer(out_channels))

    def forward(self, x):
        out = self.conv1(x)
        out = channel_shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.stride > 1:
            x = self.shortcut(x)
            out = torch.cat([out, x], dim=1)
        elif self.dilation > 1:
            out = torch.cat([out, x], dim=1)
        else:
            out = out + x
        out = F.relu(out)

        return out


# -----------------------------------------------------------------
#                      For ShuffleNetV2
# -----------------------------------------------------------------
class _DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(_DWConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups=in_channels, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ShuffleNetV2Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ShuffleNetV2Unit, self).__init__()
        assert stride in [1, 2, 3]
        self.stride = stride
        self.dilation = dilation

        inter_channels = out_channels // 2

        if (stride > 1) or (dilation > 1):
            self.branch1 = nn.Sequential(
                _DWConv(in_channels, in_channels, 3, stride, dilation, dilation),
                norm_layer(in_channels),
                _ConvBNReLU(in_channels, inter_channels, 1, norm_layer=norm_layer))
        self.branch2 = nn.Sequential(
            _ConvBNReLU(in_channels if (stride > 1) else inter_channels, inter_channels, 1, norm_layer=norm_layer),
            _DWConv(inter_channels, inter_channels, 3, stride, dilation, dilation),
            norm_layer(inter_channels),
            _ConvBNReLU(inter_channels, inter_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        if (self.stride == 1) and (self.dilation == 1):
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)

        return out


# -----------------------------------------------------------------
#                      For IGCV3
# -----------------------------------------------------------------
class PermutationBlock(nn.Module):
    def __init__(self, groups):
        super(PermutationBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, self.groups, c // self.groups, h, w).permute(0, 2, 1, 3, 4).contiguous().view(n, c, h, w)
        return x


class InvertedIGCV3(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio,
                 dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(InvertedIGCV3, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            # pw
            layers.append(_ConvBNReLU(in_channels, inter_channels, 1,
                                      groups=2, relu6=True, norm_layer=norm_layer))
            # permutation
            layers.append(PermutationBlock(groups=2))
        layers.extend([
            # dw
            _ConvBNReLU(inter_channels, inter_channels, 3, stride, dilation, dilation,
                        groups=inter_channels, relu6=True, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, groups=2, bias=False),
            norm_layer(out_channels),
            # permutation
            PermutationBlock(groups=int(round(out_channels / 2)))
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# -----------------------------------------------------------------
#                      For EfficientNet
# -----------------------------------------------------------------
class _Swish(nn.Module):
    def __init__(self):
        super(_Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class SEModuleV2(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEModuleV2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        se_channels = max(1, int(in_channels * se_ratio))
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, se_channels, 1, bias=False),
            _Swish(),
            nn.Conv2d(se_channels, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x)
        out = self.fc(out)
        return x * out.expand_as(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio,
                 dilation=1, se_ratio=0.25, drop_connect_rate=0.2, norm_layer=nn.BatchNorm2d, **kwargs):
        super(MBConvBlock, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        self.drop_connect_rate = drop_connect_rate
        use_se = (se_ratio is not None) and (0 < se_ratio <= 1.)
        if use_se:
            SELayer = SEModuleV2
        else:
            SELayer = Identity

        layers = list()
        inter_channels = int(round(in_channels * expand_ratio))
        if expand_ratio != 1:
            layers.append(_ConvBNHswish(in_channels, inter_channels, 1, norm_layer=norm_layer))
        layers.extend([
            # dw
            _ConvBNHswish(inter_channels, inter_channels, kernel_size, stride, kernel_size // 2 * dilation, dilation,
                          groups=inter_channels, norm_layer=norm_layer),  # check act function
            SELayer(inter_channels, se_ratio),
            # pw-linear
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        ])
        self.conv = nn.Sequential(*layers)

        if drop_connect_rate:
            self.dropout = nn.Dropout2d(drop_connect_rate)

    def forward(self, x):
        out = self.conv(x)
        if self.use_res_connect:
            if self.drop_connect_rate:
                out = self.dropout(out)
            out = x + out
        return out
