"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from light.model.base import BaseModel

__all__ = ['MobileNetV3Seg', 'get_mobilenet_v3_large_seg', 'get_mobilenet_v3_small_seg']


class MobileNetV3Seg(BaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        mode = backbone.split('_')[-1]
        self.head = _Head(nclass, mode, **kwargs)
        if aux:
            inter_channels = 40 if mode == 'large' else 24
            self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c2)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _Head(nn.Module):
    def __init__(self, nclass, mode='small', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()
        in_channels = 960 if mode == 'large' else 576
        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)


class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x


def get_mobilenet_v3_large_seg(dataset='citys', pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from light.data import datasets
    model = MobileNetV3Seg(datasets[dataset].NUM_CLASS, backbone='mobilenetv3_large',
                           pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        import get_model_file
        model.load_state_dict(
            torch.load(get_model_file('mobilenet_v3_large_%s_best_model' % (acronyms[dataset]), root=root)))
    return model


def get_mobilenet_v3_small_seg(dataset='citys', pretrained=False, root='~/.torch/models',
                               pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from light.data import datasets
    model = MobileNetV3Seg(datasets[dataset].NUM_CLASS, backbone='mobilenetv3_small',
                           pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from ..model import get_model_file
        model.load_state_dict(
            torch.load(get_model_file('mobilenet_v3_small_%s_best_model' % (acronyms[dataset]), root=root)))
    return model


if __name__ == '__main__':
    model = get_mobilenet_v3_small_seg()
