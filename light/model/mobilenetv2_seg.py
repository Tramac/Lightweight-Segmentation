"""MobileNetV1 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from light.model.base import BaseModel
from light.nn import _ASPP, _FCNHead

__all__ = ['MobileNetV2Seg', 'get_mobilenet_v2_seg']


class MobileNetV2Seg(BaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenetv2', pretrained_base=False, **kwargs):
        super(MobileNetV2Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        self.head = _Head(nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(160, nclass, **kwargs)

    def forward(self, x):
        size = x.size()[2:]

        _, _, c3, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _Head(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_Head, self).__init__()
        self.aspp = _ASPP(320, [12, 24, 36], norm_layer=norm_layer, **kwargs)
        self.project = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            norm_layer(256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(256, nclass, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        return self.project(x)


def get_mobilenet_v2_seg(dataset='citys', pretrained=False, root='~/.torch/models',
                         pretrained_base=False, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from light.data import datasets
    model = MobileNetV2Seg(datasets[dataset].NUM_CLASS, backbone='mobilenetv2',
                           pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from ..model import get_model_file
        model.load_state_dict(torch.load(get_model_file('mobilenetv2_%s_best_model' % (acronyms[dataset]), root=root)))
    return model


if __name__ == '__main__':
    model = get_mobilenet_v2_seg()
