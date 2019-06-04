"""Base Model for Semantic Segmentation"""
import torch.nn as nn

from light.model.base_model import mobilenet1_0, mobilenet_v2_1_0, mobilenet_v3_small_1_0, mobilenet_v3_large_1_0, \
    shufflenet_g8, shufflenet_v2_2_0, igcv3_1_0, efficientnet_b0


class BaseModel(nn.Module):
    def __init__(self, nclass, aux=False, backbone='mobilenet', pretrained_base=False, **kwargs):
        super(BaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.backbone = backbone

        if backbone == 'mobilenet':
            self.pretrained = mobilenet1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv2':
            self.pretrained = mobilenet_v2_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'mobilenetv3_large':
            self.pretrained = mobilenet_v3_large_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'shufflenet':
            self.pretrained = shufflenet_g8(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'shufflenetv2':
            self.pretrained = shufflenet_v2_2_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'igcv3':
            self.pretrained = igcv3_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        elif backbone == 'efficientnet':
            self.pretrained = efficientnet_b0(dilated=True, pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError("Unknown backnone: {}".format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        if self.backbone in ['mobilenet', 'mobilenetv2', 'mobilenetv3_small', 'mobilenetv3_large', 'igcv3',
                             'efficientnet']:
            x = self.pretrained.conv1(x)
            x = self.pretrained.layer1(x)
            c1 = self.pretrained.layer2(x)
            c2 = self.pretrained.layer3(c1)
            c3 = self.pretrained.layer4(c2)
            c4 = self.pretrained.layer5(c3)
            if self.backbone == 'efficientnet':
                c4 = self.pretrained.layer6(c4)
        elif self.backbone in ['shufflenet', 'shufflenetv2']:
            x = self.pretrained.conv1(x)
            c1 = self.pretrained.maxpool(x)
            c2 = self.pretrained.stage2(c1)
            c3 = self.pretrained.stage3(c2)
            c4 = self.pretrained.stage4(c3)
        else:
            raise ValueError

        return c1, c2, c3, c4


if __name__ == '__main__':
    model = BaseModel(19)
