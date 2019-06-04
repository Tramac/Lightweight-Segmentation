import os
from .mobilenet_seg import *
from .mobilenetv2_seg import *
from .mobilenetv3_seg import *
from .shufflenet_seg import *
from .shufflenetv2_seg import *
from .igcv3_seg import *
from .efficientnet_seg import *


def get_segmentation_model(model, **kwargs):
    models = {
        'mobilenet': get_mobilenet_seg,
        'mobilenetv2': get_mobilenet_v2_seg,
        'mobilenetv3_small': get_mobilenet_v3_small_seg,
        'mobilenetv3_large': get_mobilenet_v3_large_seg,
        'shufflenet': get_shufflenet_seg,
        'shufflenetv2': get_shufflenet_v2_seg,
        'igcv3': get_igcv3_seg,
        'efficientnet': get_efficientnet_seg,
    }
    return models[model](**kwargs)


def get_model_file(name, root='~/.torch/models'):
    root = os.path.expanduser(root)
    file_path = os.path.join(root, name + '.pth')
    if os.path.exists(file_path):
        return file_path
    else:
        raise ValueError('Model file is not found. Downloading or trainning.')
