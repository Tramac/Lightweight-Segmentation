import os
import sys

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
sys.path.append(root_path)

import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np

from light.model import get_segmentation_model
from train import parse_args


def compute_fps(args):
    if not args.no_cuda and torch.cuda.is_available():
        cudnn.benchmark = True
        args.device = "cuda"
        dtype = torch.cuda.FloatTensor
    else:
        args.distributed = False
        args.device = "cpu"
        dtype = torch.FloatTensor

    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   aux=args.aux, norm_layer=nn.BatchNorm2d).to(args.device)

    x = torch.rand(1, 3, args.crop_size, args.crop_size).type(dtype).to(args.device)
    N = 10
    model.eval()
    with torch.no_grad():
        fpss = list()
        for i in range(10):
            start_time = time.time()
            for n in range(N):
                # print("run: {}/{}".format(n + 1, i + 1))
                out = model(x)
            fpss.append(N / (time.time() - start_time))
        fps = np.mean(fpss)
        print("FPS=%.2f with %s." % (fps, args.device))

    return fps


if __name__ == '__main__':
    args = parse_args()
    fps = compute_fps(args)
