#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=1 python train.py --model mobilenetv3 \
    --dataset citys --aux --ohem True --lr 1e-2

# eval
CUDA_VISIBLE_DEVICES=1 python eval.py --model mobilenetv3 \
    --dataset citys --aux

# fps
CUDA_VISIBLE_DEVICES=1 python test_fps.py --model mobilenetv3 \
    --dataset citys --aux