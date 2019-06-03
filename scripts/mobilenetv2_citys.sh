#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=1 python train.py --model mobilenetv2 \
    --dataset citys --lr 1e-2 --epochs 80

# eval
CUDA_VISIBLE_DEVICES=1 python eval.py --model mobilenetv2 \
    --dataset citys --aux

# fps
CUDA_VISIBLE_DEVICES=1 python test_fps.py --model mobilenetv2 \
    --dataset citys --aux