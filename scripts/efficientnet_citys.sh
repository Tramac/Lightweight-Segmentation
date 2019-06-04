#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=1 python train.py --model efficientnet \
    --dataset citys --lr 1e-2 --epochs 80 --batch-size 2

# eval
CUDA_VISIBLE_DEVICES=1 python eval.py --model efficientnet \
    --dataset citys

# fps
CUDA_VISIBLE_DEVICES=1 python test_fps.py --model efficientnet \
    --dataset citys