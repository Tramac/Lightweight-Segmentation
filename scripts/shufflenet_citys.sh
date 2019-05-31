#!/usr/bin/env bash

# train
CUDA_VISIBLE_DEVICES=1 python train.py --model shufflenet \
    --dataset citys --aux --ohem True --lr 1e-2 --epochs 80

# eval
CUDA_VISIBLE_DEVICES=1 python eval.py --model shufflenet \
    --dataset citys --aux

# fps
CUDA_VISIBLE_DEVICES=1 python test_fps.py --model shufflenet \
    --dataset citys --aux