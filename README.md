# Lightweight Model for Real-Time Semantic Segmentation
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This project aims at providing the popular lightweight model implementations for real-time semantic segmentation.

## Requisites
- PyTorch 1.1
- Python 3.x

## Usage
-----------------
### Train
- **Single GPU training**
```
python train.py --model mobilenet --dataset citys --lr 0.0001 --epochs 240
```
- **Multi-GPU training**
```
# for example, train mobilenet with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model mobilenet --dataset citys --lr 0.0001 --epochs 240
```

### Evaluation
- **Single GPU training**
```
python eval.py --model mobilenet_small --dataset citys
```
- **Multi-GPU training**
```
# for example, evaluate mobilenet with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS --model mobilenet --dataset citys
```

## Result
- **Cityscapes**

|     Backbone      | mIoU/pixACC | Params(M) | FLOPs(G) | CPU(f) | GPU(f) |
| :---------------: | :---------: | :-------: | :------: | :----: | :----: |
|     mobilenet     |             |    5.31   |   4.48   |        |        |
|    mobilenetv2    |             |    4.88   |   8.08   |        |        |
| mobilenetv3_small |             |    1.02   |   3.03   |        |        |
| mobilenetv3_large |             |    2.68   |   8.52   |        |        |


## Support
- [MobileNet](https://arxiv.org/abs/1704.04861)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)


## To Do
- [ ] add eval
- [ ] add shufflenet
- [ ] train and eval

## References
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.0-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: http://dmlc.github.io/img/apache2.svg
[lic-url]: https://github.com/Tramac/mobilenetv3-segmentation/blob/master/LICENSE