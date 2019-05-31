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

|     Backbone      | Params(M) | FLOPs(G) | CPU(fps) | GPU(fps) | mIoU/pixACC |
| :---------------: | :-------: | :------: | :------: | :------: | :---------: |
|     mobilenet     |    5.31   |   4.48   |          |          | 0.457/0.920 |
|    mobilenetv2    |    4.88   |   4.04   |          |          | 0.459/0.924 |
| mobilenetv3_small |    1.02   |   1.64   |          |          | 0.415/0.909 |
| mobilenetv3_large |    2.68   |   4.59   |          |          |             |
|     shufflenet    |    6.89   |   5.68   |          |          |             |
|    shufflenetv2   |    5.24   |   4.33   |          |          |             |
|       igcv3       |    4.86   |   4.04   |          |          |             |

## Support
- [MobileNet](https://arxiv.org/abs/1704.04861)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [ShuffleNet](https://arxiv.org/abs/1707.01083)
- [ShuffleNetV2](https://arxiv.org/abs/1807.11164)
- [IGCV3](https://arxiv.org/pdf/1806.00178)


## To Do
- [ ] check dilation use
- [x] add eval
- [ ] add squeezenet, condensenet, shiftnet, mnasnet
- [ ] train and eval

## References
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [mobilenetv3-segmentation](https://github.com/Tramac/mobilenetv3-segmentation)
- [awesome-semantic-segmentation-pytorch](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)

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