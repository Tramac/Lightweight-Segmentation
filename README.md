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
- **Single GPU evaluating**
```
python eval.py --model mobilenet_small --dataset citys
```
- **Multi-GPU evaluating**
```
# for example, evaluate mobilenet with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS eval.py --model mobilenet --dataset citys
```

## Result
- **Cityscapes**

|     Backbone      | OHEM | Params(M) | FLOPs(G) | CPU(fps) | GPU(fps) | mIoU/pixACC |      Model      |
| :---------------: | :--: | :-------: | :------: | :------: | :------: | :---------: | :-------------: |
|     mobilenet     |  ✘   |    5.31   |   4.48   |   0.81   |  77.11   | 0.463/0.901 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1De4ESrHCqdev0nQrKOUzaA)(ybsg) |
|     mobilenet     |  ✓   |    5.31   |   4.48   |   0.81   |  75.35   | 0.526/0.909 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1R3k07vCiYbvz9FztEnAUsw)(u2y2) |
|    mobilenetv2    |  ✓   |    4.88   |   4.04   |   0.49   |  49.40   | 0.613/0.930 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1OWPsDvSjeOM2_VUbPze7gA)(q2g5) |
| mobilenetv3_small |  ✓   |    1.02   |   1.64   |   2.59   |  104.56  | 0.529/0.908 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/15PjAXEQHr136w-B1MalmIg)(e7no) |
| mobilenetv3_large |  ✓   |    2.68   |   4.59   |   1.39   |  79.43   | 0.584/0.916 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1ofXAfN4qDhtsI5kEI90biw)(i60c) |
|     shufflenet    |  ✓   |    6.89   |   5.68   |   0.57   |  43.79   | 0.493/0.901 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1jI2oyoGrTO6JbPp0lL28tw)(6fjh) |
|    shufflenetv2   |  ✓   |    5.24   |   4.33   |   0.72   |  57.71   | 0.528/0.914 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1HZ97h15tz42eMJohyx-H2w)(7pi5) |
|       igcv3       |  ✓   |    4.86   |   4.04   |   0.34   |  29.70   | 0.573/0.923 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1neM8JiGD5an_WXMhrfnxtA)(qe4f) |
|  efficientnet-b0  |  ✓   |    6.63   |   2.60   |   0.33   |  30.15   | 0.492/0.903 | [GoogleDrive](),[BaiduCloud](https://pan.baidu.com/s/1PVXkARVzoOPUHsznwQVZRw)(phuy) |

Note: `crop_size=768, lr=0.01, epochs=80`.

## Support
- [MobileNet](https://arxiv.org/abs/1704.04861)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [ShuffleNet](https://arxiv.org/abs/1707.01083)
- [ShuffleNetV2](https://arxiv.org/abs/1807.11164)
- [IGCV3](https://arxiv.org/pdf/1806.00178)
- [EfficientNet](https://arxiv.org/pdf/1905.11946v1)

## To Do
- [ ] improve performance
- [ ] optimize memory
- [ ] check efficientnet
- [ ] replace `nn.SyncBatchNorm` by [`nn.BatchNorm.convert_sync_batchnorm`](https://pytorch.org/docs/master/nn.html#torch.nn.SyncBatchNorm)
- [ ] check `find_unused_parameters` in `nn.parallel.DistributedDataParallel`

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
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/apache-2-blue.svg
[lic-url]: https://github.com/Tramac/mobilenetv3-segmentation/blob/master/LICENSE
