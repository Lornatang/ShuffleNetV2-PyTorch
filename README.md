# ShuffleNetV1-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/pdf/1707.01083.pdf)
.

## Table of contents

- [ShuffleNetV1-PyTorch](#shufflenetv1-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test](#test)
        - [Train model](#train-model)
        - [Resume train model](#resume-train-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](#shufflenet-an-extremely-efficient-convolutional-neural-network-for-mobile-devices)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains MNIST, CIFAR10&CIFAR100, TinyImageNet_200, MiniImageNet_1K, ImageNet_1K, Caltech101&Caltech256 and more etc.

- [Google Driver](https://drive.google.com/drive/folders/1f-NSpZc07Qlzhgi6EbBEI1wTkN1MxPbQ?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1arNM38vhDT7p4jKeD4sqwA?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test

- line 29: `model_arch_name` change to `shufflenet_v1_x1_0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/ShuffleNetV1_x1_0-ImageNet_1K-7a092cde.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `shufflenet_v1_x1_0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change
  to `./results/pretrained_models/ShuffleNetV1_x1_0-ImageNet_1K-7a092cde.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `shufflenet_v1_x1_0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/shufflenet_v1_x1_0-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1707.01083.pdf](https://arxiv.org/pdf/1707.01083.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|       Model        |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:------------------:|:-----------:|:-----------------:|:-----------------:|
| shufflenet_v1_x0_5 | ImageNet_1K | 41.2%(**41.1%**)  | 19.0%(**19.0%**)  |
| shufflenet_v1_x1_0 | ImageNet_1K | 32.0%(**31.9%**)  | 13.6%(**13.6%**)  |
| shufflenet_v1_x1_5 | ImageNet_1K | 29.0%(**29.9%**)  | 10.4%(**10.4%**)  |
| shufflenet_v1_x2_0 | ImageNet_1K | 27.1%(**27.0%**)  |  9.2%(**9.2%**)   |

```bash
# Download `ShuffleNetV1_x1_0-ImageNet_1K-7a092cde.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `shufflenet_v1_x1_0` model successfully.
Load `shufflenet_v1_x1_0` model weights `/ShuffleNetV1-PyTorch/results/pretrained_models/ShuffleNetV1_x1_0-ImageNet_1K-7a092cde.pth.tar` successfully.
tench, Tinca tinca                                                          (54.11%)
platypus, duckbill, duckbilled platypus, duck-billed platypus, Ornithorhynchus anatinus (4.75%)
triceratops                                                                 (2.94%)
armadillo                                                                   (2.64%)
barracouta, snoek                                                           (2.63%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

*Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian*

##### Abstract

We introduce an extremely computation-efficient CNN
architecture named ShuffleNet, which is designed specially
for mobile devices with very limited computing power (e.g.,
10-150 MFLOPs). The new architecture utilizes two new
operations, pointwise group convolution and channel shuffle, to greatly reduce computation cost while maintaining
accuracy. Experiments on ImageNet classification and MS
COCO object detection demonstrate the superior performance of ShuffleNet over other structures, e.g. lower top-1
error (absolute 7.8%) than recent MobileNet on ImageNet classification task, under the computation budget of
40 MFLOPs. On an ARM-based mobile device, ShuffleNet
achieves ∼13× actual speedup over AlexNet while maintaining comparable accuracy.

[[Paper]](https://arxiv.org/pdf/1707.01083.pdf)

```bibtex
@inproceedings{zhang2018shufflenet,
            title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
            author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
            pages={6848--6856},
            year={2018}
}
```