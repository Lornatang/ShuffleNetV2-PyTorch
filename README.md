# ShuffleNetV2-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/pdf/1807.11164v1.pdf)
.

## Table of contents

- [ShuffleNetV2-PyTorch](#shufflenetv2-pytorch)
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
        - [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](#shufflenet-v2-practical-guidelines-for-efficient-cnn-architecture-design)

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

- line 29: `model_arch_name` change to `shufflenet_v2_x1_0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `test`.
- line 89: `model_weights_path` change to `./results/pretrained_models/ShuffleNetV2_x1_0-ImageNet_1K-7a092cde.pth.tar`.

```bash
python3 test.py
```

### Train model

- line 29: `model_arch_name` change to `shufflenet_v2_x1_0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 50: `pretrained_model_weights_path` change
  to `./results/pretrained_models/ShuffleNetV2_x1_0-ImageNet_1K-7a092cde.pth.tar`.

```bash
python3 train.py
```

### Resume train model

- line 29: `model_arch_name` change to `shufflenet_v2_x1_0`.
- line 31: `model_mean_parameters` change to `[0.485, 0.456, 0.406]`.
- line 32: `model_std_parameters` change to `[0.229, 0.224, 0.225]`.
- line 34: `model_num_classes` change to `1000`.
- line 36: `mode` change to `train`.
- line 53: `resume` change to `./samples/shufflenet_v2_x1_0-ImageNet_1K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper results: [https://arxiv.org/pdf/1807.11164v1.pdf](https://arxiv.org/pdf/1807.11164v1.pdf))

In the following table, the top-x error value in `()` indicates the result of the project, and `-` indicates no test.

|       Model        |   Dataset   | Top-1 error (val) | Top-5 error (val) |
|:------------------:|:-----------:|:-----------------:|:-----------------:|
| shufflenet_v2_x0_5 | ImageNet_1K | 38.9%(**41.1%**)  | 17.4%(**19.0%**)  |
| shufflenet_v2_x1_0 | ImageNet_1K | 30.6%(**31.9%**)  | 11.1%(**13.6%**)  |
| shufflenet_v2_x1_5 | ImageNet_1K | 27.4%(**29.9%**)  |  9.4%(**10.4%**)  |
| shufflenet_v2_x2_0 | ImageNet_1K | 25.0%(**27.0%**)  |  7.6%(**9.2%**)   |

```bash
# Download `ShuffleNetV2_x1_0-ImageNet_1K-7a092cde.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `shufflenet_v2_x1_0` model successfully.
Load `shufflenet_v2_x1_0` model weights `/ShuffleNetV2-PyTorch/results/pretrained_models/ShuffleNetV2_x1_0-ImageNet_1K-7a092cde.pth.tar` successfully.
tench, Tinca tinca                                                          (84.78%)
barracouta, snoek                                                           (2.71%)
gar, garfish, garpike, billfish, Lepisosteus osseus                         (0.43%)
coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch    (0.28%)
American lobster, Northern lobster, Maine lobster, Homarus americanus       (0.25%)
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

### Credit

#### ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design

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

[[Paper]](https://arxiv.org/pdf/1807.11164v1.pdf)

```bibtex
@inproceedings{zhang2018shufflenet,
            title={Shufflenet: An extremely efficient convolutional neural network for mobile devices},
            author={Zhang, Xiangyu and Zhou, Xinyu and Lin, Mengxiao and Sun, Jian},
            booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
            pages={6848--6856},
            year={2018}
}
```