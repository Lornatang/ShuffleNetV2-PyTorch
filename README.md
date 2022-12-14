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
- line 89: `model_weights_path` change to `./results/pretrained_models/ShuffleNetV2_x1_0-ImageNet_1K-25000dee.pth.tar`.

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
  to `./results/pretrained_models/ShuffleNetV2_x1_0-ImageNet_1K-25000dee.pth.tar`.

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
| shufflenet_v2_x0_5 | ImageNet_1K | 38.9%(**38.9%**)  | 17.4%(**17.4%**)  |
| shufflenet_v2_x1_0 | ImageNet_1K | 30.6%(**30.6%**)  | 11.1%(**11.1%**)  |
| shufflenet_v2_x1_5 | ImageNet_1K | 27.4%(**27.4%**)  |  9.4%(**9.4%**)   |
| shufflenet_v2_x2_0 | ImageNet_1K | 25.0%(**25.0%**)  |  7.6%(**7.6%**)   |

```bash
# Download `ShuffleNetV2_x1_0-ImageNet_1K-25000dee.pth.tar` weights to `./results/pretrained_models`
# More detail see `README.md<Download weights>`
python3 ./inference.py 
```

Input:

<span align="center"><img width="224" height="224" src="figure/n01440764_36.JPEG"/></span>

Output:

```text
Build `shufflenet_v2_x1_0` model successfully.
Load `shufflenet_v2_x1_0` model weights `/ShuffleNetV2-PyTorch/results/pretrained_models/ShuffleNetV2_x1_0-ImageNet_1K-25000dee.pth.tar` successfully.
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

*Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian*

##### Abstract

Currently, the neural network architecture design is mostly
guided by the indirect metric of computation complexity, i.e., FLOPs.
However, the direct metric, e.g., speed, also depends on the other factors
such as memory access cost and platform characterics. Thus, this work
proposes to evaluate the direct metric on the target platform, beyond
only considering FLOPs. Based on a series of controlled experiments,
this work derives several practical guidelines for efficient network design. Accordingly, a new architecture is
presented, called ShuffleNet V2.
Comprehensive ablation experiments verify that our model is the stateof-the-art in terms of speed and accuracy tradeoff.

[[Paper]](https://arxiv.org/pdf/1807.11164v1.pdf)

```bibtex
@inproceedings{ma2018shufflenet, 
            title={Shufflenet v2: Practical guidelines for efficient cnn architecture design},  
            author={Ma, Ningning and Zhang, Xiangyu and Zheng, Hai-Tao and Sun, Jian},  
            booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},  
            pages={116--131}, 
            year={2018} 
}
```