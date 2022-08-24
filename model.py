# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Any, List

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "ShuffleNetV2",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
]


class ShuffleNetV2(nn.Module):

    def __init__(
            self,
            repeats_times: List[int],
            stages_out_channels: List[int],
            num_classes: int = 1000,
    ) -> None:
        super(ShuffleNetV2, self).__init__()
        if stages_out_channels == [24, 244, 488, 976, 2048]:
            self.dropout = True
        else:
            self.dropout = False

        in_channels = stages_out_channels[0]

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, in_channels, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.maxpool = nn.MaxPool2d((3, 3), (2, 2), (1, 1))

        features = []
        for state_repeats_times_index in range(len(repeats_times)):
            out_channels = stages_out_channels[state_repeats_times_index + 1]

            for i in range(repeats_times[state_repeats_times_index]):
                if i == 0:
                    features.append(
                        ShuffleNetV2Unit(
                            in_channels,
                            out_channels,
                            2))
                else:
                    features.append(
                        ShuffleNetV2Unit(
                            in_channels // 2,
                            out_channels,
                            1))
                in_channels = out_channels
        self.features = nn.Sequential(*features)

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels, stages_out_channels[-1], (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(stages_out_channels[-1]),
            nn.ReLU(True)
        )

        self.globalpool = nn.AvgPool2d((7, 7))

        if self.dropout:
            self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(stages_out_channels[-1], num_classes, bias=False),
        )

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.first_conv(x)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.conv_last(out)
        out = self.globalpool(out)
        if self.dropout:
            out = self.dropout(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(module.weight, 0, 0.01)
                else:
                    nn.init.normal_(module.weight, 0, 1.0 / module.weight.shape[1])
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0001)
                nn.init.constant_(module.running_mean, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class ShuffleNetV2Unit(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
    ) -> None:
        super(ShuffleNetV2Unit, self).__init__()
        self.stride = stride
        hidden_channels = out_channels // 2
        out_channels -= in_channels

        if stride == 2:
            self.branch_proj = nn.Sequential(
                # dw
                nn.Conv2d(in_channels, in_channels, (3, 3), (stride, stride), (1, 1), groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                # pw-linear
                nn.Conv2d(in_channels, in_channels, (1, 1), (1, 1), (0, 0), bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
            )
        else:
            self.branch_proj = None

        self.branch_main = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, hidden_channels, (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            # dw
            nn.Conv2d(hidden_channels, hidden_channels, (3, 3), (stride, stride), (1, 1), groups=hidden_channels,
                      bias=False),
            nn.BatchNorm2d(hidden_channels),
            # pw-linear
            nn.Conv2d(hidden_channels, out_channels, (1, 1), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1:
            proj_out, out = channel_shuffle(x)
            out = self.branch_main(out)
            out = torch.cat([proj_out, out], 1)
            return out
        else:
            branch_proj = self.branch_proj(x)
            branch_main = self.branch_main(x)
            out = torch.cat([branch_proj, branch_main], 1)
            return out


def channel_shuffle(x, groups: int = 2):
    batch_size, channels, height, width = x.data.size()
    assert channels % groups == 0
    group_channels = channels // groups

    out = x.reshape(batch_size * group_channels, groups, height * width)
    out = out.permute(1, 0, 2)
    out = out.reshape(2, -1, group_channels, height, width)

    return out[0], out[1]


def shufflenet_v2_x0_5(**kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)

    return model


def shufflenet_v2_x1_0(**kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)

    return model


def shufflenet_v2_x1_5(**kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)

    return model


def shufflenet_v2_x2_0(**kwargs: Any) -> ShuffleNetV2:
    model = ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)

    return model
