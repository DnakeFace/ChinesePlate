#!/usr/bin/env python3
# -*- coding:utf-8 -*-

######################################################
#
# pfld.py -
# written by  zhaozhichao
#
######################################################

import torch
import torch.nn as nn
import math
import sys

def conv_bn(inp, oup, kernel, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True) )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = use_res_connect

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, multiple=1.0):
        super(PFLDInference, self).__init__()

        Sx64 = int(64*multiple)
        Sx128 = int(128*multiple)

        self.conv1 = nn.Conv2d(3, Sx64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(Sx64)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(Sx64, Sx64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(Sx64)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3_1 = InvertedResidual(Sx64, Sx64, 2, False, 2)
        self.block3_2 = InvertedResidual(Sx64, Sx64, 1, True, 2)
        self.block3_3 = InvertedResidual(Sx64, Sx64, 1, True, 2)
        self.block3_4 = InvertedResidual(Sx64, Sx64, 1, True, 2)
        self.block3_5 = InvertedResidual(Sx64, Sx64, 1, True, 2)

        self.conv4_1 = InvertedResidual(Sx64, Sx128, 2, False, 2)
        self.conv5_1 = InvertedResidual(Sx128, Sx128, 1, False, 4)
        self.block5_2 = InvertedResidual(Sx128, Sx128, 1, True, 4)
        self.block5_3 = InvertedResidual(Sx128, Sx128, 1, True, 4)
        self.block5_4 = InvertedResidual(Sx128, Sx128, 1, True, 4)
        self.block5_5 = InvertedResidual(Sx128, Sx128, 1, True, 4)
        self.block5_6 = InvertedResidual(Sx128, Sx128, 1, True, 4)

        self.conv6_1 = InvertedResidual(Sx128, 16, 1, False, 2)  # [16, 14, 14]

        self.conv7 = conv_bn(16, 32, 3, 2)  # [32, 7, 7]
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)  # [128, 1, 1]
        self.relu8 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(176, 2 * 4)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):  # x: 3, 112, 112
        x = self.relu1(self.bn1(self.conv1(x)))  # [64, 56, 56]
        x = self.relu2(self.bn2(self.conv2(x)))  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        x = self.block3_5(x)

        x = self.conv4_1(x)
        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.block5_5(x)
        x = self.block5_6(x)

        x = self.conv6_1(x)
        x1 = self.avg_pool(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.relu8(self.conv8(x))
        x3 = x3.view(x3.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)
        return landmarks
