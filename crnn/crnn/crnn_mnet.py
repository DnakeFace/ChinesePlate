"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import math

# reference form : https://github.com/moskomule/senet.pytorch  
class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        #assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, nHeight=32, nMultiple=1.0, se=True):
        super(MobileNetV2, self).__init__()

        # setting of inverted residual blocks
        # standard mobilenet v2
        if nHeight == 32:
            self.cfgs = [
                # t   c    n    s     se
                [ 1,  64,  2,   1,    False ],
                [ 4,  96,  3,  (2,1), True ],
                [ 4, 160,  2,  (2,1), True ],
                [ 4, 320,  1,  (2,1), True ],
            ]
        elif nHeight ==64:
            self.cfgs = [
                # t   c    n    s     se
                [ 1,  32,  2,   1,    False ],
                [ 1,  64,  2,   2,    False ],
                [ 4,  96,  3,  (2,1), True ],
                [ 4, 160,  2,  (2,1), True ],
                [ 4, 320,  1,  (2,1), True ],
            ]
        else:
            assert "the height must be 32 or 64"

        # building first layer

        input_channel = _make_divisible(32 * nMultiple, 4 if nMultiple == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]

        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s, _se in self.cfgs:
            output_channel = _make_divisible(c * nMultiple, 4 if nMultiple == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel

            # 注意力机制 -- by CQC
            if se and _se:
                layers.append(SELayer(input_channel))

        layers.append(nn.MaxPool2d((2, 2), (2, 2), (0, 1)))

        self.layers = nn.Sequential(*layers)
        self.output_channel = output_channel

        self._initialize_weights()

    def forward(self, x):
        x = self.layers(x)
        return x

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

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output


class CRNN_MNET(nn.Module):
    def __init__(self, nChannel=3, nHeight=32, nClass=128, nHidden=256, nMultiple=1.0, log_softmax=False):
        super(CRNN_MNET, self).__init__()

        self.log_softmax = log_softmax

        self.backbone = MobileNetV2(nHeight=nHeight, nMultiple=nMultiple)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(self.backbone.output_channel, nHidden, nHidden),
            BidirectionalLSTM(nHidden, nHidden, nClass))

    def forward(self, x):
        x = self.backbone(x)
        #print(x.shape)

        b, c, h, w = x.size()
        assert h == 1, "the height of conv must be 1"
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        x = self.rnn(x)
        if self.log_softmax:
            x = nn.functional.log_softmax(x, dim=2)
        else:
            x = nn.functional.softmax(x, dim=2)
        return x


if __name__ == "__main__":
    nHeight = 64
    dummy = torch.randn(1, 3, nHeight, 120)
    crnn = CRNN_MNET(nHeight=nHeight)
    crnn(dummy)
