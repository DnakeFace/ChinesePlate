import numpy as np
from easydict import EasyDict as edict

config = edict()
config.height = 32
config.width = 100

config.maxLabel = 96
config.text = ' 1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ京津沪渝冀豫云辽黑湘皖鲁新苏浙赣桂甘晋蒙陕吉闽贵粤青藏琼宁川鄂港澳使领学警挂'

#resnet
config.rnet = edict()
config.rnet.hidden = 256

#mobilenetv2
config.mnet = edict()
config.mnet.hidden = 128
config.mnet.multiple = 0.75

def text_encode(data, length):
    label = []
    for k in range(len(data)):
        idx = config.text.find(data[k])
        if idx < 0:
            label.append(1)
        else:
            label.append(idx+1)
    for k in range(length-len(label)):
        label.append(1)
    return label

def text_decode(data, length):
    if length.numel() == 1:
        length = length[0]
        assert data.numel() == length, "text with length: {} does not match declared length: {}".format(data.numel(), length)
        char_list = []
        for i in range(length):
            if data[i] != 0 and (not (i > 0 and data[i - 1] == data[i])):
                if config.text[data[i] - 1] != ' ':
                    char_list.append(config.text[data[i] - 1])
        return ''.join(char_list)
    else:
        assert data.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(data.numel(), length.sum())
        texts = []
        index = 0
        for i in range(length.numel()):
            l = length[i]
            texts.append(config.text(data[index:index + l], torch.IntTensor([l])))
            index += l
        return texts


if __name__ == "__main__":
    for k in range(len(config.text)):
        print('"'+config.text[k]+'",')
