
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import argparse
import logging
import sys
import cv2
import time
import numpy as np
import os
import difflib

from dataset import PlateDatasets
from crnn.crnn_vgg import CRNN_VGG
from crnn.crnn_mnet import CRNN_MNET
from crnn.crnn_rnet import CRNN_RNET

from config import text_decode

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser(description='CRNN')
    parser.add_argument('--network', default="mnet", type=str)
    args = parser.parse_args()
    return args

args = parse_args()

def main():
    pt = './checkpoint/crnn_'+args.network+'_last.pt'

    config = torch.load(pt)['config']

    backbone = torch.load(pt)['backbone']
    backbone.load_state_dict(torch.load(pt)['data'])
    backbone.eval()

    transform = transforms.Compose([transforms.ToTensor()])

    '''
    fp = open("train1.txt", 'w')
    fp2 = open("train1_2.txt", 'w')
    '''

    error = 0
    total = 0
    with torch.no_grad():
        for line in open('../datasets/crnn/val.txt', 'r'):
            ls = line.strip().split()

            im = cv2.imread(ls[0])

            im = cv2.resize(im, (config.width, config.height))
            im = transform(im)
            im = im[None].to(device)

            preds = backbone(im)

            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            preds_size = torch.IntTensor([preds.size(0)] * im.size(0))
            sim_preds = text_decode(preds.data, preds_size.data)
            sim_preds = sim_preds.replace(' ', '')

            total += 1
            if ls[1] != sim_preds:
                error += 1
                name = os.path.basename(ls[0]).split('.')[0]
                print(name, ls[1], sim_preds)

            '''
            if ls[1] == sim_preds:
                fp.write(line)
            elif len(sim_preds) > 5 and difflib.SequenceMatcher(None, sim_preds, ls[1]).quick_ratio() > 0.5:
                fp2.write(line)
            '''

    print('error: ' + str(round(error/total, 4)))

    '''
    fp.close()
    fp2.close()
    '''


if __name__ == "__main__":
    main()
