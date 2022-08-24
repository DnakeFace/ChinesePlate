#!/usr/bin/env python3
#-*- coding:utf-8 -*-

import argparse
import logging
from pathlib import Path
import time
import os
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.PlateDatasets import PlateDatasets

from pfld.plate.pfld import PFLDInference
import pfld.plate.ResNet as ResNet

from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    logging.info('Save checkpoint to {0:}'.format(filename))


def train(train_loader, pfld_backbone, criterion, optimizer, epoch, batches):
    ts = time.time()
    epoch_losses = []
    batch_losses = []
    pfld_backbone.train()
    for idx, (img, landmark_gt) in enumerate(train_loader):
        img = img.to(device)
        landmark_gt = landmark_gt.to(device)
        pfld_backbone = pfld_backbone.to(device)

        landmarks = pfld_backbone(img)

        loss = criterion(landmark_gt, landmarks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        batch_losses.append(loss.item())
        if (idx+1) % args.show_iter == 0:
            ts = (args.show_iter*args.train_batchsize)/(time.time()-ts)
            logging.info('Epoch[{:0>3d}/{:0>3d}] batch [{:0>4d}-{:0>4d}] speed: {:.1f} loss: {:.6f}'.format(epoch, args.end_epoch, idx+1, batches, ts, np.mean(batch_losses)))
            ts = time.time()
            batch_losses = []

    return np.mean(epoch_losses)


def validate(val_dataloader, pfld_backbone, criterion):
    losses = []
    pfld_backbone.eval()
    with torch.no_grad():
        for img, landmark_gt in val_dataloader:
            img = img.to(device)
            landmark_gt = landmark_gt.to(device)
            pfld_backbone = pfld_backbone.to(device)
            landmarks = pfld_backbone(img)
            loss = criterion(landmark_gt, landmarks)
            losses.append(loss.cpu().numpy())
    return np.mean(losses)


class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmark_gt, landmarks):
        l2_distant = torch.sum((landmark_gt-landmarks)**2, axis=1)
        return torch.mean(l2_distant)


def main(args):
    # Step 1: parse args config
    log_url = './checkpoint/train_plate_'+args.network+'.logs'
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_url, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    # Step 2: model, optimizer, scheduler
    if args.network == 'mnet':
        pfld_backbone = PFLDInference(config.plate.multiple).to(device)
    elif args.network == 'rnet':
        pfld_backbone = ResNet.resnet34(num_classes=8).to(device)
    logging.info(pfld_backbone)

    criterion = PFLDLoss()

    optimizer = torch.optim.Adam([{
        'params': pfld_backbone.parameters()
    }], lr=args.lr, weight_decay=args.weight_decay)

    # 2 epoch lr
    lr_epoch = [int(epoch) for epoch in args.lr_step.split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_epoch, gamma=0.1)

    # step 3: data
    # argumetion
    transform = transforms.Compose([transforms.ToTensor()])

    wlfwdataset = PlateDatasets('/opt/yolo/datasets/plate_lm/train.txt', transforms=transform, im_size=config.plate.size, enhance=False)
    dataloader = DataLoader(wlfwdataset, batch_size=args.train_batchsize, shuffle=True, num_workers=args.workers, drop_last=False)

    val_dataset = PlateDatasets('/opt/yolo/datasets/plate_lm/val.txt', transforms=transform, im_size=config.plate.size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batchsize, shuffle=False, num_workers=args.workers)

    for epoch in range(args.start_epoch, args.end_epoch):
        logging.info("Epoch[%d/%d] lr=%f" % (epoch, args.end_epoch, optimizer.param_groups[0]['lr']))

        train_loss = train(dataloader, pfld_backbone, criterion, optimizer, epoch, len(wlfwdataset)//args.train_batchsize)

        val_loss = validate(val_dataloader, pfld_backbone, criterion)
        logging.info('Eval epoch '+str(epoch)+' Validate loss: {:.6f}'.format(val_loss)+' Train loss: {:.6f}'.format(train_loss))

        filename = os.path.join(str(args.snapshot), 'plate_'+args.network+'_last.pth.tar')
        save_checkpoint({ 'epoch': epoch, 'backbone': pfld_backbone, 'data': pfld_backbone.state_dict() }, filename)

        scheduler.step()


def parse_args():
    parser = argparse.ArgumentParser(description='plate')
    # general
    parser.add_argument('-j', '--workers', default=4, type=int)

    # training
    ##  -- optimizer
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_step', default='15,30,45', type=str)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    parser.add_argument('--network', default="mnet", type=str)

    # -- epoch
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=50, type=int)

    parser.add_argument("--show_iter", default=200, type=int)

    # -- snapshot、tensorboard log and checkpoint
    parser.add_argument('--snapshot', default='./checkpoint/snapshot/', type=str, metavar='PATH')

    parser.add_argument('--resume', default='', type=str, metavar='PATH')

    # --dataset
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=128, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
