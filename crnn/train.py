
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import argparse
import logging
import sys
import cv2
import time
import numpy as np

from dataset import PlateDatasets
from crnn.crnn_vgg import CRNN_VGG
from crnn.crnn_mnet import CRNN_MNET
from crnn.crnn_rnet import CRNN_RNET

from config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_args(args):
    for arg in vars(args):
        s = arg + ': ' + str(getattr(args, arg))
        logging.info(s)


def save_checkpoint(state, filename='crnn.pt'):
    torch.save(state, filename)


def main(args):
    log_url = './checkpoint/train_crnn_'+args.network+'.logs'
    logging.basicConfig(
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_url, mode='w'),
            logging.StreamHandler()
        ])
    print_args(args)

    if args.network == 'mnet':
        backbone = CRNN_MNET(nChannel=3, nHeight=config.height, nClass=config.maxLabel, nHidden=config.mnet.hidden, nMultiple=config.mnet.multiple, log_softmax=True).to(device)
    elif args.network == 'rnet':
        backbone = CRNN_RNET(nChannel=3, nHeight=config.height, nClass=config.maxLabel, nHidden=config.rnet.hidden, log_softmax=True).to(device)
    logging.info(backbone)

    criterion = torch.nn.CTCLoss().to(device)

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = PlateDatasets('../datasets/crnn/train.txt', transforms=transform, enhance=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batchsize, num_workers=args.workers, shuffle=True, drop_last=False)

    val_dataset = PlateDatasets('../datasets/crnn/val.txt', transforms=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batchsize, num_workers=args.workers, shuffle=False, drop_last=False)

    optimizer = torch.optim.Adam([{
        'params': backbone.parameters()
    }], lr=args.lr, weight_decay=args.weight_decay)

    lr_epoch = [int(epoch) for epoch in args.lr_step.split(',')]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_epoch, gamma=args.lr_gamma)

    show_iter = args.show_iter
    for epoch in range(args.start_epoch, args.end_epoch):
        logging.info("Epoch[%d/%d] lr=%f" % (epoch, args.end_epoch, optimizer.param_groups[0]['lr']))

        batch_losses = []
        epoch_losses = []
        ts = time.time()
        backbone.train()
        for idx, (img, label, length, text) in enumerate(train_dataloader):
            img = img.to(device)
            preds = backbone(img)
            p_size = torch.IntTensor([preds.size(0)] * img.size(0))
            loss = criterion(preds, label, p_size, length)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            epoch_losses.append(loss.item())
            if (idx+1) % show_iter == 0:
                ts = (show_iter * args.train_batchsize) / (time.time() - ts)
                logging.info('Epoch[%d/%d] Batch[%d/%d] Speed: %.1f Losses: %.4f'%(epoch, args.end_epoch, idx+1, len(train_dataset)//args.train_batchsize, ts, np.mean(batch_losses)))
                batch_losses = []
                ts = time.time()

        losses = []
        backbone.eval()
        with torch.no_grad():
            for idx, (img, label, length, text) in enumerate(val_dataloader):
                img = img.to(device)
                preds = backbone(img)
                p_size = torch.IntTensor([preds.size(0)] * img.size(0))
                loss = criterion(preds, label, p_size, length)
                losses.append(loss.item())

        logging.info('Eval Validate Losses: {:.4f} Train Losses: {:.4f}'.format(np.mean(losses), np.mean(epoch_losses)))

        scheduler.step()

        filename = './checkpoint/crnn_'+args.network+'_last.pt'
        save_checkpoint({
            'epoch': epoch,
            'config': config,
            'backbone': backbone,
            'data': backbone.state_dict()
        }, filename)


def parse_args():
    parser = argparse.ArgumentParser(description='crnn')
    # general
    parser.add_argument('-j', '--workers', default=8, type=int)

    parser.add_argument('--network', default="mnet", type=str)

    # training
    ##  -- optimizer
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--lr_gamma', default=0.2, type=float)
    parser.add_argument('--lr_step', default='15,30,40', type=str)
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float)

    # -- epoch
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--end_epoch', default=50, type=int)

    parser.add_argument("--show_iter", default=200, type=int)

    # --dataset
    parser.add_argument('--train_batchsize', default=128, type=int)
    parser.add_argument('--val_batchsize', default=128, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
