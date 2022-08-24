import numpy as np
import random
import cv2
import sys
sys.path.append('..')

import torch
from torch.utils import data
from torch.utils.data import DataLoader

from config import config


def motion_blur(image, degree=10, angle=20):
    image = np.array(image)

    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree/2, degree/2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree        
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


class PlateDatasets(data.Dataset):
    def __init__(self, url, transforms=None, im_size=224, enhance=False):
        self.transforms = transforms
        self.im_size = im_size
        self.enhance = enhance

        self.lines = []
        for line in open(url, 'r'):
            self.lines.append(line)


    def __enhance__(self):
        # 随机遮挡
        bgr = np.asarray([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
        ww = np.random.randint(1, self.img.shape[1]//5)
        hh = np.random.randint(1, self.img.shape[0]//3)
        xx = np.random.randint(0, self.img.shape[1]-ww)
        yy = np.random.randint(0, self.img.shape[0]-hh)
        self.img[yy:yy+hh, xx:xx+ww] = bgr

        if np.random.randint(0, 3) == 1:
            lm = self.landmark.reshape(-1, 2) * self.img.shape[1]
            ww = self.img.shape[1]//5
            hh = self.img.shape[0]//5

            left = np.random.randint(0, ww)
            right = np.random.randint(0, ww)
            top = np.random.randint(0, hh)
            bottom = np.random.randint(0, hh)

            self.img = cv2.copyMakeBorder(self.img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
            lm[:, 0] = (lm[:, 0] + left) / self.img.shape[1]
            lm[:, 1] = (lm[:, 1] + top) / self.img.shape[0]
            self.landmark = lm.reshape(-1)
        elif np.random.randint(0, 3) == 1:
            lm = self.landmark.reshape(-1, 2) * self.img.shape[1]
            ww = self.img.shape[1]//10
            hh = self.img.shape[0]//8

            left = np.random.randint(0, ww)
            right = np.random.randint(0, ww)
            top = np.random.randint(0, hh)
            bottom = np.random.randint(0, hh)

            self.img = self.img[top:self.img.shape[0]-bottom, left:self.img.shape[1]-right]
            lm[:, 0] = (lm[:, 0] - left) / self.img.shape[1]
            lm[:, 1] = (lm[:, 1] - top) / self.img.shape[0]
            self.landmark = lm.reshape(-1)

        if np.random.randint(0, 3) == 1:
            if np.random.randint(0, 2) == 1: # 腐蚀
                if img.shape[0] < 160:
                    ks = np.random.randint(3, 17)
                else:
                    ks = np.random.randint(6, 34)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks,ks))
                img = cv2.erode(img, kernel)
            if np.random.randint(0, 2) == 1: # 膨胀
                if img.shape[0] < 160:
                    ks = np.random.randint(3, 17)
                else:
                    ks = np.random.randint(6, 34)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks,ks))
                img = cv2.dilate(img, kernel)

        if np.random.randint(0, 3) == 1:
            if np.random.randint(0, 2) == 1: # 运动模糊
                if self.img.shape[0] < 160:
                    degree = np.random.randint(5, 13)
                else:
                    degree = np.random.randint(15, 25)
                angle = np.random.randint(0, 90)
                self.img = motion_blur(self.img, degree=degree, angle=angle)
            else: # 高斯模糊
                if self.img.shape[0] < 160:
                    ks = (np.random.randint(11, 23) // 2) * 2 + 1
                else:
                    ks = (np.random.randint(29, 45) // 2) * 2 + 1
                self.img = cv2.GaussianBlur(self.img, (ks, ks), sigmaX=0, sigmaY=0)


    def __getitem__(self, index):
        line = self.lines[index].strip().split()

        self.img = cv2.imread(line[0])
        self.landmark = np.asarray(line[1:9], dtype=np.float32)

        if self.enhance: # 数据增强处理
            self.__enhance__()

        if self.img.shape[0] != self.im_size or self.img.shape[1] != self.im_size:
            self.img = cv2.resize(self.img, (self.im_size, self.im_size))
        if self.transforms:
            self.img = self.transforms(self.img)
        return (self.img, self.landmark)


    def __len__(self):
        return len(self.lines)
