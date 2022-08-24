
from torch.utils import data
from torchvision import transforms

import cv2
import random
import numpy as np

from config import config, text_encode


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

def rotate(image, scale=1.0):
    angle = np.random.randint(-5, 6)
    (h, w) = image.shape[:2]

    rw = w // 10
    rh = h // 10
    rw = np.random.randint(-rw, rw)
    rh = np.random.randint(-rh, rh)
    cx = (w // 2) + rw
    cy = (h // 2) + rh

    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    im = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
    return im

def gama_transfer(img, power1):
    '''Gamma 变换进行图像增强，power1 为 Gamma 变换因子'''
    img = 255*np.power(img/255, power1)
    img = np.around(img)
    img[img>255] = 255
    out_img = img.astype(np.uint8)
    return out_img


class PlateDatasets(data.Dataset):
    def __init__(self, url, transforms=None, enhance=False):
        self.transforms = transforms
        self.enhance = enhance

        self.lines = []
        for line in open(url, 'r'):
            self.lines.append(line)

    def __getitem__(self, index):
        line = self.lines[index].strip().split()

        img = cv2.imread(line[0])

        text = line[1]
        label = text_encode(text, 8)

        if self.enhance:
            # 随机遮挡
            for k in range(5):
                bgr = np.asarray([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
                ww = np.random.randint(1, 5)
                hh = np.random.randint(1, 5)
                xx = np.random.randint(0, img.shape[1]-ww)
                yy = np.random.randint(0, img.shape[0]-hh)
                img[yy:yy+hh, xx:xx+ww] = bgr

            if np.random.randint(0, 4) == 1:
                ww = img.shape[1]//16
                hh = img.shape[0]//10
                left = np.random.randint(0, ww)
                right = img.shape[1] - np.random.randint(0, ww)
                top = np.random.randint(0, hh)
                bottom = img.shape[0] - np.random.randint(0, hh)
                img = img[top:bottom, left:right]
            if np.random.randint(0, 4) == 1:
                ww = img.shape[1]//16
                hh = img.shape[0]//10
                left = np.random.randint(0, ww)
                right = np.random.randint(0, ww)
                top = np.random.randint(0, hh)
                bottom = np.random.randint(0, hh)
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
            if np.random.randint(0, 4) == 1:
                img = rotate(img)

            if np.random.randint(0, 3) == 1:
                if np.random.randint(0, 2) == 1: # 腐蚀
                    if img.shape[0] < 48:
                        ks = np.random.randint(1, 5)
                    else:
                        ks = np.random.randint(2, 10)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks,ks))
                    img = cv2.erode(img, kernel)
                if np.random.randint(0, 2) == 1: # 膨胀
                    if img.shape[0] < 48:
                        ks = np.random.randint(1, 5)
                    else:
                        ks = np.random.randint(2, 10)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks,ks))
                    img = cv2.dilate(img, kernel)

            if np.random.randint(0, 3) == 1:
                if np.random.randint(0, 3) == 1: # 运动模糊
                    if img.shape[0] < 48:
                        degree = np.random.randint(1, 5)
                    else:
                        degree = np.random.randint(2, 10)
                    angle = np.random.randint(0, 90)
                    img = motion_blur(img, degree=degree, angle=angle)
                else: # 高斯模糊
                    if img.shape[0] < 48:
                        ks = (np.random.randint(3, 7) // 2) * 2 + 1
                    else:
                        ks = (np.random.randint(5, 15) // 2) * 2 + 1
                    img = cv2.GaussianBlur(img, (ks, ks), sigmaX=0, sigmaY=0)


        if img.shape[0] != config.height or img.shape[1] != config.width:
            img = cv2.resize(img, (config.width, config.height))

        if self.transforms:
            img = self.transforms(img)
        return (img, np.array(label), len(label), text)

    def __len__(self):
        return len(self.lines)
