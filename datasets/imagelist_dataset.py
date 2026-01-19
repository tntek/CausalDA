# Builds upon: https://github.com/DianCh/AdaContrast/blob/master/image_list.py

import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional
import torch
import torchvision
from torchvision.transforms import autoaugment, transforms
from PIL import Image, ImageFilter
import numpy as np

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from .selectedRotateImageFolder import SelectedRotateImageFolder

logger = logging.getLogger(__name__)

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')
        

# class ImageList(Dataset):
#     def __init__(
#         self,
#         image_root: str,
#         label_files: Sequence[str],
#         transform: Optional[Callable] = None
#     ):
#         self.image_root = image_root
#         self.label_files = label_files
#         self.transform = transform

#         self.samples = []
#         for file in label_files:
#             self.samples += self.build_index(label_file=file)

#     def build_index(self, label_file):
#         # read in items; each item takes one line
#         with open(label_file, "r") as fd:
#             lines = fd.readlines()
#         lines = [line.strip() for line in lines if line]

#         item_list = []
#         for item in lines:
#             img_file, label = item.split()
#             img_path = os.path.join(self.image_root, img_file)
#             domain = img_file.split(os.sep)[0]
#             item_list.append((img_path, int(label), domain))

#         return item_list

#     def __getitem__(self, idx):
#         img_path, label, domain = self.samples[idx]
#         img = Image.open(img_path).convert("RGB")
#         if self.transform:
#             img = self.transform(img)

#         return img, label,idx

#     def __len__(self):
#         return len(self.samples)

class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform

        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)

        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                   std=[0.26862954, 0.26130258, 0.27577711])
        crop_size = 224
        self.rf_1 = transforms.Compose([
                transforms.Resize(crop_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),#​ToTensor()​​​将​​shape​​​为​​(H, W, C)​​​的​​nump.ndarray​​​或​​img​​​转为​​shape​​​为​​(C, H, W)​​​的​​tensor​​​，其将每一个数值归一化到​​[0,1]​​，其归一化方法比较简单，直接除以255即可
                normalize
            ])

    def build_index(self, label_file):
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            domain = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain))

        return item_list

    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            data = self.transform(img)

        img_1 = self.rf_1(img)
        re_ls = [img_1]
        
        return (data, re_ls), label, idx


    def __len__(self):
        return len(self.samples)  

class ImageList_idx_aug_fix(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        self.ra_obj = autoaugment.RandAugment()
        #self.ra_obj = RandAugment(2,9)#数据增强
        self.committee_size = 1
        resize_size = 256 
        crop_size = 224 
        #对大于crop_size的图片进行随机裁剪，训练阶段是随机裁剪，验证阶段是随机裁剪或中心裁剪
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#归一化
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                   std=[0.26862954, 0.26130258, 0.27577711])
                                
        # RandomRotate_1 = ts.transforms.RandomRotate(0.5)#以一定的概率（0.5）对图像在[-rotate_range, rotate_range]角度范围内进行旋转
        self.rf_1 = transforms.Compose([
                transforms.Resize(crop_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),#​ToTensor()​​​将​​shape​​​为​​(H, W, C)​​​的​​nump.ndarray​​​或​​img​​​转为​​shape​​​为​​(C, H, W)​​​的​​tensor​​​，其将每一个数值归一化到​​[0,1]​​，其归一化方法比较简单，直接除以255即可
                normalize
            ])
        #用Compose把多个步骤整合到一起
        imgs = make_dataset(image_list, labels)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)

        if self.transform is not None:
            data = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target)

        img_1 = self.rf_1(img)
        re_ls = [img_1]
        return (data, re_ls), target, index

    def __len__(self):
        return len(self.imgs)
