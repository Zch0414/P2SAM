import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image


class BasicDataset(Dataset):
    def __init__(self, data, transform, split='tr', target_length=1024, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]):
        self.data = data
        self.transform = transform
        self.split = split
        self.target_length = target_length
        self.pixel_mean = torch.Tensor(pixel_mean).view(-1, 1, 1) if pixel_mean is not None else None
        self.pixel_std = torch.Tensor(pixel_std).view(-1, 1, 1) if pixel_std is not None else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        batch = dict()

        # debug
        image_name = str(self.data[idx]['image']).split('/')[-1].split('.')[0].split('_')[-1]
        label_name = str(self.data[idx]['label']).split('/')[-1].split('.')[0].split('_')[-1]
        assert image_name == label_name
        batch['name'] = image_name

        # load
        image = Image.open(self.data[idx]['image'])
        image = np.array(image)
        label = Image.open(self.data[idx]['label'])
        label = np.array(label)
        if len(label.shape) == 3:
            label = label[:, :, 0]

        # record
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        if self.split == 'ts' or self.split == 'val':
            batch['original_size'] = torch.Tensor([image.shape[0], image.shape[1]]).int()
            batch['resize_size'] = torch.Tensor(target_size).int()
            batch['ignore_mask'] = torch.Tensor(self.get_ignore_mask(image))
        
        # resiz
        image = np.array(resize(to_pil_image(image), target_size))
        label = np.array(resize(to_pil_image(label), target_size))[..., None]

        # augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
        
        # totensor
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = torch.from_numpy(label).permute(2, 0, 1)

        # normalization
        if self.pixel_mean is not None and self.pixel_std is not None:
            image = (image - self.pixel_mean) / self.pixel_std
        else:
            image = (image - image.min()) / torch.clamp(image.max() - image.min(), min=1e-8, max=None) 
        
        # padding
        batch['image'] = self.pad(image)
        batch['label'] = self.pad(label)

        return batch

    def pad(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        padh = self.target_length - h
        padw = self.target_length - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int):
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return [newh, neww]
    
    @staticmethod
    def get_ignore_mask(img):
        img = img.astype(np.uint8)
        
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(gray, dtype=np.int8)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(mask, [contours[0]], -1, 1, -1)

        mask = mask * 255
        return mask[None, ...]


def get_file_list(path, suffix):
    return sorted([p for p in Path(path).rglob(suffix)])


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return


def split_dataset(data, sets, split=None):
    total_len = len(data)
    indices = torch.arange(0, total_len) if split is None else split

    if len(sets) == 1:
        ts_data = data
        return None, None, ts_data

    if len(sets) == 2:
        tr_data = [data[i] for i in indices[:sets[0]]]
        val_data = [data[i] for i in indices[sets[0]:sets[0]+sets[1]]]
        return tr_data, val_data, None
    
    if len(sets) == 3:
        tr_data = [data[i] for i in indices[:sets[0]]]
        val_data = [data[i] for i in indices[sets[0]:sets[0]+sets[1]]]
        ts_data = [data[i] for i in indices[sets[0]+sets[1]:sets[0]+sets[1]+sets[2]]]
        return tr_data, val_data, ts_data
