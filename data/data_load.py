import os
import torch
import numpy as np
from PIL import Image as Image
from data import PairCompose, PairRandomCrop, PairRandomHorizontalFilp, PairToTensor, PairResize
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True):
    image_dir = os.path.join(path, 'train')

    transform = None
    if use_transform:
        transform = PairCompose(
            [
                # PairRandomCrop(256),
                PairResize(256),
                PairRandomHorizontalFilp(),
                PairToTensor()
            ]
        )
    dataloader = DataLoader(DeblurDataset(image_dir, transform=transform),
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers,
                            pin_memory=True)
    return dataloader


def test_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(DeblurDataset(image_dir, is_test=True),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    return dataloader


def valid_dataloader(path, batch_size=1, num_workers=0):
    image_dir = os.path.join(path, 'test')
    dataloader = DataLoader(DeblurDataset(image_dir, is_test=True),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            pin_memory=True)
    return dataloader


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir, 'raw/'))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir, 'raw', self.image_list[idx]))
        image_d = Image.open(os.path.join(self.image_dir, 'raw_depth', self.image_list[idx]))
        image_g = Image.open(os.path.join(self.image_dir, 'raw_grad', self.image_list[idx]))
        label = Image.open(os.path.join(self.image_dir, 'reference', self.image_list[idx]))

        if self.transform:
            image, image_d, image_g, label = self.transform(image, image_d, image_g, label)
        else:
            image = F.resize(image, (256, 256))
            image_d = F.resize(image_d, (256, 256))
            image_g = F.resize(image_g, (256, 256))
            label = F.resize(label, (256, 256))
            image = F.to_tensor(image)
            image_d = F.to_tensor(image_d)
            image_g = F.to_tensor(image_g)
            label = F.to_tensor(label)
        if self.is_test:
            name = self.image_list[idx]
            return image, image_d, image_g, label, name
        return image, image_d, image_g, label

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError
