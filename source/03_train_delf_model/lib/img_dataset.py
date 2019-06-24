import os

import numpy as np

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from delf import feature_io


def ceil(v):
    return int(v)


def floor(v):
    return -int(-v // 1)


class LandmarkDataset(Dataset):

    def __init__(self, dir_images, list_train_imgs=None):
        self.dir_images = dir_images

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if list_train_imgs is None:
            self.list_file_names = sorted(os.listdir(self.dir_images))
        else:
            self.list_file_names = list_train_imgs

        self.num_loaded = 0

    def __len__(self):
        return len(self.list_file_names)

    def __getitem__(self, idx):

        return self.get_item_pos(idx, pos=0)

    def get_item_pos(self, idx, pos):

        self.num_loaded += 1

        if isinstance(idx, int):
            file_name = self.list_file_names[idx]
        elif isinstance(idx, str):
            file_name = idx
        else:
            raise TypeError('idx must be int or str')

        target_path = os.path.join(self.dir_images, '{}.delf'.format(file_name))

        try:
            locations_1, scales_1, descriptors, _, _ = feature_io.ReadFromFile(target_path)
            num_features_1 = locations_1.shape[0]
        except ValueError:
            # not enough values to unpack (expected 5, got 4)
            return torch.randn(41, 1000).float()

        descriptors = np.swapaxes(descriptors, 0, 1)
        zeros = np.zeros((41, 1000))
        zeros[:40, :num_features_1] = descriptors
        zeros[40, :num_features_1] = (scales_1 - 0.5) / 10.0

        return torch.FloatTensor(zeros)
