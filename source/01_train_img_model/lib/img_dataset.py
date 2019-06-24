import os

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


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

    def get_item_pos(self, idx, pos, size=288):

        self.num_loaded += 1

        if isinstance(idx, int):
            file_name = self.list_file_names[idx]
        elif isinstance(idx, str):
            file_name = idx
        else:
            raise TypeError('idx must be int or str')

        target_path = os.path.join(self.dir_images, '{}.jpg'.format(file_name))

        img = cv2.cvtColor(cv2.imread(target_path), cv2.COLOR_BGR2RGB)

        h = img.shape[0]
        w = img.shape[1]

        if w > h:
            # wide image
            if pos == 0:
                img = img[:, w // 2 - ceil(h / 2): w // 2 + floor(h / 2), :]
            elif pos == 1:
                img = img[:, :h, :]
            elif pos == 2:
                img = img[:, -h:, :]
            else:
                raise Exception('undefined position')

        elif w < h:
            # narrow image
            if pos == 0:
                img = img[h // 2 - ceil(w / 2): h // 2 + floor(w / 2), :, :]
            elif pos == 1:
                img = img[:w, :, :]
            elif pos == 2:
                img = img[-w:, :, :]
            else:
                raise Exception('undefined position')

        assert img.shape[0] == img.shape[1]

        img = cv2.resize(img, (size, size))
        img = np.array(img).astype(np.float)

        img = self.to_tensor(img) / 255.0

        os.makedirs('_input_imgs', exist_ok=True)
        if self.num_loaded < 20:
            transforms.ToPILImage()(img.float()).save('_input_imgs/img_{}.png'.format(self.num_loaded))

        input_img = self.normalize(img)

        return input_img.float()
