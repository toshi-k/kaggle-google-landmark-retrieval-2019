import os
import random

import pandas as pd

import logging
logger = logging.getLogger('root')

from torch.utils.data import Sampler


class SubSampler:

    N = 100

    def __init__(self):

        train_csv = pd.read_csv('../../dataset/train.csv')
        logger.info('number of train images (train.csv): {:d}'.format(len(train_csv)))
        dict_id2landmark = {id: landmark for id, landmark in zip(train_csv.id, train_csv.landmark_id)}

        train_imgs = os.listdir('../../input/train')
        train_ids = [s[:-4] for s in train_imgs]
        train_landmarks = [dict_id2landmark[s] for s in train_ids]

        available_train_data = pd.DataFrame(train_ids, columns=['id'])
        available_train_data['landmark_id'] = train_landmarks

        logger.info('number of train images (avaiable): {:d}'.format(len(available_train_data)))

        landmark_count = available_train_data.landmark_id.value_counts()
        logger.info('number of landmarks: {:d}'.format(len(landmark_count)))

        num_multi_landmark = len(landmark_count[landmark_count > 1])
        self.N = num_multi_landmark

        landmark_top2000 = landmark_count[:self.N].index
        self.landmark_top2000_counts = landmark_count[:self.N].values

        logger.info('maximum number of images in top {} landmarks: {:d}'.format(self.N, landmark_count.iloc[0]))
        logger.info('minimum number of images in top {} landmarks: {:d}'.format(self.N, landmark_count.iloc[self.N-1]))

        train_info = available_train_data[available_train_data.landmark_id.isin(landmark_top2000)]

        logger.info('number of train images (using for training): {:d}'.format(len(train_info)))

        train_info.reset_index(inplace=True, drop=True)
        train_info['index_'] = train_info.index

        self.series_landmark_images = train_info.groupby('landmark_id')['index_'].apply(list)

        self.list_train_imgs = train_info.id.tolist()

        # dict_id2index = {id_: i for i, id_ in enumerate(train_csv.id)}
        #
        # print(train_info)
        # print(dict_id2index)
        # exit()

    def get_train_imgs(self):
        return self.list_train_imgs

    @staticmethod
    def sample_group(l, group_size):
        if len(l) >= group_size:
            return random.sample(l, group_size)
        else:
            pos = random.choice(l)
            copy = list(l)
            copy.remove(pos)
            tail = random.choices(copy, k=group_size-1)
            return [pos] + tail

    def get_sample(self, batch_size, group_size):

        ids = list()

        target_classes = random.sample(range(self.N), batch_size)

        for target_class in target_classes:
            group = self.sample_group(self.series_landmark_images.iloc[target_class], group_size)
            ids.extend(group)

        assert len(ids) == batch_size * group_size

        ids = [id_ for id_ in ids]

        return ids


class PyTorchSampler(Sampler):

    def __init__(self, sub_sampler, batch_size, group_size, iter_inside):

        # don't need
        # super().__init__()

        self.sub_sampler = sub_sampler
        self.iter_inside = iter_inside
        self.batch_size = batch_size
        self.group_size = group_size

        self._i = 0

    def __len__(self):
        return self.iter_inside

    def get_sample(self):

        return self.sub_sampler.get_sample(self.batch_size, self.group_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self._i == self.iter_inside:
            raise StopIteration()

        self._i += 1
        return self.get_sample()
