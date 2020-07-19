""" Compose multiple datasets in a single loader. """

import os
import numpy as np
import torch
import cv2
from copy import deepcopy
from pathlib import Path
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .utils.homographies import sample_homography
from .utils.data_augmentation import photometric_augmentation
from . import get_dataset


class MixedDataset(BaseDataset):
    def __init__(self, config, device):
        super().__init__(config, device)
        base_config = {'sizes': {'train': 30000, 'val': 500, 'test': 1000}}
        base_config.update(self._config)
        self._config = base_config.copy()

        # Initialize the datasets
        self._datasets = []
        for i, d in enumerate(config['datasets']):
            base_config['name'] = d
            base_config['data_path'] = config['data_paths'][i]
            base_config['photometric_augmentation']['enable'] = config[
                'photo_aug'][i]
            self._datasets.append(get_dataset(d)(deepcopy(base_config),
                                                 device))

        self._weights = config['weights']

    def get_dataset(self, split):
        return _Dataset(self._datasets, self._weights,
                        split, self._config['sizes'][split])
    

class _Dataset(Dataset):
    def __init__(self, datasets, weights, split, size):
        self._datasets = [d.get_dataset(split) for d in datasets]
        self._weights = weights
        self._size = size

    def __getitem__(self, item):
        dataset = self._datasets[np.random.choice(
            range(len(self._datasets)), p=self._weights)]
        return dataset[np.random.randint(len(dataset))]

    def __len__(self):
        return self._size
