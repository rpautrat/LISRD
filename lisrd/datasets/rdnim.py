""" Rotated Day-Night Image Matching dataset. """

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .utils.data_reader import read_timestamps
from ..utils.geometry_utils import select_k_best
from ..utils.pytorch_utils import keypoints_to_grid


class Rdnim(BaseDataset):
    def __init__(self, config, device):
        super().__init__(config, device)
        root_dir = Path(os.path.expanduser(config['data_path']))
        ref = config['reference']

        # Extract the timestamps
        timestamp_files = [p for p in Path(root_dir, 'time_stamps').iterdir()]
        timestamps = {}
        for f in timestamp_files:
            id = f.stem
            timestamps[id] = read_timestamps(str(f))

        # Extract the reference images paths
        references = {}
        seq_paths = [p for p in Path(root_dir, 'references').iterdir()]
        for seq in seq_paths:
            id = seq.stem
            references[id] = str(Path(seq, ref + '.jpg'))

        # Extract the images paths and homographies
        seq_path = [p for p in Path(root_dir, 'images').iterdir()]
        self._paths = {'test': []}
        for seq in seq_path:
            id = seq.stem
            images_path = [x for x in seq.iterdir() if x.suffix == '.jpg']
            for img in images_path:
                timestamp = timestamps[id]['time'][
                    timestamps[id]['name'].index(img.name)]
                H = np.loadtxt(str(img)[:-4] + '.txt').astype(float)
                self._paths['test'].append({
                    'img': str(img), 'ref': str(references[id]),
                    'H': H, 'timestamp': timestamp})

    def get_dataset(self, split):
        assert split == 'test', 'RDNIM only available in test mode.'
        return _Dataset(self._paths[split], self._config)


class _Dataset(Dataset):
    def __init__(self, paths, config):
        self._paths = paths
        self._config = config

    def __getitem__(self, item):
        img0_path = self._paths[item]['ref']
        img0 = cv2.cvtColor(cv2.imread(img0_path), cv2.COLOR_BGR2RGB)
        img1_path = self._paths[item]['img']
        img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
        img_size = img0.shape[:2]
        
        # Normalize the images in [0, 1]
        img0 = img0.astype(float) / 255.
        img1 = img1.astype(float) / 255.
        
        # Compute valid masks
        H = self._paths[item]['H']
        H_inv = np.linalg.inv(H)

        # Extract the keypoints and descriptors of each method
        features = {}
        for m in self._config['models_name']:
            features[m] = {}
            feat0 = np.load(self._paths[item]['ref'] + '.' + m)
            feat1 = np.load(self._paths[item]['img'] + '.' + m)

            # Extract a fixed number of shared keypoints between the images
            kp0, mask0 = select_k_best(
                feat0['keypoints'][:, [1, 0]], feat0['scores'],
                self._config['num_kp'], H, img_size, margin=3)
            features[m]['keypoints0'] = kp0
            kp1, mask1 = select_k_best(
                feat1['keypoints'][:, [1, 0]], feat1['scores'],
                self._config['num_kp'], H_inv, img_size, margin=3)
            features[m]['keypoints1'] = kp1

            # Extract the local descriptors
            features[m]['descriptors0'] = feat0['descriptors'][mask0]
            features[m]['descriptors1'] = feat1['descriptors'][mask1]

            # Extract meta descriptors if they exist
            if 'meta_descriptors' in feat0:
                meta_desc0_t = torch.tensor(feat0['meta_descriptors'])
                grid0 = keypoints_to_grid(torch.tensor(kp0),
                                          img_size).repeat(4, 1, 1, 1)
                features[m]['meta_descriptors0'] = F.normalize(F.grid_sample(
                    meta_desc0_t,
                    grid0).squeeze(3).permute(2, 0, 1), dim=2).numpy()
                meta_desc1_t = torch.tensor(feat1['meta_descriptors'])
                grid1 = keypoints_to_grid(torch.tensor(kp1),
                                          img_size).repeat(4, 1, 1, 1)
                features[m]['meta_descriptors1'] = F.normalize(F.grid_sample(
                    meta_desc1_t,
                    grid1).squeeze(3).permute(2, 0, 1), dim=2).numpy()
                
        return {'img0': img0, 'img1': img1, 'img_size': img_size,
                'timestamp': self._paths[item]['timestamp'],
                'features': features, 'homography': H}

    def __len__(self):
        return len(self._paths)
