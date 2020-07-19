""" Rotated Day-Night Image Matching dataset. """

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .utils.data_reader import read_timestamps


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

        # Extract the images paths and the homographies
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
    
    def _compute_valid_mask(self, H, img_size, erosion_radius=0.):
        mask = np.ones(img_size, dtype=float)
        mask = cv2.warpPerspective(mask, H, (img_size[1], img_size[0]),
                                   flags=cv2.INTER_NEAREST)
        if erosion_radius > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (erosion_radius * 2, ) * 2)
            mask = cv2.erode(mask, kernel)
        return mask

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
        H = self._files[item]['H']
        H_inv = np.linalg.inv(H)
        valid_mask0 = self._compute_valid_mask(H_inv, img_size, 3)
        valid_mask1 = self._compute_valid_mask(H, img_size, 3)

        img0 = torch.tensor(img0.transpose(2, 0, 1), dtype=torch.float,
                            device=self._device)
        img1 = torch.tensor(img1.transpose(2, 0, 1), dtype=torch.float,
                            device=self._device)
        H = torch.tensor(H, dtype=torch.float, device=self._device)
        valid_mask0 = torch.tensor(valid_mask0, dtype=torch.float,
                                   device=self._device)
        valid_mask1 = torch.tensor(valid_mask1, dtype=torch.float,
                                   device=self._device)
        
        return {'image0': img0, 'image1': img1, 'homography': H,
                'valid_mask0': valid_mask0, 'valid_mask1': valid_mask1,
                'timestamp': self._files[item]['timestamp'],
                'img0_path': img0_path, 'img1_path': img1_path}

    def __len__(self):
        return len(self._paths)
