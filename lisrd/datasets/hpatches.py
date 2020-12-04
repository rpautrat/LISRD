""" HPatches dataset. """

import os
import numpy as np
import cv2
import logging
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .utils.data_reader import resize_and_crop
from ..utils.geometry_utils import select_k_best
from ..utils.pytorch_utils import keypoints_to_grid


class Hpatches(BaseDataset):
    def __init__(self, config, device):
        super().__init__(config, device)

    def get_dataset(self, split):
        assert split == 'test', 'Only test split supported'
        return _Dataset(self._config)


class _Dataset(Dataset):
    def __init__(self, config):
        self._config = config   
        root_dir = Path(os.path.expanduser(config['data_path']))
        folder_paths = [x for x in root_dir.iterdir() if x.is_dir()]
        if len(folder_paths) == 0:
            raise ValueError(
                f'Could not find any image in folder: {root_dir}.')
        logging.info(f'Found {len(folder_paths)} scenes in image folder.')
        self._image0_paths = []
        self._image1_paths = []
        self._homographies = []
        for path in folder_paths:
            if (config['alteration'] != 'all'
                and config['alteration'] != path.stem[0]):
                continue
            for i in range(2, 7):
                self._image0_paths.append(str(Path(path, "1.ppm")))
                self._image1_paths.append(str(Path(path, str(i) + '.ppm')))
                self._homographies.append(
                    np.loadtxt(str(Path(path, "H_1_" + str(i)))))
    
    def adapt_homography_to_preprocessing(self, H, img_shape0, img_shape1):
        source_size0 = np.array(img_shape0, dtype=float)
        source_size1 = np.array(img_shape1, dtype=float)
        target_size = np.array(self._config['resize'], dtype=float)
        
        # Get the scaling factor in resize
        scale0 = np.amax(target_size / source_size0)
        scaling0 = np.diag([1. / scale0, 1. / scale0, 1.]).astype(float)
        scale1 = np.amax(target_size / source_size1)
        scaling1 = np.diag([scale1, scale1, 1.]).astype(float)

        # Get the translation params in crop
        pad_y0 = (source_size0[0] * scale0 - target_size[0]) / 2.
        pad_x0 = (source_size0[1] * scale0 - target_size[1]) / 2.
        translation0 = np.array([[1., 0., pad_x0],
                                 [0., 1., pad_y0],
                                 [0., 0., 1.]], dtype=float)
        pad_y1 = (source_size1[0] * scale1 - target_size[0]) / 2.
        pad_x1 = (source_size1[1] * scale1 - target_size[1]) / 2.
        translation1 = np.array([[1., 0., -pad_x1],
                                 [0., 1., -pad_y1],
                                 [0., 0., 1.]], dtype=float)

        return translation1 @ scaling1 @ H @ scaling0 @ translation0

    def __getitem__(self, item):
        img0_path = self._image0_paths[item]
        img0 = cv2.imread(img0_path)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1_path = self._image1_paths[item]
        img1 = cv2.imread(img1_path)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        H = self._homographies[item].astype(float)

        if 'resize' in self._config:
            H = self.adapt_homography_to_preprocessing(H, img0.shape[:2],
                                                       img1.shape[:2])
            img0 = resize_and_crop(img0, self._config['resize'])
            img1 = resize_and_crop(img1, self._config['resize'])
        img_size = img0.shape[:2]
        H_inv = np.linalg.inv(H)

        img0 = img0.astype(float) / 255.
        img1 = img1.astype(float) / 255.

        # Extract the keypoints and descriptors of each method
        features = {}
        for m in self._config['models_name']:
            features[m] = {}
            feat0 = np.load(img0_path + '.' + m)
            feat1 = np.load(img1_path + '.' + m)

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
        
        return {'image0': img0, 'image1': img1, 'homography': H,
                'img0_path': img0_path, 'img1_path': img1_path,
                'features': features, 'img_size': img_size}

    def __len__(self):
        return len(self._homographies)