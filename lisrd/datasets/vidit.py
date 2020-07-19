""" Virtual Image Dataset for Illumination Transfer (VIDIT). """

import os
import numpy as np
from copy import deepcopy
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .utils.data_reader import resize_and_crop
from .utils.homographies import (sample_homography, compute_valid_mask,
                                 get_keypoints_and_mask)
from .utils.data_augmentation import photometric_augmentation


class Vidit(BaseDataset):
    def __init__(self, config, device):
        super().__init__(config, device)
        self._root_dir = Path(os.path.expanduser(config['data_path']))
        self._paths = {}
        np.random.seed(config['seed'])

        # Define the images paths
        files = [str(path) for path in self._root_dir.iterdir()]
        files = np.sort(files).reshape(300, 40)
        self._paths['train'] = files[10:]
        self._paths['val'] = files[:10]
        self._paths['test'] = []

    def get_dataset(self, split):
        return _Dataset(self._paths[split], self._config, self._device)


class _Dataset(Dataset):
    def __init__(self, paths, config, device):
        self._paths = paths
        self._config = config
        self._angle_lim = np.pi / 4
        self._device = device

    def __getitem__(self, item):
        scene_id = item // 50
        paths = np.random.choice(self._paths[scene_id], 2, replace=False)
        img0 = cv2.cvtColor(cv2.imread(paths[0]), cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(cv2.imread(paths[1]), cv2.COLOR_BGR2RGB)

        # Resize the images
        if 'img_size' in self._config:
            img0 = resize_and_crop(img0, self._config['img_size'])
            img1 = resize_and_crop(img1, self._config['img_size'])
        img_size = img0.shape[:2]

        # Sample a homography and warp img1
        rot_invar = True
        self._config['warped_pair']['params']['rotation'] = False
        H_no_rot, _ = sample_homography(
            img_size, **self._config['warped_pair']['params'])
        if (np.random.rand()
            < self._config['warped_pair']['no_rot_proba']):
            rot_invar = False
        else:
            self._config['warped_pair']['params']['rotation'] = True
        H, rot_angle = sample_homography(
            img_size, **self._config['warped_pair']['params'])
        rot_angle = np.clip(np.abs(rot_angle) / self._angle_lim, 0., 1.)
        self._config['warped_pair']['params']['rotation'] = True
        H_inv = np.linalg.inv(H)
        img1 = cv2.warpPerspective(img1, H, (img_size[1], img_size[0]),
                                   flags=cv2.INTER_LINEAR)
        img2 = cv2.warpPerspective(
            img0, H_no_rot, (img_size[1], img_size[0]),
            flags=cv2.INTER_LINEAR)

        # Compute SIFT keypoints (optional)
        compute_sift = self._config.get('compute_sift', False)
        if compute_sift:
            kp_lists, mask = get_keypoints_and_mask(
                [img0, img1, img2], H, H_no_rot, n_kp=self._config['n_kp'])

        # Apply photometric augmentation
        config_aug = self._config['photometric_augmentation']
        if config_aug['enable']:
            img0 = photometric_augmentation(img0, config_aug)
            img2 = photometric_augmentation(img2, config_aug)
            # Add light augmentation
            light_aug = deepcopy(config_aug)
            light_aug['primitives'] += [
                'random_brightness', 'random_contrast', 'additive_shade']
            img1 = photometric_augmentation(img1, light_aug)

        # Normalize the images in [0, 1]
        img0 = img0.astype(float) / 255.
        img1 = img1.astype(float) / 255.
        img2 = img2.astype(float) / 255.
        
        outputs = {'light_invariant': True}
        if compute_sift:
            outputs['valid_mask'] = torch.tensor(mask, dtype=torch.float,
                                                 device=self._device)
            outputs['keypoints0'] = torch.tensor(
                kp_lists[0], dtype=torch.float, device=self._device)
            outputs['keypoints1'] = torch.tensor(
                kp_lists[1], dtype=torch.float, device=self._device)
            outputs['keypoints2'] = torch.tensor(
                kp_lists[2], dtype=torch.float, device=self._device)
        else:
            # Compute valid masks
            valid_mask2_0 = compute_valid_mask(
                np.linalg.inv(H_no_rot), img_size,
                self._config['warped_pair']['valid_border_margin'])
            valid_mask2_2 = compute_valid_mask(
                H_no_rot, img_size,
                self._config['warped_pair']['valid_border_margin'])
            valid_mask0 = compute_valid_mask(
                H_inv, img_size,
                self._config['warped_pair']['valid_border_margin'])
            valid_mask0 *= valid_mask2_0
            outputs['valid_mask0'] = torch.tensor(
                valid_mask0, dtype=torch.float, device=self._device)
            valid_mask1 = compute_valid_mask(
                H, img_size,
                self._config['warped_pair']['valid_border_margin'])
            valid_mask1 *= valid_mask2_2
            outputs['valid_mask1'] = torch.tensor(
                valid_mask1, dtype=torch.float, device=self._device)

        outputs['image0'] = torch.tensor(
            img0.transpose(2, 0, 1), dtype=torch.float,
            device=self._device)
        outputs['image1'] = torch.tensor(
            img1.transpose(2, 0, 1), dtype=torch.float,
            device=self._device)
        outputs['image2'] = torch.tensor(
            img2.transpose(2, 0, 1), dtype=torch.float,
            device=self._device)
        outputs['homography'] = torch.tensor(H, dtype=torch.float,
                                             device=self._device)
        outputs['H_no_rot'] = torch.tensor(H_no_rot, dtype=torch.float,
                                           device=self._device)

        # Useful additional information
        outputs['rot_invariant'] = rot_invar
        outputs['rot_angle'] = torch.tensor([rot_angle], dtype=torch.float,
                                            device=self._device)

        return outputs

    def __len__(self):
        # 50 image pairs per scene
        return len(self._paths) * 50