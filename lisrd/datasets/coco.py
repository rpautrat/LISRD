""" MS COCO dataset. """

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from torch.utils.data import Dataset

from .base_dataset import BaseDataset
from .utils.data_reader import resize_and_crop
from .utils.homographies import (sample_homography, compute_valid_mask,
                                 get_keypoints_and_mask)
from .utils.data_augmentation import photometric_augmentation


class Coco(BaseDataset):
    def __init__(self, config, device):
        super().__init__(config, device)
        root_dir = Path(os.path.expanduser(config['data_path']))
        self._paths = {}
        
        # Train split
        train_dir = Path(root_dir, 'train2014')
        self._paths['train'] = [str(p) for p in list(train_dir.iterdir())]

        # Val split
        val_dir = Path(root_dir, 'val2014')
        val_images = list(val_dir.iterdir())
        self._paths['val'] = [str(p)
                              for p in val_images[:config['sizes']['val']]]

        # Test split
        self._paths['test'] = [str(p)
                               for p in val_images[-config['sizes']['test']:]]

    def get_dataset(self, split):
        return _Dataset(self._paths[split], self._config, self._device)


class _Dataset(Dataset):
    def __init__(self, paths, config, device):
        self._paths = paths
        self._config = config
        self._angle_lim = np.pi / 4
        self._device = device

    def __getitem__(self, item):
        img0 = cv2.imread(self._paths[item])
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)

        # Resize the image
        if 'img_size' in self._config:
            img0 = resize_and_crop(img0, self._config['img_size'])
        img_size = img0.shape[:2]

        # Warp the image
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
        img1 = cv2.warpPerspective(img0, H, (img_size[1], img_size[0]),
                                   flags=cv2.INTER_LINEAR)
        img2 = cv2.warpPerspective(img0, H_no_rot, (img_size[1], img_size[0]),
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
            img1 = photometric_augmentation(img1, config_aug)
            img2 = photometric_augmentation(img2, config_aug)

        # Normalize the images in [0, 1]
        img0 = img0.astype(float) / 255.
        img1 = img1.astype(float) / 255.
        img2 = img2.astype(float) / 255.

        outputs = {'light_invariant': False}
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
        return len(self._paths)