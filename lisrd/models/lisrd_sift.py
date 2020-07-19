""" Module to train and run LISRD-SIFT. """

import warnings
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func

from .base_model import BaseModel, Mode
from .backbones.net_vlad import NetVLAD
from ..datasets.utils.homographies import warp_points
from ..utils.geometry_utils import keep_true_keypoints


class LisrdSiftModule(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self._config = config
        self._device = device
        self._variances = ['sift', 'upright_sift']
        self.vlad_sift = NetVLAD(
            num_clusters=self._config['n_clusters'],
            dim=self._config['meta_desc_dim'])
        self.vlad_upright_sift = NetVLAD(
            num_clusters=self._config['n_clusters'],
            dim=self._config['meta_desc_dim'])
        self.vlad_layers = {'sift': self.vlad_sift,
                            'upright_sift': self.vlad_upright_sift}
    
    def forward(self, inputs, mode):
        outputs = self._get_sift_desc(inputs)
        self._compute_meta_descriptors(outputs)
        return outputs

    def _get_sift_desc(self, inputs):
        images = np.uint8(inputs.cpu().numpy().transpose(0, 2, 3, 1) * 255)
        descs = {v: [] for v in self._variances}
        keypoints = []
        assignments = []
        tile = self._config['tile']
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_size = np.array(img.shape[:2])
            tile_size = img_size / tile
            sift = cv2.xfeatures2d.SIFT_create(nfeatures=self._config['n_kp'],
                                               contrastThreshold=0.01)
            points = sift.detect(img, None)

            if len(points) == 0:  # No point detected
                keypoints.append(np.zeros((1, 2)))  # Dummy kp
                assignments.append(np.zeros(1, dtype=int))
                for v in self._variances:
                    descs[v].append(np.ones((1, self._config['desc_size'])))
                continue

            for v in self._variances:
                kp = points.copy()
                if v == 'upright_sift':
                    for k in kp:
                        k.angle = 0.  # Set all orientations to 0
                _, desc = sift.compute(img, kp)
                descs[v].append(desc)                

            points = [[k.pt[1], k.pt[0]] for k in points]
            keypoints.append(np.array(points))

            # For each keypoint, compute in which tile it lands up
            ass = np.clip(points // tile_size, 0, tile - 1)
            ass = ass[:, 1] + tile * ass[:, 0]
            assignments.append(ass.astype(int))

        outputs = {'keypoints': keypoints, 'assignments': assignments}
        for v in self._variances:
            outputs[v + '_desc'] = descs[v]
        return outputs

    def _compute_meta_descriptor(self, assignments, descs, netvlad):
        b_size = len(assignments)
        n_tiles = self._config['tile'] * self._config['tile']
        meta_descs = []
        for i in range(b_size):
            meta_desc = []
            for j in range(n_tiles):
                if np.sum(assignments[i] == j) == 0:  # no points in this tile
                    meta_desc.append(  # Dummy meta desc
                        torch.ones(self._config['meta_desc_dim']
                                   * self._config['n_clusters'],
                                   dtype=torch.float, device=self._device))
                    continue
                desc = descs[i][assignments[i] == j]
                desc = desc.reshape(1, self._config['desc_size'], -1, 1)
                desc = torch.tensor(desc, dtype=torch.float,
                                    device=self._device)
                meta_desc.append(netvlad.forward(desc).flatten())
            meta_desc = torch.stack(meta_desc, dim=0)
            meta_descs.append(meta_desc)
        return torch.stack(meta_descs, dim=0)

    def _compute_meta_descriptors(self, outputs):
        """
        For each kind of descriptor, compute a meta descriptor encoding
        a sub area of the total image.
        """
        for v in self._variances:
            outputs[v + '_meta_desc'] = self._compute_meta_descriptor(
                outputs['assignments'], outputs[v + '_desc'],
                self.vlad_layers[v])


class LisrdSift(BaseModel):
    required_config_keys = []

    def __init__(self, dataset, config, device):
        self._device = device
        super().__init__(dataset, config, device)
        self._variances = ['sift', 'upright_sift']

    def _model(self, config):
        return LisrdSiftModule(config, self._device)

    def _forward(self, inputs, mode, config):
        outputs = {}
        if mode == Mode.EXPORT:
            outputs['descriptors'] = {}
            outputs['meta_descriptors'] = {}
            with torch.no_grad():
                output = self._net.forward(inputs['image0'], mode)
            outputs['keypoints'] = output['keypoints']
            outputs['assignments'] = output['assignments']
            for v in self._variances:
                outputs['descriptors'][v] = output[v + '_desc']
                outputs['meta_descriptors'][v] = output[v + '_meta_desc']
        else:
            num_img = 3 if 'image2' in inputs else 2
            for i in range(num_img):
                n = str(i)
                output = self._net.forward(inputs['image' + n], mode)
                outputs['keypoints' + n] = output['keypoints']
                outputs['assignments' + n] = output['assignments']
                for v in self._variances:
                    outputs[v + '_desc' + n] = output[v + '_desc']
                    outputs[v + '_meta_desc' + n] = output[v + '_meta_desc']
        return outputs

    def _loss(self, outputs, inputs, config):
        # Loss for the meta descriptors only
        meta_desc_loss = self._meta_descriptors_loss(outputs, inputs, config)
        return meta_desc_loss

    def _meta_descriptors_loss(self, outputs, inputs, config):
        # Filter out the points not in common between the two images
        H = inputs['homography'].detach().cpu().numpy()
        img_size = np.array(inputs['image0'].size()[2:4])
        b_size = len(H)
        losses = []
        for i in range(b_size):
            kp0, idx0 = keep_true_keypoints(
                outputs['keypoints0'][i], H[i], img_size)
            kp1, idx1 = keep_true_keypoints(
                outputs['keypoints1'][i], np.linalg.inv(H[i]), img_size)
            if (np.sum(idx0) == 0) or (np.sum(idx1) == 0):  # No common points
                return torch.tensor(0, dtype=torch.float, device=self._device,
                                    requires_grad=True)
            assignments0 = outputs['assignments0'][i][idx0]
            assignments1 = outputs['assignments1'][i][idx1]
            
            # Compute the distance between all descriptors
            desc_dists = []
            for v in self._variances:
                desc0 = torch.tensor(outputs[v + '_desc0'][i][idx0],
                                     dtype=torch.float, device=self._device)
                desc1 = torch.tensor(outputs[v + '_desc1'][i][idx1],
                                     dtype=torch.float, device=self._device)
                desc_dist = torch.norm(desc0.unsqueeze(1) - desc1.unsqueeze(0),
                                       dim=2)
                desc_dists.append(desc_dist)
            desc_dists = torch.stack(desc_dists, dim=2)

            # Compute the similarity for each meta descriptor
            meta_desc_sims = []
            for v in self._variances:
                meta_desc0 = outputs[v + '_meta_desc0'][i][assignments0]
                meta_desc0 = func.normalize(meta_desc0, dim=1)
                meta_desc1 = outputs[v + '_meta_desc1'][i][assignments1]
                meta_desc1 = func.normalize(meta_desc1, dim=1)
                meta_desc_sims.append(meta_desc0 @ meta_desc1.t())
            meta_desc_sims = torch.stack(meta_desc_sims, dim=2)

            # Weight the descriptor distances
            meta_desc_sims = func.softmax(meta_desc_sims, dim=2)
            desc_dist = torch.sum(desc_dists * meta_desc_sims, dim=2)

            # Compute correct matches
            warped_kp0 = warp_points(kp0, H[i])
            points_dist = torch.tensor(np.linalg.norm(
                warped_kp0[:, None, :] - kp1[None, :, :], axis=2))
            wrong_matches = points_dist > self._config['correct_thresh']
            dist_mask = points_dist <= self._config['dist_thresh']

            # Positive loss
            pos_desc_dist = desc_dist.clone()
            pos_desc_dist[wrong_matches] = 0.
            pos_dist = torch.max(pos_desc_dist, dim=1)[0]

            # Negative loss
            desc_dist[dist_mask] = torch.tensor(np.inf)
            neg_dist = torch.min(desc_dist, dim=1)[0]

            losses.append(func.relu(config['margin']
                                    + pos_dist - neg_dist).mean())
        
        # Total loss
        loss = torch.stack(losses, dim=0).mean()
        return loss

    def _matching_score(self, outputs, inputs, config):
        # Filter out the points not in common between the two images
        H = inputs['homography'].detach().cpu().numpy()
        img_size = np.array(inputs['image0'].size()[2:4])
        b_size = len(H)
        matching_scores = []
        for i in range(b_size):
            kp0, idx0 = keep_true_keypoints(
                outputs['keypoints0'][i], H[i], img_size)
            kp1, idx1 = keep_true_keypoints(
                outputs['keypoints1'][i], np.linalg.inv(H[i]), img_size)
            if (np.sum(idx0) == 0) or (np.sum(idx1) == 0):  # No common points
                return 0.
            assignments0 = outputs['assignments0'][i][idx0]
            assignments1 = outputs['assignments1'][i][idx1]
            
            # Compute the distance between all descriptors
            desc_dists = []
            for v in self._variances:
                desc0 = torch.tensor(outputs[v + '_desc0'][i][idx0],
                                     dtype=torch.float, device=self._device)
                desc1 = torch.tensor(outputs[v + '_desc1'][i][idx1],
                                     dtype=torch.float, device=self._device)
                desc_dist = torch.norm(desc0.unsqueeze(1) - desc1.unsqueeze(0),
                                       dim=2)
                desc_dists.append(desc_dist)
            desc_dists = torch.stack(desc_dists, dim=2)

            # Compute the similarity for each meta descriptor
            meta_desc_sims = []
            for v in self._variances:
                meta_desc0 = outputs[v + '_meta_desc0'][i][assignments0]
                meta_desc0 = func.normalize(meta_desc0, dim=1)
                meta_desc1 = outputs[v + '_meta_desc1'][i][assignments1]
                meta_desc1 = func.normalize(meta_desc1, dim=1)
                meta_desc_sims.append(meta_desc0 @ meta_desc1.t())
            meta_desc_sims = torch.stack(meta_desc_sims, dim=2)

            # Weight the descriptor distances
            meta_desc_sims = func.softmax(meta_desc_sims, dim=2)
            desc_dist = torch.sum(desc_dists * meta_desc_sims, dim=2)
            desc_dist = desc_dist.detach().cpu().numpy()

            # Compute correct matches
            warped_kp0 = warp_points(kp0, H[i])
            points_dist = np.linalg.norm(
                warped_kp0[:, None, :] - kp1[None, :, :], axis=2)
            best_matches = np.argmin(points_dist, axis=1)
            min_dist = points_dist[np.arange(len(points_dist)), best_matches]
            true_matches = min_dist < self._config['correct_thresh']

            # Compute percentage of correct matches
            closest = np.argmin(desc_dist, axis=1)
            m_score = (0. if np.sum(true_matches) == 0
                       else (closest == best_matches)[true_matches].mean())
            matching_scores.append(m_score)

        return np.stack(matching_scores, axis=0).mean()

    def _metrics(self, outputs, inputs, config):
        m_score = self._matching_score(outputs, inputs, config)
        return {'matching_score': m_score}

    def initialize_weights(self):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

        self._net.apply(init_weights)
