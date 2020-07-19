""" Module to train and run LISRD. """

import warnings
warnings.filterwarnings(action='once')
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as func

from .base_model import BaseModel, Mode
from .backbones.vgg import VGGLikeModule
from .backbones.net_vlad import NetVLAD
from ..utils.metrics import matching_score
from ..utils.losses import (invar_desc_triplet_loss, var_desc_triplet_loss,
                            meta_desc_triplet_loss)


class LisrdModule(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self._config = config
        self.relu = torch.nn.ReLU(inplace=True)
        self._backbone = VGGLikeModule()
        self._variances = ['rot_var_illum_var', 'rot_invar_illum_var',
                           'rot_var_illum_invar', 'rot_invar_illum_invar']
        self._desc_size = self._config['desc_size']

        # Illum variant, rotation variant branch
        self.conv_illum_var_rot_var_1 = nn.Conv2d(256, 256, kernel_size=3,
                                                  stride=1, padding=1)
        self.bn_illum_var_rot_var_1 = nn.BatchNorm2d(256)
        self.conv_illum_var_rot_var_2 = nn.Conv2d(
            256, self._desc_size, kernel_size=1, stride=1, padding=0)
        
        # Illum variant, rotation invariant branch
        self.conv_illum_var_rot_invar_1 = nn.Conv2d(256, 256, kernel_size=3,
                                                    stride=1, padding=1)
        self.bn_illum_var_rot_invar_1 = nn.BatchNorm2d(256)
        self.conv_illum_var_rot_invar_2 = nn.Conv2d(
            256, self._desc_size, kernel_size=1, stride=1, padding=0)
        
        # Illum invariant, rotation variant branch
        self.conv_illum_invar_rot_var_1 = nn.Conv2d(256, 256, kernel_size=3,
                                                    stride=1, padding=1)
        self.bn_illum_invar_rot_var_1 = nn.BatchNorm2d(256)
        self.conv_illum_invar_rot_var_2 = nn.Conv2d(
            256, self._desc_size, kernel_size=1, stride=1, padding=0)
        
        # Illum invariant, rotation invariant branch
        self.conv_illum_invar_rot_invar_1 = nn.Conv2d(256, 256, kernel_size=3,
                                                      stride=1, padding=1)
        self.bn_illum_invar_rot_invar_1 = nn.BatchNorm2d(256)
        self.conv_illum_invar_rot_invar_2 = nn.Conv2d(
            256, self._desc_size, kernel_size=1, stride=1, padding=0)

        # Meta descriptors aggregation
        if config['compute_meta_desc']:
            self.vlad_rot_var_illum_var = NetVLAD(
                num_clusters=self._config['n_clusters'],
                dim=self._config['meta_desc_dim']).to(device)
            self.vlad_rot_invar_illum_var = NetVLAD(
                num_clusters=self._config['n_clusters'],
                dim=self._config['meta_desc_dim']).to(device)
            self.vlad_rot_var_illum_invar = NetVLAD(
                num_clusters=self._config['n_clusters'],
                dim=self._config['meta_desc_dim']).to(device)
            self.vlad_rot_invar_illum_invar = NetVLAD(
                num_clusters=self._config['n_clusters'],
                dim=self._config['meta_desc_dim']).to(device)
            self.vlad_layers = {
                'rot_var_illum_var': self.vlad_rot_var_illum_var,
                'rot_invar_illum_var': self.vlad_rot_invar_illum_var,
                'rot_var_illum_invar': self.vlad_rot_var_illum_invar,
                'rot_invar_illum_invar': self.vlad_rot_invar_illum_invar}
    
    def forward(self, inputs, mode):
        if self._config['freeze_local_desc']:
            with torch.no_grad():
                features = self._backbone(inputs)
                outputs = self._multi_descriptors(features, mode)
        else:
            features = self._backbone(inputs)
            outputs = self._multi_descriptors(features, mode)
        if self._config['compute_meta_desc']:
            self._compute_meta_descriptors(outputs)
        return outputs

    def _multi_descriptors(self, features, mode):
        """
        Compute multiple descriptors from pre-extracted features
        with different variances / invariances.
        """
        # Illumination variant, rotation variant
        illum_var_rot_var_head = self.relu(self.conv_illum_var_rot_var_1(
            features))
        illum_var_rot_var_head = self.bn_illum_var_rot_var_1(
            illum_var_rot_var_head)
        illum_var_rot_var_head = self.conv_illum_var_rot_var_2(
            illum_var_rot_var_head)

        # Illumination variant, rotation invariant
        illum_var_rot_invar_head = self.relu(self.conv_illum_var_rot_invar_1(
            features))
        illum_var_rot_invar_head = self.bn_illum_var_rot_invar_1(
            illum_var_rot_invar_head)
        illum_var_rot_invar_head = self.conv_illum_var_rot_invar_2(
            illum_var_rot_invar_head)

        # Illumination invariant, rotation variant
        illum_invar_rot_var_head = self.relu(self.conv_illum_invar_rot_var_1(
            features))
        illum_invar_rot_var_head = self.bn_illum_invar_rot_var_1(
            illum_invar_rot_var_head)
        illum_invar_rot_var_head = self.conv_illum_invar_rot_var_2(
            illum_invar_rot_var_head)

        # Illumination invariant, rotation invariant
        illum_invar_rot_invar_head = self.relu(
            self.conv_illum_invar_rot_invar_1(features))
        illum_invar_rot_invar_head = self.bn_illum_invar_rot_invar_1(
            illum_invar_rot_invar_head)
        illum_invar_rot_invar_head = self.conv_illum_invar_rot_invar_2(
            illum_invar_rot_invar_head)

        outputs = {'raw_rot_var_illum_var': illum_var_rot_var_head,
                   'raw_rot_invar_illum_var': illum_var_rot_invar_head,
                   'raw_rot_var_illum_invar': illum_invar_rot_var_head,
                   'raw_rot_invar_illum_invar': illum_invar_rot_invar_head}
        return outputs

    def _compute_meta_descriptor(self, raw_desc, netvlad):
        tile = self._config['tile']
        # Illumination variant, rotation variant
        meta_desc = raw_desc.clone()
        b, c, h, w = meta_desc.size()
        if (h % tile != 0) or (w % tile != 0):  # should be divisible by tile
            h, w = tile * (h // tile), tile * (w // tile)
            meta_desc = func.interpolate(meta_desc, size=(h, w),
                                         mode='bilinear', align_corners=False)
        sub_h, sub_w = h // tile, w // tile
        meta_desc = meta_desc.reshape(b, c, tile, sub_h, tile, sub_w)
        meta_desc = meta_desc.permute(0, 2, 4, 1, 3, 5)
        meta_desc = meta_desc.reshape(b * tile * tile, c, sub_h, sub_w)
        meta_desc = netvlad(meta_desc).reshape(
            b, tile, tile, self._config['meta_desc_dim']
                           * self._config['n_clusters']).permute(0, 3, 1, 2)
        # meta_desc size = (b_size, meta_desc_dim * n_clusters, tile, tile)
        return meta_desc

    def _compute_meta_descriptors(self, outputs):
        """
        For each kind of descriptor, compute a meta descriptor encoding
        a sub area of the total image.
        """
        for v in self._variances:
            outputs[v + '_meta_desc'] = self._compute_meta_descriptor(
                outputs['raw_' + v], self.vlad_layers[v])


class Lisrd(BaseModel):
    required_config_keys = []

    def __init__(self, dataset, config, device):
        self._device = device
        super().__init__(dataset, config, device)
        self._variances = ['rot_var_illum_var', 'rot_invar_illum_var',
                           'rot_var_illum_invar', 'rot_invar_illum_invar']
        self._compute_meta_desc = config['compute_meta_desc']
        self._desc_weights = {'rot_var_illum_var': 1.,
                              'rot_invar_illum_var': 0.5,
                              'rot_var_illum_invar': 0.5,
                              'rot_invar_illum_invar': 0.25}

    def _model(self, config):
        return LisrdModule(config, self._device)

    def _forward(self, inputs, mode, config):
        outputs = {}
        if mode == Mode.EXPORT:
            outputs['descriptors'] = {}
            if self._compute_meta_desc:
                outputs['meta_descriptors'] = {}
            with torch.no_grad():
                desc = self._net.forward(inputs['image0'], mode)
            for v in self._variances:
                outputs['descriptors'][v] = desc['raw_' + v]
                if self._compute_meta_desc:
                    outputs['meta_descriptors'][v] = desc[v + '_meta_desc']
        else:
            desc0 = self._net.forward(inputs['image0'], mode)
            for v in self._variances:
                outputs['raw_' + v + '0'] = desc0['raw_' + v]
                if self._compute_meta_desc:
                    outputs[v + '_meta_desc0'] = desc0[v + '_meta_desc']

            if 'image1' in inputs:
                desc1 = self._net.forward(inputs['image1'], mode)
                for v in self._variances:
                    outputs['raw_' + v + '1'] = desc1['raw_' + v]
                    if self._compute_meta_desc:
                        outputs[v + '_meta_desc1'] = desc1[v + '_meta_desc']

            if 'image2' in inputs:
                desc2 = self._net.forward(inputs['image2'], mode)
                for v in self._variances:
                    outputs['raw_' + v + '2'] = desc2['raw_' + v]
                    if self._compute_meta_desc:
                        outputs[v + '_meta_desc2'] = desc2[v + '_meta_desc']

        return outputs

    def _loss(self, outputs, inputs, config):
        b_size = inputs['image0'].size()[0]
        one = torch.ones(1, dtype=torch.float, device=self._device)

        if not config['freeze_local_desc']:
            local_desc_losses = []
            invar_descs = ['raw_rot_invar_illum_invar']
            var_descs = []

            # Loss for the local descriptors
            for i in range(b_size):
                if not inputs['rot_invariant'][i]:
                    invar_descs.append('raw_rot_var_illum_invar')
                else:
                    var_descs.append(('raw_rot_var_illum_invar',
                                    inputs['rot_angle'][i]))
                if not inputs['light_invariant'][i]:
                    invar_descs.append('raw_rot_invar_illum_var')
                else:
                    var_descs.append(('raw_rot_invar_illum_var', one))
                if ((not inputs['rot_invariant'][i])
                    and (not inputs['light_invariant'][i])):
                    invar_descs.append('raw_rot_var_illum_var')
                else:
                    var_descs.append(('raw_rot_var_illum_var', one))

                # Losses for invariant descriptors
                for desc in invar_descs:
                    sub_input = {k: v[i:(i+1)] for (k, v) in inputs.items()}
                    local_desc_losses.append(invar_desc_triplet_loss(
                        outputs[desc + '0'][i:(i+1)],
                        outputs[desc + '1'][i:(i+1)],
                        sub_input, config, self._device))

                # Losses for variant descriptors
                for desc, gap in var_descs:
                    sub_input = {k: v[i:(i+1)] for (k, v) in inputs.items()}
                    local_desc_losses.append(var_desc_triplet_loss(
                        outputs[desc + '0'][i:(i+1)],
                        outputs[desc + '1'][i:(i+1)],
                        outputs[desc + '2'][i:(i+1)],
                        gap, sub_input, config, self._device))
            local_desc_loss = torch.stack(local_desc_losses, dim=0).mean()
        else:
            local_desc_loss = torch.tensor(0, dtype=torch.float,
                                           device=self._device)

        # Loss for the meta descriptors
        if self._compute_meta_desc:
            meta_desc_losses = []
            for i in range(b_size):
                sub_input = {k: v[i:(i+1)] for (k, v) in inputs.items()}
                sub_output = {k: v[i:(i+1)] for (k, v) in outputs.items()}
                meta_desc_losses.append(meta_desc_triplet_loss(
                    sub_output, sub_input, self._variances,
                    config, self._device))
            meta_desc_loss = torch.stack(meta_desc_losses, dim=0).mean()
        else:
            meta_desc_loss = torch.tensor(0, dtype=torch.float,
                                          device=self._device)

        self._writer.add_scalar('local_desc_loss', local_desc_loss, self._it)
        self._writer.add_scalar('meta_desc_loss',
                                config['lambda'] * meta_desc_loss, self._it)

        loss = local_desc_loss + config['lambda'] * meta_desc_loss
        return loss

    def _metrics(self, outputs, inputs, config):
        b_size = inputs['image0'].size()[0]
        m_scores = []

        if self._compute_meta_desc:
            for i in range(b_size):
                sub_input = {k: v[i:(i+1)] for (k, v) in inputs.items()}
                sub_output = {k: v[i:(i+1)] for (k, v) in outputs.items()}
                descs, meta_descs = [], []
                for v in self._variances:
                    descs.append([sub_output['raw_' + v + '0'],
                                  sub_output['raw_' + v + '1']])
                    meta_descs.append([sub_output[v + '_meta_desc0'],
                                       sub_output[v + '_meta_desc1']])
                m_scores.append(matching_score(sub_input, descs, meta_descs,
                                               device=self._device))
            m_score = torch.stack(m_scores, dim=0).mean()
        else:
            normalization = 0.
            for i in range(b_size):
                optim_desc = ['raw_rot_invar_illum_invar']
                if not inputs['rot_invariant'][i]:
                    optim_desc.append('raw_rot_var_illum_invar')
                if not inputs['light_invariant'][i]:
                    optim_desc.append('raw_rot_invar_illum_var')
                    if not inputs['rot_invariant'][i]:
                        optim_desc.append('raw_rot_var_illum_var')

                for desc in optim_desc:
                    sub_input = {k: v[i:(i+1)] for (k, v) in inputs.items()}
                    normalization += self._desc_weights[desc[4:]]
                    m_scores.append(
                        self._desc_weights[desc[4:]] * matching_score(
                            sub_input, [[outputs[desc + '0'][i:(i+1)],
                                         outputs[desc + '1'][i:(i+1)]]],
                            device=self._device))
            m_score = torch.sum(torch.tensor(m_scores)) / normalization

        return {'matching_score': m_score}

    def initialize_weights(self):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

        self._net.apply(init_weights)
