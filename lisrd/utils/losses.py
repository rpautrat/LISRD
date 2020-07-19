import numpy as np
import torch
import torch.nn.functional as func

from ..utils.geometry_utils import get_lisrd_desc_dist
from ..utils.pytorch_utils import keypoints_to_grid


def get_pos_dist(desc_dists):
    return torch.diag(desc_dists)


def get_dist_mask(kp0, kp1, valid_mask, dist_thresh):
    """ Return a 2D matrix indicating the local neighborhood of each point
        for a given threshold and two lists of corresponding keypoints. """
    b_size, n_points, _ = kp0.size()
    dist_mask0 = torch.norm(kp0.unsqueeze(2) - kp0.unsqueeze(1), dim=-1)
    dist_mask1 = torch.norm(kp1.unsqueeze(2) - kp1.unsqueeze(1), dim=-1)
    dist_mask = torch.min(dist_mask0, dist_mask1)
    dist_mask = dist_mask <= dist_thresh
    dist_mask = dist_mask.repeat(1, 1, b_size).reshape(b_size * n_points,
                                                       b_size * n_points)
    dist_mask = dist_mask[valid_mask, :][:, valid_mask]
    return dist_mask


def get_neg_dist(desc_dists, dist_mask, device):
    n_correct_points = len(desc_dists)
    max_dist = torch.tensor(4., dtype=torch.float, device=device)
    desc_dists[torch.arange(n_correct_points, dtype=torch.long),
               torch.arange(n_correct_points,
                            dtype=torch.long)] = max_dist
    desc_dists[dist_mask] = max_dist
    neg_dist = torch.min(torch.min(desc_dists, dim=1)[0],
                         torch.min(desc_dists, dim=0)[0])
    return neg_dist


def invar_desc_triplet_loss(desc0, desc1, inputs, config, device):
    b_size, _, Hc, Wc = desc0.size()
    img_size = (Hc * 8, Wc * 8)

    valid_mask = inputs['valid_mask']
    n_points = valid_mask.size()[1]
    n_correct_points = torch.sum(valid_mask.int()).item()
    if n_correct_points == 0:
        return torch.tensor(0., dtype=torch.float, device=device)
    if torch.__version__ >= '1.2':
        valid_mask = valid_mask.bool().flatten()
    else:
        valid_mask = valid_mask.byte().flatten()
    keypoints0 = inputs['keypoints0']
    keypoints1 = inputs['keypoints1']

    # Compute the distances between the keypoints of image1
    dist_mask = get_dist_mask(keypoints0, keypoints1, valid_mask,
                              config['dist_threshold'])

    # Keep only the valid points
    keypoints0 = keypoints0.reshape(b_size * n_points, 2)[valid_mask]
    keypoints1 = keypoints1.reshape(b_size * n_points, 2)[valid_mask]

    # Convert the keypoints to a grid suitable for interpolation
    grid0 = keypoints_to_grid(keypoints0, img_size)
    grid1 = keypoints_to_grid(keypoints1, img_size)
    
    # Extract the descriptors
    valid_desc0 = func.grid_sample(desc0, grid0).permute(
        0, 2, 3, 1).reshape(n_correct_points, -1)
    valid_desc0 = func.normalize(valid_desc0, dim=1)
    valid_desc1 = func.grid_sample(desc1, grid1).permute(
        0, 2, 3, 1).reshape(n_correct_points, -1)
    valid_desc1 = func.normalize(valid_desc1, dim=1)
    desc_dist = 2 - 2 * (valid_desc0 @ valid_desc1.t())

    # Positive loss
    pos_dist = get_pos_dist(desc_dist)

    # Negative loss
    neg_dist = get_neg_dist(desc_dist, dist_mask, device)

    return torch.mean(func.relu(config['margin'] + pos_dist - neg_dist))


def var_desc_triplet_loss(desc0, desc1, desc2, gap, inputs, config, device):
    Hc, Wc = desc0.size()[2:4]
    img_size = (Hc * 8, Wc * 8)

    valid_mask = inputs['valid_mask']
    n_correct_points = torch.sum(valid_mask.int()).item()
    if n_correct_points == 0:
        return torch.tensor(0., dtype=torch.float, device=device)
    if torch.__version__ >= '1.2':
        valid_mask = valid_mask.bool()
    else:
        valid_mask = valid_mask.byte()
    keypoints0 = inputs['keypoints0'][valid_mask]
    keypoints1 = inputs['keypoints1'][valid_mask]
    keypoints2 = inputs['keypoints2'][valid_mask]

    # Convert the keypoints to a grid suitable for interpolation
    grid0 = keypoints_to_grid(keypoints0, img_size)
    grid1 = keypoints_to_grid(keypoints1, img_size)
    grid2 = keypoints_to_grid(keypoints2, img_size)
    
    # Extract the descriptors
    valid_desc0 = func.grid_sample(desc0, grid0).permute(
        0, 2, 3, 1).reshape(n_correct_points, -1)
    valid_desc0 = func.normalize(valid_desc0, dim=1)
    valid_desc1 = func.grid_sample(desc1, grid1).permute(
        0, 2, 3, 1).reshape(n_correct_points, -1)
    valid_desc1 = func.normalize(valid_desc1, dim=1)
    valid_desc2 = func.grid_sample(desc2, grid2).permute(
        0, 2, 3, 1).reshape(n_correct_points, -1)
    valid_desc2 = func.normalize(valid_desc2, dim=1)

    # Negative distance
    desc_dist = 2 - 2 * (valid_desc0 @ valid_desc1.t())
    neg_dist = torch.diag(desc_dist)

    # Positive distance
    desc_dist = 2 - 2 * (valid_desc0 @ valid_desc2.t())
    pos_dist = torch.diag(desc_dist)

    return torch.mean(func.relu(gap * config['margin'] + pos_dist - neg_dist))


def meta_desc_triplet_loss(outputs, inputs, variances, config, device):
    b_size, _, Hc, Wc = outputs['raw_rot_var_illum_var0'].size()
    img_size = (Hc * 8, Wc * 8)

    valid_mask = inputs['valid_mask']
    n_points = valid_mask.size()[1]
    n_correct_points = torch.sum(valid_mask.int()).item()
    if n_correct_points == 0:
        return torch.tensor(0., dtype=torch.float, device=device)
    if torch.__version__ >= '1.2':
        valid_mask = valid_mask.bool().flatten()
    else:
        valid_mask = valid_mask.byte().flatten()
    keypoints0 = inputs['keypoints0']
    keypoints1 = inputs['keypoints1']

    # Compute the distances between the keypoints of image1
    dist_mask = get_dist_mask(keypoints0, keypoints1, valid_mask,
                              config['dist_threshold'])

    # Keep only the valid points
    keypoints0 = keypoints0.reshape(b_size * n_points, 2)[valid_mask]
    keypoints1 = keypoints1.reshape(b_size * n_points, 2)[valid_mask]

    # Convert the keypoints to a grid suitable for interpolation
    grid0 = keypoints_to_grid(keypoints0, img_size)
    grid1 = keypoints_to_grid(keypoints1, img_size)

    # Retrieve all local descriptors
    descs = []
    for v in variances:
        valid_desc0 = func.grid_sample(
            outputs['raw_' + v + '0'], grid0).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
        valid_desc0 = func.normalize(valid_desc0, dim=1)
        valid_desc1 = func.grid_sample(
            outputs['raw_' + v + '1'], grid1).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
        valid_desc1 = func.normalize(valid_desc1, dim=1)
        descs.append([valid_desc0, valid_desc1])

    # Retrieve all meta descriptors
    meta_descs = []
    for v in variances:
        valid_meta_desc0 = func.grid_sample(
            outputs[v + '_meta_desc0'], grid0).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
        valid_meta_desc0 = func.normalize(valid_meta_desc0, dim=1)
        valid_meta_desc1 = func.grid_sample(
            outputs[v + '_meta_desc1'], grid1).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
        valid_meta_desc1 = func.normalize(valid_meta_desc1, dim=1)
        meta_descs.append([valid_meta_desc0, valid_meta_desc1])

    # Get the weighted descriptor distances
    desc_dist = get_lisrd_desc_dist(descs, meta_descs)

    # Positive loss
    pos_dist = get_pos_dist(desc_dist)

    # Negative loss
    neg_dist = get_neg_dist(desc_dist, dist_mask, device)
    
    return torch.mean(func.relu(config['margin'] + pos_dist - neg_dist))
