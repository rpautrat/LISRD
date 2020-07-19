import torch
import torch.nn.functional as func

from ..utils.geometry_utils import get_lisrd_desc_dist
from ..utils.pytorch_utils import keypoints_to_grid


def matching_score(inputs, descs, meta_descs=None, device='cuda:0'):
    b_size, _, Hc, Wc = descs[0][0].size()
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

    # Keep only the valid points
    keypoints0 = keypoints0.reshape(b_size * n_points, 2)[valid_mask]
    keypoints1 = keypoints1.reshape(b_size * n_points, 2)[valid_mask]

    # Convert the keypoints to a grid suitable for interpolation
    grid0 = keypoints_to_grid(keypoints0, img_size)
    grid1 = keypoints_to_grid(keypoints1, img_size)

    # Distances of reprojected points of image0 with the original points
    if meta_descs:
        descs_n, meta_descs_n = [], []
        for i in range(len(descs)):
            desc0 = func.grid_sample(descs[i][0], grid0).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
            desc0 = func.normalize(desc0, dim=1)
            desc1 = func.grid_sample(descs[i][1], grid1).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
            desc1 = func.normalize(desc1, dim=1)
            descs_n.append([desc0, desc1])

            meta_desc0 = func.grid_sample(meta_descs[i][0], grid0).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
            meta_desc0 = func.normalize(meta_desc0, dim=1)
            meta_desc1 = func.grid_sample(meta_descs[i][1], grid1).permute(
                0, 2, 3, 1).reshape(n_correct_points, -1)
            meta_desc1 = func.normalize(meta_desc1, dim=1)
            meta_descs_n.append([meta_desc0, meta_desc1])
        desc_dist = get_lisrd_desc_dist(descs_n, meta_descs_n)
    else:
        desc0 = func.grid_sample(descs[0][0], grid0).permute(
            0, 2, 3, 1).reshape(n_correct_points, -1)
        desc0 = func.normalize(desc0, dim=1)
        desc1 = func.grid_sample(descs[0][1], grid1).permute(
            0, 2, 3, 1).reshape(n_correct_points, -1)
        desc1 = func.normalize(desc1, dim=1)
        desc_dist = 2 - 2 * (desc0 @ desc1.t())

    # Compute percentage of correct matches
    matches0 = torch.min(desc_dist, dim=1)[1]
    matches1 = torch.min(desc_dist, dim=0)[1]
    matching_score = (matches1[matches0]
                      == torch.arange(len(matches0)).to(device))
    return matching_score.float().mean()
