import torch


def keypoints_to_grid(keypoints, img_size):
    """
    Convert a tensor [N, 2] or batched tensor [B, N, 2] of N keypoints into
    a grid in [-1, 1]² that can be used in torch.nn.functional.interpolate.
    """
    n_points = keypoints.size()[-2]
    device = keypoints.device
    grid_points = keypoints.float() * 2. / torch.tensor(
        img_size, dtype=torch.float, device=device) - 1.
    grid_points = grid_points[..., [1, 0]].view(-1, n_points, 1, 2)
    return grid_points