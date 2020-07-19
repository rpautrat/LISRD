import torch
import torch.nn as nn
import torch.nn.functional as func


class NetVLAD(nn.Module):
    """
    NetVLAD layer implementation
    Credits: https://github.com/lyakaap/NetVLAD-pytorch
    """

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):
        """
        Args:
            num_clusters: number of clusters.
            dim: dimension of descriptors.
            alpha: parameter of initialization. Larger is harder assignment.
            normalize_input: if true, descriptor-wise L2 normalization
                             is applied to input.
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters,
                              kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1))
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1))

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = func.normalize(x, p=2, dim=1)  # across descriptor dim

        # Soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = func.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)
        
        # Calculate residuals to each clusters
        residual = (x_flatten.expand(
            self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3)
            - self.centroids.expand(
                x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0))
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = func.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = func.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad
