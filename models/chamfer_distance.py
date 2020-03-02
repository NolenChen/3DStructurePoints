from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn

def query_KNN_tensor(points, query_pts, k):
    '''

       :param points: bn x n x 3
       :param query_pts: bn x m x 3
       :param k: num of neighbors
       :return: nb x m x k  ids, sorted_squared_dis
       '''

    diff = query_pts[:, :, None, :] - points[:, None, :, :]

    squared_dis = torch.sum(diff*diff, dim=3)  # bn x m x n
    sorted_squared_dis, sorted_idxs = torch.sort(squared_dis, dim=2)
    sorted_idxs = sorted_idxs[:, :, :k]
    sorted_squared_dis = sorted_squared_dis[:, :, :k]

    return sorted_idxs, sorted_squared_dis

def compute_chamfer_distance(p1, p2):
    '''
    Calculate Chamfer Distance between two point sets
    :param p1: size[bn, N, D]
    :param p2: size[bn, M, D]
    :param debug: whether need to output debug info
    :return: sum of Chamfer Distance of two point sets
    '''




    diff = p1[:, :, None, :] - p2[:, None, :, :]
    dist = torch.sum(diff*diff,  dim=3)
    dist1 = dist
    dist2 = torch.transpose(dist, 1, 2)

    dist_min1, _ = torch.min(dist1, dim=2)
    dist_min2, _ = torch.min(dist2, dim=2)

    return dist_min1, dist_min2

class ComputeCDLoss(nn.Module):
    def __init__(self):
        super(ComputeCDLoss, self).__init__()

    def forward(self, recon_points, gt_points):

        dist1, dist2 = compute_chamfer_distance(recon_points, gt_points)

        loss = (torch.sum(dist1) + torch.sum(dist2)) / (recon_points.shape[0])
        return loss

