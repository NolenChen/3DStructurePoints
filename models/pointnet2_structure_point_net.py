from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch
import torch.nn as nn
import pointnet2.utils.etw_pytorch_utils as pt_utils
from pointnet2.utils.pointnet2_modules import PointnetSAModuleMSG
from models import chamfer_distance


class ComputeLoss3d(nn.Module):
    def __init__(self):
        super(ComputeLoss3d, self).__init__()

        self.mse_func = nn.MSELoss()
        self.cd_loss_fun = chamfer_distance.ComputeCDLoss()
        self.loss = None
        self.consistent_loss = None
        self.cd_loss = None

    def forward(self, gt_points, structure_points, transed_gt_points=None, transed_structure_points=None, trans_func_list=None):

        gt_points = gt_points.cuda()
        structure_points = structure_points.cuda()

        batch_size = gt_points.shape[0]
        pts_num = gt_points.shape[1]
        dim = 3
        stpts_num = structure_points.shape[1]

        self.cd_loss = self.cd_loss_fun(structure_points, gt_points)

        trans_num = 0
        if transed_structure_points is not None:
            transed_structure_points = transed_structure_points.cuda()
            transed_gt_points = transed_gt_points.cuda()
            trans_num = transed_structure_points.shape[0]
            self.cd_loss = self.cd_loss + self.cd_loss_fun(transed_structure_points.view(trans_num * batch_size, stpts_num, dim),
                                                                             transed_gt_points.view(trans_num * batch_size, pts_num, dim))
            self.consistent_loss = None
            for i in range(0, trans_num):
                tmp_structure_points = trans_func_list[i](structure_points)
                tmp_structure_points = tmp_structure_points.detach()
                tmp_structure_points.requires_grad = False
                tmp_consistent_loss = self.mse_func(tmp_structure_points, transed_structure_points[i])
                if self.consistent_loss is None:
                    self.consistent_loss = tmp_consistent_loss
                else:
                    self.consistent_loss = self.consistent_loss + tmp_consistent_loss
            self.consistent_loss = self.consistent_loss / trans_num * 1000


        self.cd_loss = self.cd_loss / (trans_num + 1)

        self.loss = self.cd_loss

        if transed_structure_points is not None:
            self.loss = self.loss + self.consistent_loss
        return self.loss

    def get_cd_loss(self):
        return self.cd_loss

    def get_consistent_loss(self):
        return self.consistent_loss


class Pointnet2StructurePointNet(nn.Module):

    def __init__(self, num_structure_points, input_channels=3, use_xyz=True):
        super(Pointnet2StructurePointNet, self).__init__()
        self.point_dim = 3
        self.num_structure_points = num_structure_points
        self.input_channels = input_channels
        self.SA_modules = nn.ModuleList()
        self.stpts_prob_map = None

        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[
                    [input_channels, 32, 32, 64],
                    [input_channels, 64, 64, 128],
                    [input_channels, 64, 96, 128],
                ],
                use_xyz=use_xyz,
            )
        )

        input_channels = 64 + 128 + 128
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[
                    [input_channels, 64, 64, 128],
                    [input_channels, 128, 128, 256],
                    [input_channels, 128, 128, 256],
                ],
                use_xyz=use_xyz,
            )
        )

        conv1d_stpts_prob_modules = []
        if num_structure_points <= 128 + 256 + 256:
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=512, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(512))
            conv1d_stpts_prob_modules.append(nn.ReLU())
            in_channels = 512
            while in_channels >= self.num_structure_points * 2:
                out_channels = int(in_channels / 2)
                conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
                conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
                conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
                conv1d_stpts_prob_modules.append(nn.ReLU())
                in_channels = out_channels

            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))

            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_structure_points))
            conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))
        else:
            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=128 + 256 + 256, out_channels=1024, kernel_size=1))
            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(1024))
            conv1d_stpts_prob_modules.append(nn.ReLU())

            in_channels = 1024
            while in_channels <= self.num_structure_points / 2:
                out_channels = int(in_channels * 2)
                conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
                conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
                conv1d_stpts_prob_modules.append(nn.BatchNorm1d(out_channels))
                conv1d_stpts_prob_modules.append(nn.ReLU())
                in_channels = out_channels

            conv1d_stpts_prob_modules.append(nn.Dropout(0.2))
            conv1d_stpts_prob_modules.append(nn.Conv1d(in_channels=in_channels, out_channels=self.num_structure_points, kernel_size=1))

            conv1d_stpts_prob_modules.append(nn.BatchNorm1d(self.num_structure_points))
            conv1d_stpts_prob_modules.append(nn.Softmax(dim=2))

        self.conv1d_stpts_prob = nn.Sequential(*conv1d_stpts_prob_modules)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def forward(self, pointcloud, return_weighted_feature=False):
        '''
        :param pointcloud: input point cloud with shape (bn, num_of_pts, 3)
        :param return_weighted_feature: whether return features for the structure points or not
        :return:
        '''
        pointcloud = pointcloud.cuda()
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        self.stpts_prob_map = self.conv1d_stpts_prob(features)

        weighted_xyz = torch.sum(self.stpts_prob_map[:, :, :, None] * xyz[:, None, :, :], dim=2)
        if return_weighted_feature:
            weighted_features = torch.sum(self.stpts_prob_map[:, None, :, :] * features[:, :, None, :], dim=3)

        if return_weighted_feature:
            return weighted_xyz, weighted_features
        else:

            return weighted_xyz







