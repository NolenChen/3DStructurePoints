from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import sys
sys.path.append("..")
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from torchvision import transforms
import etw_pytorch_utils as pt_utils
import os
import argparse
import gc

from models import Pointnet2LandmarkNet
from models import ComputeLoss3d
from dataset import bhcp_dataloader
import dataset.data_utils as d_utils
from utils.logutils import LogUtils
import utils.check_points_utils as checkpoint_util
import utils.point_cloud_utils as point_cloud_utils
import numpy as np
import random
from utils.logutils import LogUtils
from dataset import data_utils
from eval import draw_eval
from eval import raw_data_for_bhcp_cmp
import pickle
def read_feat_file(fname):
    f = open(fname)
    feat_pts = []

    count = 0
    for line in f:
        str = line.split()
        if len(str) == 7:
            x = float(str[4])
            y = float(str[5])
            z = float(str[6])
            feat_pts.append([x, y, z])
            count = count + 1
        elif str[0] == '-1':
            feat_pts.append([np.nan, np.nan, np.nan])
            count = count + 1
    f.close()
    return np.array(feat_pts).astype(np.float32)


def create_color_list(num):
    colors = np.ndarray(shape=(num, 3))
    random.seed(30)
    for i in range(0, num):
        colors[i, 0] = random.randint(0, 255)
        colors[i, 1] = random.randint(0, 255)
        colors[i, 2] = random.randint(0, 255)

    colors[0, :] = np.array([0, 0, 0]).astype(np.int)
    colors[1, :] = np.array([146, 61, 10]).astype(np.int)
    colors[2, :] = np.array([102, 97, 0]).astype(np.int)
    colors[3, :] = np.array([255, 0, 0]).astype(np.int)
    colors[4, :] = np.array([113, 0, 17]).astype(np.int)
    colors[5, :] = np.array([255, 127, 39]).astype(np.int)
    colors[6, :] = np.array([255, 242, 0]).astype(np.int)
    colors[7, :] = np.array([0, 255, 0]).astype(np.int)
    colors[8, :] = np.array([0, 0, 255]).astype(np.int)
    colors[9, :] = np.array([15, 77, 33]).astype(np.int)
    colors[10, :] = np.array([163, 73, 164]).astype(np.int)
    colors[11, :] = np.array([255, 174, 201]).astype(np.int)
    colors[12, :] = np.array([255, 220, 14]).astype(np.int)
    colors[13, :] = np.array([181, 230, 29]).astype(np.int)
    colors[14, :] = np.array([153, 217, 234]).astype(np.int)
    colors[15, :] = np.array([112, 146, 190]).astype(np.int)

    return colors


COLOR_LIST = create_color_list(5000)


def compute_correspondence_dis_two_pair_tensor(model_data_a, model_data_b):
    feat_pts_a = model_data_a['feat_pts'].cpu()
    gt_feat_pts_a = model_data_a['gt_feat_pts'].cpu()
    feat_pts_b = model_data_b['feat_pts'].cpu()
    gt_feat_pts_b = model_data_b['gt_feat_pts'].cpu()





    knn_a_idxs, knn_a_dis = point_cloud_utils.query_KNN_tensor(feat_pts_a, gt_feat_pts_a, 1)
    nei_pts_in_b = feat_pts_b[knn_a_idxs[:, 0], :]
    diff = nei_pts_in_b - gt_feat_pts_b
    tmp_dis = torch.sqrt(torch.sum(diff * diff, dim=1))

    res_dis = []
    for i in range(tmp_dis.shape[0]):
        if torch.isnan(gt_feat_pts_a[i, 0]) == False and  torch.isnan(gt_feat_pts_b[i, 0]) == False:
            res_dis.append(tmp_dis[i])

    return res_dis



def compute_correspondence_accuracy(model_datas):

    dis_list = []
    for i in range(len(model_datas)):
        for j in range(len(model_datas)):
            if i == j:
                continue

            model_data_i = model_datas[i]
            model_data_j = model_datas[j]
            corres_dis = compute_correspondence_dis_two_pair_tensor(model_data_i, model_data_j)
            dis_list = dis_list + corres_dis

    dis_array = np.array(dis_list)

    dis_thresholds = np.array(range(0, 26, 1)) / 100.0
    dis_ratios = []

    for i in range(dis_thresholds.shape[0]):
        threshold = dis_thresholds[i]
        ratio = dis_array[dis_array <= threshold].shape[0] / dis_array.shape[0]

        dis_ratios.append(ratio)

    dis_ratios = np.array(dis_ratios)
    return dis_ratios, dis_thresholds



def bhcp_validate_one_epoch(category, model, data_loader, out_data_dir=None, save_tmp_data=False, use_pca=False, ref_meta_fname=None, subsample_num=-1):

    print('\nvalidate_one_epoch\n')


    model.eval()

    count = 0

    model_datas = []
    if os.path.exists(out_data_dir) is not True:
        os.mkdir(out_data_dir)

    ref_feature_points = None
    if ref_meta_fname is not None and use_pca:
        file = open(ref_meta_fname, 'rb')
        meta_data = pickle.load(file)
        ref_points = meta_data['points']
        ref_points = torch.from_numpy(ref_points)
        ref_feature_points, _ = model(ref_points[None, :, :])
        ref_feature_points = ref_feature_points.detach().cpu().numpy()



    for batch in data_loader:

        batch_points = batch['points']

        batch_gt_feat_pts = batch['feat_pts']
        batch_data_id = batch['data_id']

        '''
        batch_points, transfunc_list = d_utils.AddTransformsToBatchPoints(batch_points, 1)
        batch_gt_feat_pts = transfunc_list[0](batch_gt_feat_pts)
        batch_points = batch_points[0]
        '''

        if use_pca:
            batch_points, rot_mats, _ = d_utils.AddPCATransformsToBatchPoints(batch_points, num_of_trans=1)
            batch_points = batch_points.squeeze(dim=0)
            rot_mats = rot_mats.squeeze(dim=0)
            for bi in range(batch_gt_feat_pts.shape[0]):
                batch_gt_feat_pts[bi] = torch.transpose(torch.matmul(rot_mats[bi], torch.transpose(batch_gt_feat_pts[bi], 0, 1)), 0, 1)


            if ref_meta_fname is not None and False:
                tmp_feature_points, _ = model(batch_points)

                for bi in range(0, batch_points.shape[0]):
                    tmp_rot_mat, _ = point_cloud_utils.estimate_rigid_transformation(tmp_feature_points[bi].detach().cpu().numpy(), ref_feature_points[0])
                    tmp_rot_mat = torch.from_numpy(tmp_rot_mat)
                    batch_gt_feat_pts[bi] = torch.transpose(torch.matmul(tmp_rot_mat, torch.transpose(batch_gt_feat_pts[bi], 0, 1)), 0, 1)
                    batch_points[bi] = torch.transpose(torch.matmul(tmp_rot_mat, torch.transpose(batch_points[bi], 0, 1)), 0, 1)

        batch_points = batch_points.cuda()
        batch_gt_feat_pts = batch_gt_feat_pts.cuda()
        if subsample_num > 0:
            batch_points = point_cloud_utils.farthest_pts_sampling_tensor(batch_points, subsample_num)



        feature_points, features_all = model(batch_points)


        for i in range(0, batch_points.shape[0]):

            tmp_data = {}
            tmp_data['feat_pts'] = feature_points[i].cpu().detach()
            tmp_data['gt_feat_pts'] = batch_gt_feat_pts[i].cpu().detach()

            tmp_data['points'] = batch_points[i].cpu().detach()
            tmp_data['idx'] = batch_data_id[i].cpu().detach()
            model_datas.append(tmp_data)

            if out_data_dir is not None and save_tmp_data:
                tmp_fpts = feature_points[i, :, :].cpu().detach().numpy()
                tmp_fpts_color = COLOR_LIST[:tmp_fpts.shape[0], :]

                tmp_input_pts = batch_points[i, :, :].cpu().detach().numpy()

                fname_inpts = os.path.join(out_data_dir, 'data{0}_inpts.off'.format(batch_data_id[i]))
                point_cloud_utils.write_points_off(fname_inpts, tmp_input_pts, None)

                fname_feat = os.path.join(out_data_dir, 'data{0}_feat.off'.format(batch_data_id[i]))
                point_cloud_utils.write_points_off(fname_feat, tmp_fpts, tmp_fpts_color)


        gc.collect()

        count = count + 1


    dis_ratios, dis_thresholds = compute_correspondence_accuracy(model_datas)
    fname = os.path.join(out_data_dir, 'results.txt')
    LogUtils.write_correspondence_accuracy(fname, dis_ratios, dis_thresholds)

    linechart_fname = os.path.join(out_data_dir, 'eval.png')
    batch_dis_ratio, batch_dis_threshold, batch_lagend = raw_data_for_bhcp_cmp.get_raw_data_for_cmp(category)
    batch_dis_threshold.append(dis_thresholds)
    batch_dis_ratio.append(dis_ratios*100)
    legend = 'ours'
    batch_lagend.append(legend)
    draw_eval.save_linechart(linechart_fname, category, batch_dis_ratio, batch_dis_threshold, batch_lagend)

    return dis_ratios, dis_thresholds



def main(model_fname, num_feat_points, data_dir, category, output_dir, is_pts_aligned=False, ref_meta_fname=None, subsample_num=-1):


    valid_set = bhcp_dataloader.bhcp_dataloader(data_dir, category, transforms=None, is_pts_aligned=is_pts_aligned)
    valid_loader = DataLoader(
        valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )

    model = Pointnet2LandmarkNet(num_feature_points=num_feat_points, input_channels=0, use_xyz=True)
    model.cuda()
    checkpoint_util.load_checkpoint(model_3d=model, filename=model_fname)

    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    if is_pts_aligned == False:
        output_dir = os.path.join(output_dir, category)
    else:
        output_dir = os.path.join(output_dir, category + '_aligned')

    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    batch_dis_ratio, batch_dis_threshold, batch_lagend = raw_data_for_bhcp_cmp.get_raw_data_for_cmp(category)
    if is_pts_aligned is True:
        use_pca = False
    else:
        use_pca = True
    dis_ratios, dis_thresholds = bhcp_validate_one_epoch(category, model, valid_loader, output_dir, save_tmp_data=True, use_pca=use_pca, ref_meta_fname=ref_meta_fname, subsample_num=subsample_num)
    batch_dis_threshold.append(dis_thresholds)
    batch_dis_ratio.append(dis_ratios*100)
    legend = 'ours'
    if is_pts_aligned:
        legend = 'ours_aligned'
    batch_lagend.append(legend)
    draw_eval.draw_linechart(category, batch_dis_ratio, batch_dis_threshold, batch_lagend)
    linechart_fname = os.path.join(output_dir, 'eval.png')
    draw_eval.save_linechart(linechart_fname, category, batch_dis_ratio, batch_dis_threshold, batch_lagend)




if __name__ == "__main__":

    checkpoint = 'best_checkpoint'
    train_category = 'plane'
    test_category = 'helicopter'
    feat_num = 512
    trans_num = 3
    subsample_num = 2048
    model_fname = 'E:/ImagePointCloudMatching/PointCloudLandmark/logs/bhcp_{0}_feat{1}_trans{2}/checkpoints/{3}'.format(train_category, feat_num, trans_num, checkpoint)
    data_dir = 'H:/ImagePointCloudMatchingDataset/bhcp/validation/'


    output_dir = 'E:/ImagePointCloudMatching/PointCloudLandmark/logs/bhcp_{0}_feat{1}_trans{2}/validation/'.format(train_category, feat_num, trans_num)
    ref_meta_fname = 'H:/ImagePointCloudMatchingDataset/shapenet_for_bhcp/{0}/0/meta.pkl'.format(train_category)
    if train_category != test_category:
        ref_meta_fname = None
    main(model_fname, feat_num, data_dir, test_category, output_dir, is_pts_aligned=False, ref_meta_fname=ref_meta_fname, subsample_num=subsample_num)













