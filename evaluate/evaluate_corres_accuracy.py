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
from torch.utils.data import DataLoader
import argparse
import gc
from models import Pointnet2StructurePointNet
from dataset import bhcp_dataloader
import utils.check_points_utils as checkpoint_util
import utils.point_cloud_utils as point_cloud_utils
import numpy as np
import dataset.data_utils as d_utils


def compute_correspondence_dis(model_data_a, model_data_b):
    structure_pts_a = model_data_a['structure_pts'].cpu()
    gt_feat_pts_a = model_data_a['gt_feat_pts'].cpu()
    structure_pts_b = model_data_b['structure_pts'].cpu()
    gt_feat_pts_b = model_data_b['gt_feat_pts'].cpu()

    knn_a_idxs, knn_a_dis = point_cloud_utils.query_KNN_tensor(structure_pts_a, gt_feat_pts_a, 1)
    corres_pts_in_b = structure_pts_b[knn_a_idxs[:, 0], :]
    diff = corres_pts_in_b - gt_feat_pts_b
    tmp_dis = torch.sqrt(torch.sum(diff * diff, dim=1))

    res_dis = []
    for i in range(tmp_dis.shape[0]):
        #nan means this feature point is missing on groundtruth model
        if torch.isnan(gt_feat_pts_a[i, 0]) == False and torch.isnan(gt_feat_pts_b[i, 0]) == False:
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
            corres_dis = compute_correspondence_dis(model_data_i, model_data_j)
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


def bhcp_evaluate_one_epoch(model, data_loader, test_on_aligned, subsample_num=2048):

    model.eval()
    count = 0
    model_datas = []

    for batch in data_loader:
        batch_points = batch['points']
        batch_gt_feat_pts = batch['feat_pts']

        batch_points = batch_points.cuda()
        batch_gt_feat_pts = batch_gt_feat_pts.cuda()
        if subsample_num > 0 and subsample_num != batch_points.shape[1]:
            batch_points = point_cloud_utils.farthest_pts_sampling_tensor(batch_points, subsample_num)
        if test_on_aligned is not True:
            batch_points, rot_mats, _ = d_utils.AddPCATransformsToBatchPoints(batch_points, num_of_trans=1)
            batch_points = batch_points.squeeze(dim=0)
            rot_mats = rot_mats.squeeze(dim=0)

            for bi in range(batch_gt_feat_pts.shape[0]):
                batch_gt_feat_pts[bi] = torch.transpose(torch.matmul(rot_mats[bi], torch.transpose(batch_gt_feat_pts[bi], 0, 1)), 0, 1)

        structure_points = model(batch_points)

        for i in range(0, batch_points.shape[0]):
            tmp_data = {'structure_pts': structure_points[i].cpu().detach(), 'gt_feat_pts': batch_gt_feat_pts[i].cpu().detach()}
            model_datas.append(tmp_data)

        gc.collect()
        count = count + 1

    dis_ratios, dis_thresholds = compute_correspondence_accuracy(model_datas)
    return dis_ratios, dis_thresholds


def main(args):
    test_set = bhcp_dataloader.bhcp_dataloader(args.data_dir, args.category, is_pts_aligned=args.test_on_aligned)
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
    )

    model = Pointnet2StructurePointNet(num_structure_points=args.num_structure_points, input_channels=0, use_xyz=True)
    model.cuda()
    checkpoint_util.load_checkpoint(model_3d=model, filename=args.model_fname)
    dis_ratios, dis_thresholds = bhcp_evaluate_one_epoch(model, test_loader, args.test_on_aligned)

    print('distance threshold|radio:')
    for i in range(dis_ratios.shape[0]):
        print(' {0}|{1}'.format(dis_thresholds[i], ("%.6f" % dis_ratios[i])), end='')
    print('\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-model_fname", type=str, default='', help="path to the trained model"
    )
    parser.add_argument(
        "-category", type=str, default='chair', help="category to test"
    )
    parser.add_argument(
        "-num_structure_points", type=int, default=512, help="number of structure points, should be same with the trained network"
    )
    parser.add_argument(
        "-data_dir", type=str, default='', help="path to testing data"
    )
    parser.add_argument(
        "-test_on_aligned", type=str, default='True', help="whether the testing shape is aligned or not. If set to False, the network should be trained with num_of_transform > 0 to use PCA data aug"
    )
    args = parser.parse_args()
    if args.test_on_aligned.lower() == 'true':
        args.test_on_aligned = True
    else:
        args.test_on_aligned = False
    main(args)













