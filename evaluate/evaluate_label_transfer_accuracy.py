import sys
sys.path.append("..")
import torch
from torch.utils.data import DataLoader
import os
import gc
import ntpath
from models import Pointnet2StructurePointNet
from dataset import shapenet_seg_dataloader
import utils.check_points_utils as checkpoint_util
import utils.point_cloud_utils as point_cloud_utils
import numpy as np
import random
import pickle
import argparse


def create_color_list(num):
    colors = np.ndarray(shape=(num, 3))
    for i in range(0, num):
        colors[i, 0] = random.randint(0, 255)
        colors[i, 1] = random.randint(0, 255)
        colors[i, 2] = random.randint(0, 255)
    colors[0, :] = np.array([0, 0, 255]).astype(np.int)
    colors[1, :] = np.array([0, 255, 255]).astype(np.int)
    colors[2, :] = np.array([255, 0, 0]).astype(np.int)
    colors[3, :] = np.array([0, 255, 0]).astype(np.int)
    colors[4, :] = np.array([0, 100, 50]).astype(np.int)
    colors[5, :] = np.array([255, 255, 0]).astype(np.int)
    colors[6, :] = np.array([255, 100, 50]).astype(np.int)
    return colors


COLOR_LIST = create_color_list(50)


def compute_shape_mean_iou(res_seg_data):
#

    mean_iou = 0
    unique_labels = None
    for i in range(len(res_seg_data)):
        seg_res = res_seg_data[i]
        gt_pts_labels = seg_res['gt_pts_labels']
        tmp_unique_labels = np.unique(gt_pts_labels)
        if unique_labels is None:
            unique_labels = tmp_unique_labels
        else:
            unique_labels = np.unique(np.concatenate((unique_labels, tmp_unique_labels), axis=0))

    for i in range(len(res_seg_data)):

        seg_res = res_seg_data[i]
        res_pts_labels = seg_res['res_pts_labels']
        gt_pts_labels = seg_res['gt_pts_labels']
        iou_a_dict = {} 
        iou_b_dict = {}

        for pi in range(res_pts_labels.shape[0]):

            if res_pts_labels[pi] not in iou_a_dict.keys():
                iou_a_dict[res_pts_labels[pi]] = 0
                iou_b_dict[res_pts_labels[pi]] = 0

            if gt_pts_labels[pi] not in iou_a_dict.keys():
                iou_a_dict[gt_pts_labels[pi]] = 0
                iou_b_dict[gt_pts_labels[pi]] = 0

            if res_pts_labels[pi] == gt_pts_labels[pi]:
                iou_a_dict[res_pts_labels[pi]] += 1
                iou_b_dict[res_pts_labels[pi]] += 1
            else:
                iou_b_dict[res_pts_labels[pi]] += 1
                iou_b_dict[gt_pts_labels[pi]] += 1

        tmp_mean_iou = 0

        for label in unique_labels:
            if label not in iou_a_dict.keys():# part is not present, no prediction as well (same way as BAENet)
                iou_a_dict[label] = 1
                iou_b_dict[label] = 1

        for key in iou_a_dict.keys():
            tmp_mean_iou += iou_a_dict[key] / iou_b_dict[key]
        tmp_mean_iou = tmp_mean_iou / len(iou_a_dict)
        mean_iou += tmp_mean_iou
    mean_iou /= len(res_seg_data)
    return mean_iou


def label_transfer(src_pts, src_labels, tgt_pts, nn_num):
    ids_cand, _ = point_cloud_utils.query_KNN(src_pts, tgt_pts, nn_num)
    labels_cand = src_labels[ids_cand.reshape(tgt_pts.shape[0] * nn_num)]
    labels_cand = labels_cand.reshape(tgt_pts.shape[0], nn_num)

    res_tgt_pts_labels = []
    for pi in range(labels_cand.shape[0]):
        tmp_labels, ext_times = np.unique(labels_cand[pi], return_counts=True)
        max_idx = np.argmax(ext_times)
        max_label = tmp_labels[max_idx]
        res_tgt_pts_labels.append(max_label)

    res_tgt_pts_labels = np.array(res_tgt_pts_labels)
    return res_tgt_pts_labels


def load_ref_data(ref_data_dir, ref_shape_names=None):
    if ref_shape_names is None:
        subdirs = [ntpath.basename(f.path) for f in os.scandir(ref_data_dir) if f.is_dir()]
        ref_num = len(subdirs)
    else:
        subdirs = ref_shape_names
        ref_num = len(ref_shape_names)

    ref_data = []
    for i in range(ref_num):
        subdir = subdirs[i]
        print(subdir)
        meta_fname = os.path.join(ref_data_dir, subdir, 'meta.pkl')
        f = open(meta_fname, 'rb')
        meta_data = pickle.load(f)
        ref_data.append(meta_data)

    return ref_data


def test_label_transfer(model_fname, num_structure_points, ref_data_dir, test_data_dir, category, out_data_dir=None, ref_shape_names=None, save_tmp_data=True):
    ref_data = load_ref_data(os.path.join(ref_data_dir, category), ref_shape_names=ref_shape_names)
    test_set = shapenet_seg_dataloader.ShapenetSegDataloader(data_root=test_data_dir, category=category, train=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
    model = Pointnet2StructurePointNet(num_structure_points=num_structure_points, input_channels=0, use_xyz=True)
    model.cuda()

    checkpoint_util.load_checkpoint(model_3d=model, filename=model_fname)

    if out_data_dir is not None:
        out_data_dir = os.path.join(out_data_dir, category)
        if os.path.exists(out_data_dir) is False:
            os.makedirs(out_data_dir)

    model.eval()
    nn_num_for_label = 10
    nn_num_for_feat = 1

    if out_data_dir is not None and os.path.exists(out_data_dir) is not True:
        os.mkdir(out_data_dir)

    ref_features = []
    ref_st_pts_labels = []
    count = 0
    for ref_meta in ref_data:
        points = ref_meta['points']
        gt_points = ref_meta['gt_points']
        gt_points_labels = ref_meta['gt_points_labels'] - 1

        points = point_cloud_utils.farthest_pts_sampling_tensor(torch.from_numpy(points[None, :, :]).cuda(), 2048)
        st_points, features = model(points, return_weighted_feature=True)
        st_points = st_points.squeeze().cpu().detach().numpy()
        features = features.squeeze().cpu().detach().numpy().transpose(1, 0)
        st_pts_labels = label_transfer(gt_points, gt_points_labels, st_points, nn_num=nn_num_for_label)
        ref_features.append(features)
        ref_st_pts_labels.append(st_pts_labels)
        count = count + 1

        
        if out_data_dir is not None:
            gt_pts_colors = COLOR_LIST[gt_points_labels]
            fname_inpts = os.path.join(out_data_dir, 'ref{0}.off'.format(count))
            point_cloud_utils.write_points_off(fname_inpts, gt_points, gt_pts_colors)

    gc.collect()

    ref_features = np.array(ref_features)
    ref_st_pts_labels = np.array(ref_st_pts_labels)
    count = 0
    seg_res = []
    for batch in test_loader:

        batch_points = batch['points'].cuda()
        batch_gt_points = batch['gt_points'].cpu().detach().numpy()
        batch_gt_points_labels = batch['gt_points_labels'].cpu().detach().numpy()

        batch_points = point_cloud_utils.farthest_pts_sampling_tensor(batch_points, 2048)
        st_points, features = model(batch_points, return_weighted_feature=True)
        batch_features = features.cpu().detach().numpy()
        batch_st_points = st_points.cpu().detach().numpy()
        batch_points = batch_points.cpu().detach().numpy()

        for i in range(0, batch_points.shape[0]):
            print('\r{0}'.format(count), end=" ")
            features = batch_features[i].transpose(1, 0)
            st_points = batch_st_points[i]
            gt_points = batch_gt_points[i]
            st_labels = []
            for fi in range(features.shape[0]):
                tmp_label = label_transfer(ref_features[:, fi, :], ref_st_pts_labels[:, fi], features[fi, None, :], nn_num=nn_num_for_feat)
                st_labels.append(tmp_label[0])

            st_labels = np.array(st_labels)

            res_gt_pts_labels = label_transfer(st_points, st_labels, gt_points, nn_num=nn_num_for_label)

            gt_pts_labels = batch_gt_points_labels[i]
            gt_pts_colors = COLOR_LIST[gt_pts_labels]
            tmp_seg_res = {'gt_pts_labels': gt_pts_labels, 'res_pts_labels': res_gt_pts_labels}
            seg_res.append(tmp_seg_res)

            if out_data_dir is not None and save_tmp_data:
                pts_colors = COLOR_LIST[res_gt_pts_labels]
                fname_inpts = os.path.join(out_data_dir, 'batch{0}_segres_pts.off'.format(count, i))
                point_cloud_utils.write_points_off(fname_inpts, batch_gt_points[i], pts_colors)

                fname_inpts = os.path.join(out_data_dir, 'batch{0}_gt_pts.off'.format(count, i))
                point_cloud_utils.write_points_off(fname_inpts, batch_gt_points[i], gt_pts_colors)

        count = count + 1

    shape_mean_iou = compute_shape_mean_iou(seg_res)


    print('shape mean iou: %.2f' % (shape_mean_iou*100))

    return shape_mean_iou


def read_ref_model_id(fname):
    f = open(fname)
    ref_str = f.readline()
    ref_str = ref_str.split(' ')
    return ref_str


def test(args):

    num_structure_points = args.num_structure_points
    category = args.category
    model_fname = args.pretrained
    test_data_dir = os.path.join(args.data_dir, 'test')
    ref_data_dir = os.path.join(args.data_dir, 'train')
    output_dir = args.output_dir
    ref_keys = read_ref_model_id(args.ref_fname)

    test_label_transfer(model_fname, num_structure_points, ref_data_dir, test_data_dir, category, output_dir, ref_shape_names=ref_keys)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-pretrained", type=str, default='label_transfer_pretrained/Chair_1024_model', help="pretrained model path"
    )
    parser.add_argument(
        "-num_structure_points", type=int, default=1024, help="Number of structure points"
    )
    parser.add_argument(
        "-category", type=str, default='Chair', help="Category of the objects to train"
    )
    parser.add_argument(
        "-ref_fname", type=str, default='label_transfer_ref_model_id/Chair.txt', help='path to data set'
    )
    parser.add_argument(
        "-data_dir", type=str, default="datasets/shapenet_for_seg/", help="Root of the training data"
    )
    parser.add_argument(
        "-output_dir", type=str, default=None, help=""
    )

    return parser.parse_args()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    test(args)
















