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
import argparse
from models import Pointnet2StructurePointNet
import utils.check_points_utils as checkpoint_util
import utils.point_cloud_utils as point_cloud_utils
import dataset.data_utils as d_utils


def compute_correspondence(structure_pts_src, query_pts_src, structure_pts_tgt):
    structure_pts_src = structure_pts_src.cpu()
    query_pts_src = query_pts_src.cpu()
    structure_pts_tgt = structure_pts_tgt.cpu()

    knn_src_idxs, knn_src_dis = point_cloud_utils.query_KNN_tensor(structure_pts_src, query_pts_src, 1)
    corres_pts_in_tgt = structure_pts_tgt[knn_src_idxs[:, 0], :]

    return corres_pts_in_tgt


def main(args):
    model = Pointnet2StructurePointNet(num_structure_points=args.num_structure_points, input_channels=0, use_xyz=True)
    model.cuda()
    checkpoint_util.load_checkpoint(model_3d=model, filename=args.model_fname)

    src_shape_fname = args.src_shape_fname
    query_pts_fname = args.query_pts_fname
    tgt_shape_fname = args.tgt_shape_fname
    out_corres_pts_fname = args.out_corres_pts_fname

    src_pts = point_cloud_utils.read_points_off(src_shape_fname)
    tgt_pts = point_cloud_utils.read_points_off(tgt_shape_fname)
    query_pts, query_pts_colors = point_cloud_utils.read_points_off(query_pts_fname, read_color=True)

    batch_src_pts = torch.from_numpy(src_pts)[None, :, :].cuda()
    batch_tgt_pts = torch.from_numpy(tgt_pts)[None, :, :].cuda()
    query_pts = torch.from_numpy(query_pts)[:, :]

    if args.test_on_aligned is False:
        batch_src_pts, src_rot_mats, _ = d_utils.AddPCATransformsToBatchPoints(batch_src_pts, num_of_trans=1)
        batch_src_pts = batch_src_pts.squeeze(dim=0)
        src_rot_mat = src_rot_mats.squeeze(dim=0)[0]
        batch_tgt_pts, tgt_rot_mats, _ = d_utils.AddPCATransformsToBatchPoints(batch_tgt_pts, num_of_trans=1)
        batch_tgt_pts = batch_tgt_pts.squeeze(dim=0)
        tgt_rot_mat = tgt_rot_mats.squeeze(dim=0)[0]

    structure_points_src = model(batch_src_pts)[0]
    structure_points_tgt = model(batch_tgt_pts)[0]
    structure_points_src = torch.mm(src_rot_mat.transpose(0, 1), structure_points_src.transpose(0, 1)).transpose(0, 1)
    structure_points_tgt = torch.mm(tgt_rot_mat.transpose(0, 1), structure_points_tgt.transpose(0, 1)).transpose(0, 1)

    corres_pts_in_tgt = compute_correspondence(structure_points_src, query_pts, structure_points_tgt)
    point_cloud_utils.write_points_off(out_corres_pts_fname, corres_pts_in_tgt, query_pts_colors)

    print('output saved to {0}'.format(out_corres_pts_fname))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-model_fname", type=str, default='', help="path to the trained model"
    )
    parser.add_argument(
        "-num_structure_points", type=int, default=512, help="number of structure points"
    )
    parser.add_argument(
        "-src_shape_fname", type=str, default='', help="path to the source shape"
    )
    parser.add_argument(
        "-query_pts_fname", type=str, default='', help="path to the file containing the query points"
    )
    parser.add_argument(
        "-tgt_shape_fname", type=str, default='', help="path to the target shape"
    )
    parser.add_argument(
        "-out_corres_pts_fname", type=str, default='', help="path to the output correspondence on the target shape"
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













