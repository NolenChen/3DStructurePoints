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
import pointnet2.utils.etw_pytorch_utils as pt_utils
import os
import argparse
import gc
from models import Pointnet2StructurePointNet
from models import ComputeLoss3d
from dataset import bhcp_dataloader
from utils.logutils import LogUtils
import utils.check_points_utils as checkpoint_util
import utils.point_cloud_utils as point_cloud_utils
from dataset import data_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def train_one_epoch(model, optimizer, data_loader, lr_scheduler, bnm_scheduler, current_iter, criterions, num_of_trans, num_inpts):
    model.train()
    ave_loss = 0
    ave_cd_loss = 0
    ave_consistent_loss = 0
    count = 0

    for batch in data_loader:
        if lr_scheduler is not None:
            lr_scheduler.step(current_iter)
        if bnm_scheduler is not None:
            bnm_scheduler.step(current_iter)
        optimizer.zero_grad()

        batch_points = batch['points']
        batch_size = batch_points.shape[0]

        if num_of_trans > 0:
            transed_batch_points, rot_mats, trans_func_list = data_utils.AddPCATransformsToBatchPoints(batch_points, num_of_trans=num_of_trans)
            transed_batch_points = transed_batch_points.view(batch_size * num_of_trans, transed_batch_points.shape[2], transed_batch_points.shape[3])
            batch_points_all = torch.cat((batch_points, transed_batch_points), dim=0)
        else:
            batch_points_all = batch_points

        batch_points_all = batch_points_all.cuda()

        if num_inpts > 0:
            batch_points_all = point_cloud_utils.farthest_pts_sampling_tensor(batch_points_all, num_inpts)

        structure_points_all = model(batch_points_all)

        structure_points_all = structure_points_all.view(num_of_trans + 1, batch_size, structure_points_all.shape[1], structure_points_all.shape[2])

        structure_points = structure_points_all[0]
        if num_of_trans > 0:
            transed_batch_points = transed_batch_points.view(num_of_trans, batch_size, transed_batch_points.shape[1], transed_batch_points.shape[2])
            transed_structure_points = structure_points_all[1:(1 + num_of_trans)]

        if num_of_trans > 0:
            loss = criterions['ComputeLoss3d'](batch_points, structure_points, transed_batch_points, transed_structure_points, trans_func_list)
        else:
            loss = criterions['ComputeLoss3d'](batch_points, structure_points, None, None, None)

        loss.backward()
        optimizer.step()
        current_iter += 1

        cd_loss = criterions['ComputeLoss3d'].get_cd_loss()
        consistent_loss = criterions['ComputeLoss3d'].get_consistent_loss()

        if consistent_loss is not None:
            print("\rbatch {0} current_loss {1}, cd_loss {2}, consistent_loss {3}".format(count, ("%.8f" % loss.item()), ("%.8f" % cd_loss.item()), ("%.8f" % consistent_loss.item())), end=" ")
        else:
            print("\rbatch {0} current_loss {1}, cd_loss {2}".format(count, ("%.8f" % loss.item()), ("%.8f" % cd_loss.item())), end=" ")

        ave_loss += loss.item()
        ave_cd_loss += cd_loss.item()

        if consistent_loss is not None:
            ave_consistent_loss += consistent_loss.item()

        gc.collect()

        count += 1
    ave_loss /= count
    ave_cd_loss /= count
    ave_consistent_loss /= count

    gc.collect()
    return ave_loss, ave_cd_loss, ave_consistent_loss, current_iter


def train(cmd_args):
    if os.path.exists(cmd_args.log_dir) is not True:
        os.makedirs(cmd_args.log_dir)

    checkpoints_dir = os.path.join(cmd_args.log_dir, "checkpoints")

    if os.path.exists(checkpoints_dir) is not True:
        os.mkdir(checkpoints_dir)

    lr_clip = 1e-5
    bnm_clip = 1e-2

    train_set = bhcp_dataloader.bhcp_dataloader(cmd_args.data_dir, cmd_args.category, is_pts_aligned=False)
    train_loader = DataLoader(
        train_set,
        batch_size=cmd_args.batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=False,
    )
    model = Pointnet2StructurePointNet(num_structure_points=cmd_args.num_structure_points, input_channels=0, use_xyz=True)
    model.cuda()
    optimizer = optim.Adam(
        model.parameters(), lr=cmd_args.lr, weight_decay=cmd_args.weight_decay
    )
    lr_lbmd = lambda it: max(
        cmd_args.lr_decay ** (int(it * cmd_args.batch_size / cmd_args.decay_step)),
        lr_clip / cmd_args.lr,
    )
    bn_lbmd = lambda it: max(
        cmd_args.bn_momentum
        * cmd_args.bnm_decay ** (int(it * cmd_args.batch_size / cmd_args.decay_step)),
        bnm_clip,
    )
    iters = -1
    criterions = {'ComputeLoss3d': ComputeLoss3d()}

    # load status from checkpoint
    log_open_mode = 'w'
    start_epoch = 0
    if cmd_args.checkpoint is not None:
        fname = os.path.join(checkpoints_dir, cmd_args.checkpoint)
        start_epoch, iters = checkpoint_util.load_checkpoint(model_3d=model, optimizer=optimizer, filename=fname)
        start_epoch += 1
        log_open_mode = 'a'

    log = LogUtils(os.path.join(cmd_args.log_dir, 'logfile'), log_open_mode)

    log.write('train unsupervised structure points for bhcp\n')
    log.write_args(cmd_args)

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=iters)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=iters
    )

    iters = max(iters, 0)
    for epoch_i in range(start_epoch, cmd_args.max_epochs):
        log.write('\nepoch: {0} ###########################'.format(epoch_i))
        train_loss, train_cd_loss, train_consistent_loss, iters = train_one_epoch(model, optimizer, train_loader, lr_scheduler, bnm_scheduler, iters, criterions, num_of_trans=cmd_args.num_of_transform, num_inpts=cmd_args.num_inpts)
        log.write('\nave_train_loss:{0}, cd_loss:{1}, consis_loss:{2}'.format(("%.8f" % train_loss), ("%.8f" % train_cd_loss), ("%.8f" % train_consistent_loss)))

        if cmd_args.checkpoint_save_step != -1 and (epoch_i + 1) % cmd_args.checkpoint_save_step is 0:
            fname = os.path.join(checkpoints_dir, 'checkpoint_{}'.format(epoch_i))
            checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters, epoch=epoch_i)

        fname = os.path.join(checkpoints_dir, 'model')
        checkpoint_util.save_checkpoint(filename=fname, model_3d=model, optimizer=optimizer, iters=iters, epoch=epoch_i)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-weight_decay", type=float, default=1e-5, help="L2 regularization coeff"
    )
    parser.add_argument("-lr", type=float, default=1e-2, help="Initial learning rate")
    parser.add_argument(
        "-lr_decay", type=float, default=0.7, help="Learning rate decay gamma"
    )
    parser.add_argument(
        "-decay_step", type=float, default=10000, help="Learning rate decay step"
    )
    parser.add_argument(
        "-bn_momentum", type=float, default=0.5, help="Initial batch norm momentum"
    )
    parser.add_argument(
        "-bnm_decay", type=float, default=0.5, help="Batch norm momentum decay gamma"
    )

    parser.add_argument(
        "-checkpoint_save_step", type=int, default=10, help="Step for saving Checkpoint"
    )
    parser.add_argument("-batch_size", type=int, default=4, help="Batch size")
    parser.add_argument(
        "-checkpoint", type=str, default=None
        , help="Checkpoint to start from"
    )
    parser.add_argument(
        "-num_of_transform", type=int, default=0, help="Number of transforms for rotation data augmentation. Useful when testing on shapes without alignment"
    )

    parser.add_argument(
        "-num_inpts", type=int, default=2048, help="sample points from initial point cloud"
    )

    parser.add_argument(
        "-num_structure_points", type=int, default=512
        , help="Number of structure points"
    )
    parser.add_argument(
        "-category", type=str, default='chair', help="Category of the objects to train"
    )

    parser.add_argument(
        "-data_dir", type=str, default="", help="Root of the training data"
    )
    parser.add_argument(
        "-max_epochs", type=int, default=200, help="Number of epochs to train for"
    )
    parser.add_argument(
        "-log_dir", type=str, default="", help="Root of the log"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    train(cmd_args=args)




