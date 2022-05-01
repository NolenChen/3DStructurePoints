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
import etw_pytorch_utils as pt_utils
import os
import argparse
import gc
import random
from models import Pointnet2StructurePointNet
from models import ComputeLoss3d
from dataset import shapenet_seg_dataloader
from utils.logutils import LogUtils
import utils.check_points_utils as checkpoint_util
import utils.point_cloud_utils as point_cloud_utils

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(10)

def train_one_epoch(model, optimizer, data_loader, lr_scheduler, bnm_scheduler, current_iter, criterions, num_inpts):
    model.train()
    ave_loss = 0
    ave_cd_loss = 0
    count = 0

    for batch in data_loader:
        if lr_scheduler is not None:
            lr_scheduler.step(current_iter)
        if bnm_scheduler is not None:
            bnm_scheduler.step(current_iter)
        optimizer.zero_grad()

        batch_points = batch['points'].cuda()

        if num_inpts > 0:
            batch_points = point_cloud_utils.farthest_pts_sampling_tensor(batch_points, num_inpts)

        st_points = model(batch_points)

        loss = criterions['ComputeLoss3d'](batch_points, st_points, None, None, None)

        loss.backward()
        optimizer.step()
        current_iter += 1

        cd_loss = criterions['ComputeLoss3d'].get_cd_loss()

        print("\rbatch {0} current_loss {1}, cd_loss {2}".format(count, ("%.8f" % loss.item()), ("%.8f" % cd_loss.item())), end=" ")

        ave_loss += loss.item()
        ave_cd_loss += cd_loss.item()

        gc.collect()

        count += 1
    ave_loss /= count
    ave_cd_loss /= count

    gc.collect()
    return ave_loss, ave_cd_loss, current_iter


def train(cmd_args):
    if os.path.exists(cmd_args.log_dir) is not True:
        os.makedirs(cmd_args.log_dir)

    checkpoints_dir = os.path.join(cmd_args.log_dir, "checkpoints")

    if os.path.exists(checkpoints_dir) is not True:
        os.mkdir(checkpoints_dir)

    lr_clip = 1e-5
    bnm_clip = 1e-2

    train_set = shapenet_seg_dataloader.ShapenetSegDataloader(cmd_args.data_dir, cmd_args.category)

    train_loader = DataLoader(
        train_set,
        batch_size=cmd_args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
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


    log_open_mode = 'w'
    start_epoch = 0

    log = LogUtils(os.path.join(cmd_args.log_dir, 'logfile'), log_open_mode)

    log.write('train unsupervised structure points for label transfer\n')
    log.write_args(cmd_args)

    lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lbmd, last_epoch=iters)
    bnm_scheduler = pt_utils.BNMomentumScheduler(
        model, bn_lambda=bn_lbmd, last_epoch=iters
    )

    iters = max(iters, 0)
    for epoch_i in range(start_epoch, cmd_args.max_epochs):
        log.write('\nepoch: {0} ###########################'.format(epoch_i))
        train_loss, train_cd_loss, iters = train_one_epoch(model, optimizer, train_loader, lr_scheduler, bnm_scheduler, iters, criterions, num_inpts=cmd_args.num_inpts)
        log.write('\nave_train_loss:{0}, cd_loss:{1}'.format(("%.8f" % train_loss), ("%.8f" % train_cd_loss)))


        fname = os.path.join(checkpoints_dir, 'model')
        checkpoint_util.save_checkpoint(filename=fname, model_3d=model)

def gen_name_from_args(cmd_args):

    name = 'label_transfer_{0}'.format(cmd_args.category)
    return name
    
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
    parser.add_argument("-batch_size", type=int, default=4, help="Batch size")

    parser.add_argument(
        "-num_inpts", type=int, default=2048, help="sample points from initial point cloud"
    )
    parser.add_argument(
        "-num_structure_points", type=int, default=1024, help="Number of structure points"
    )
    parser.add_argument(
        "-category", type=str, default='Chair', help="Category of the objects to train"
    )
    parser.add_argument(
        "-data_dir", type=str, default="datasets/shapenet_for_seg/train", help="Root of the training data"
    )
    parser.add_argument(
        "-max_epochs", type=int, default=2000, help="Number of max epochs. For each category we train the network until convergence. See logs for the actual epochs trained"
    )
    
    log_name = gen_name_from_args(parser.parse_args())
    parser.add_argument(
        "-log_dir", type=str, default="logs_/{0}".format(log_name), help="Root of the log"
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    train(cmd_args=args)