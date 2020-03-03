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
import os
import argparse
from models import Pointnet2StructurePointNet
import utils.check_points_utils as checkpoint_util
import utils.point_cloud_utils as point_cloud_utils
import numpy as np
import random
import glob

def create_color_list(num):
    colors = np.ndarray(shape=(num, 3))
    random.seed(30)
    for i in range(0, num):
        colors[i, 0] = random.randint(0, 255)
        colors[i, 1] = random.randint(0, 255)
        colors[i, 2] = random.randint(0, 255)
    return colors
COLOR_LIST = create_color_list(5000)


def main(args):

    model = Pointnet2StructurePointNet(num_structure_points=args.num_structure_points, input_channels=0, use_xyz=True)
    model.cuda()
    checkpoint_util.load_checkpoint(model_3d=model, filename=args.model_fname)

    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    fnames = glob.glob(os.path.join(args.data_dir, '*.off'))

    for fname in fnames:
     
        fname = os.path.basename(fname)
        pts = point_cloud_utils.read_points_off(os.path.join(args.data_dir,fname))

        batch_pts = torch.from_numpy(pts)[None, :, :].cuda()
        structure_points = model(batch_pts)

        structure_points = structure_points[0].cpu().detach().numpy()
        outfname = os.path.join(args.output_dir, fname[:-4] + '_stpts.off')
        point_cloud_utils.write_points_off(outfname, structure_points, COLOR_LIST[:structure_points.shape[0], :])

    print('output saved to {0}'.format(args.output_dir))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Arguments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-model_fname", type=str, default='', help="path to the trained model"
    )
    parser.add_argument(
        "-num_structure_points", type=int, default=16, help="number of structure points"
    )
    parser.add_argument(
        "-output_dir", type=str, default='', help="output dir"
    )
    parser.add_argument(
        "-data_dir", type=str, default='', help="path to testing data"
    )
    args = parser.parse_args()
    main(args)













