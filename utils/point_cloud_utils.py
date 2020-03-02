import numpy as np
import torch
import cv2
from pointnet2 import _ext

def write_points_off(fname, points, colors=None):

    with open(fname, 'w') as f:

        num = points.shape[0]
        f.write('COFF\n')
        f.write('{0} 0 0\n'.format(num))
        for i in range(0, num):
            if colors is not None:
                f.write('{0} {1} {2} {3} {4} {5}\n'.format(points[i, 0], points[i, 1], points[i, 2], int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
            else:
                f.write('{0} {1} {2}\n'.format(points[i, 0], points[i, 1], points[i, 2]))


def write_points_obj(fname, points, colors=None):

    with open(fname, 'w') as f:

        num = points.shape[0]
        for i in range(0, num):
            if colors is not None:
                f.write('v {0} {1} {2} {3} {4} {5}\n'.format(points[i, 0], points[i, 1], points[i, 2], int(colors[i, 0]), int(colors[i, 1]), int(colors[i, 2])))
            else:
                f.write('v {0} {1} {2}\n'.format(points[i, 0], points[i, 1], points[i, 2]))


def compute_pca(points):
    mean, eigvec = cv2.PCACompute(points, mean=None)
    if np.dot(np.cross(eigvec[0], eigvec[1]), eigvec[2])<0:
        eigvec[2] = -eigvec[2]

    eigvec[0] = eigvec[0] / np.linalg.norm(eigvec[0])
    eigvec[1] = eigvec[1] / np.linalg.norm(eigvec[1])
    eigvec[2] = eigvec[2] / np.linalg.norm(eigvec[2])

    return eigvec

def query_KNN(points, query_pts, k, return_dis=True):
    '''

    :param points: n x 3
    :param query_pts: m x 3
    :param k: num of neighbors
    :return: m x k  ids, sorted_dis
    '''

    diff = query_pts[:, None, :] - points[None, :, :]
    dis = np.sqrt(np.sum(diff * diff, axis=2))# m x n
    sorted_idx = np.argsort(dis, axis=1)
    sorted_idx = sorted_idx[:, :k]

    if return_dis:
        sorted_dis = dis[None, 0, sorted_idx[0, :]]
        for i in range(1, query_pts.shape[0]):
            sorted_dis = np.concatenate((sorted_dis, dis[None, i, sorted_idx[i, :]]), axis=0)

        return sorted_idx, sorted_dis
    else:
        return sorted_idx


def query_KNN_tensor(points, query_pts, k):
    '''

    :param points: n x 3
    :param query_pts: m x 3
    :param k: num of neighbors
    :return: m x k  ids, sorted_dis
    '''

    diff = query_pts[:, None, :] - points[None, :, :]
    dis = torch.sqrt(torch.sum(diff * diff, dim=2))# m x n
    sorted_idx = torch.argsort(dis, dim=1)
    sorted_idx = sorted_idx[:, :k]

    sorted_dis = dis[None, 0, sorted_idx[0, :]]
    for i in range(1, query_pts.shape[0]):
        sorted_dis = torch.cat((sorted_dis, dis[None, i, sorted_idx[i, :]]), dim=0)

    return sorted_idx, sorted_dis



def read_pointcloud_obj(fname):
    vertices = []
    try:
        f = open(fname)

        for line in f:
            if line[:2] == "v ":
                strs = line.split(' ')
                v0 = float(strs[1])
                v1 = float(strs[2])
                v2 = float(strs[3])
                vertex = [v0, v1, v2]
                vertices.append(vertex)

        f.close()
    except IOError:
        print(".obj file not found.")

    vertices = np.array(vertices)


    return vertices


def read_points_off(fname, read_color=False):
    vertices = []
    colors = []

    try:
        f = open(fname)
        head = f.readline()
        strline = f.readline()
        strs = strline.split(' ')
        vnum = int(strs[0])
        fnum = int(strs[1])
        for i in range(0, vnum):
            strline = f.readline()
            strs = strline.split(' ')
            v0 = float(strs[0])
            v1 = float(strs[1])
            v2 = float(strs[2])
            vertex = [v0, v1, v2]
            vertices.append(vertex)

            if len(strs) > 3:
                c0 = float(strs[3])
                c1 = float(strs[4])
                c2 = float(strs[5])
                color = [c0, c1, c2]
                colors.append(color)




        f.close()
    except IOError:
        print(".off file not found.")

    pts = np.array(vertices).astype(np.float32)

    if len(colors) > 0 and read_color == True:
        colors = np.array(colors).astype(np.float32)
        return pts, colors
    else:
        return pts

def trans_pointcloud(rot_mat, trans_mat, points):
    '''

    :param rot_mat: 3 x 3
    :param trans_mat: 3
    :param points: n x 3
    :return: n x 3
    '''
    tmp_points = np.matmul(rot_mat, np.transpose(points, (1, 0)))
    tmp_points = tmp_points + trans_mat[:, None]
    tmp_points = np.transpose(tmp_points, (1, 0))
    return tmp_points




def farthest_pts_sampling_tensor(pts, num_samples, return_sampled_idx=False):
    '''

    :param pts: bn, n, 3
    :param num_samples:
    :return:
    '''
    sampled_pts_idx = _ext.furthest_point_sampling(pts, num_samples)
    sampled_pts_idx_viewed = sampled_pts_idx.view(sampled_pts_idx.shape[0]*sampled_pts_idx.shape[1]).cuda().type(torch.LongTensor)
    batch_idxs = torch.tensor(range(pts.shape[0])).type(torch.LongTensor)
    batch_idxs_viewed = batch_idxs[:, None].repeat(1, sampled_pts_idx.shape[1]).view(batch_idxs.shape[0]*sampled_pts_idx.shape[1])
    sampled_pts = pts[batch_idxs_viewed, sampled_pts_idx_viewed, :]
    sampled_pts = sampled_pts.view(pts.shape[0], num_samples, 3)

    if return_sampled_idx == False:
        return sampled_pts
    else:
        return sampled_pts, sampled_pts_idx























