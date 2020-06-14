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
import numpy as np
from torchvision import transforms
import utils.point_cloud_utils as point_cloud_utils


def angle_axis_tensor(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: torch.Tensor
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / torch.norm(axis)
    cosval, sinval = torch.cos(angle), torch.sin(angle)

    # yapf: disable
    cross_prod_mat = torch.Tensor([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]]).type(torch.FloatTensor)
    R = cosval * torch.eye(3).type(torch.FloatTensor) + sinval * cross_prod_mat + (1.0 - cosval) * torch.ger(u, u)
    return R


def angle_axis(angle, axis):
    # type: (float, np.ndarray) -> float
    r"""Returns a 4x4 rotation matrix that performs a rotation around axis by angle

    Parameters
    ----------
    angle : float
        Angle to rotate by
    axis: np.ndarray
        Axis to rotate about

    Returns
    -------
    torch.Tensor
        3x3 rotation matrix
    """
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudRandomScale(object):
    def __init__(self, lo=0.8, hi=1.25):
        self.lo, self.hi = lo, hi

    def __call__(self, points):
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points


class PointcloudRandomRotate(object):
    def __init__(self, axis=np.array([0.0, 1.0, 0.0])):
        self.axis = axis

    def __call__(self, points):
        rotation_angle = np.random.uniform() * 2 * np.pi
        axis = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
        axis = axis / np.sqrt(np.sum(axis * axis))
        rotation_matrix = angle_axis(rotation_angle, axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudRandomRotatePerturbation(object):
    def __init__(self, angle_sigma=0.06, angle_clip=0.18):
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self):
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points):
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t())
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points


class PointcloudJitter(object):
    def __init__(self, std=0.01, clip=0.05):
        self.std, self.clip = std, clip

    def __call__(self, points):
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points


class PointcloudNormalize(object):
    def __init__(self, max_size=1.0):

        self.max_size = max_size

    def __call__(self, points):

        points_max, _ = torch.max(points, dim=0)
        points_min, _ = torch.min(points, dim=0)
        points_center = (points_max + points_min) / 2
        points = points - points_center[None, :]
        max_radius = torch.max(torch.sqrt(torch.sum(points * points, dim=1)))
        points = points / max_radius * self.max_size / 2.0
        return points


class PointcloudRandomPermutation(object):
    def __call__(self, points):
        num = points.shape[0]
        idxs = torch.randperm(num).type(torch.LongTensor)
        points = torch.index_select(points, 0, idxs).clone()
        return points


class PointcloudRandomTranslate(object):
    def __init__(self, translate_range=0.1):
        self.translate_range = translate_range

    def __call__(self, points):
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points


class PointcloudToTensor(object):
    def __call__(self, points):
        return torch.from_numpy(points).float()


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.875):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points):
        pc = points.numpy()

        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point

        return torch.from_numpy(pc).float()


class PointcloudTranslate(object):
    def __init__(self, translation=np.array([0.0, 0.1, 0.0])):
        '''
        :param translation: pytorch tensor, translation vector(x,y,z)
        '''
        self.translation = torch.from_numpy(translation)


    def __call__(self, points):
        '''

        :param points: ... , num_of_points, 3
        :return: points after trans
        '''
        translation = self.translation

        if points.is_cuda is True:
            translation = translation.cuda()
            translation.requires_grad = False

        respoints = points[..., 0:3] + translation
        return respoints


class PointcloudScale(object):
    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, points):

        respoints = points * self.scaler
        return respoints


class PointcloudRotate(object):
    def __init__(self, angle_in_degree=np.pi, axis=np.array([0.0, 1.0, 0.0]), is_cuda=True):
        self.axis = axis
        self.angle_in_degree = angle_in_degree
        self.rotation_matrix_t = angle_axis(self.angle_in_degree, self.axis).t()

    def __call__(self, points):

        '''
            :param points: ... , num_of_points, 3
            :return: points after rotate
            '''
        rotation_matrix_t = self.rotation_matrix_t.clone()
        if points.is_cuda is True:
            rotation_matrix_t = rotation_matrix_t.cuda()
        tpoints = torch.matmul(points, rotation_matrix_t)

        return tpoints


def GenPointcloudRandomTransformFunction(max_rot_angle=2*np.pi):
    scale_lo = 0.8
    scale_hi = 1.25
    scaler = np.random.uniform(scale_lo, scale_hi)
    scale_func = PointcloudScale(scaler)

    rotation_angle = np.random.uniform() * max_rot_angle
    rotation_axis = np.array([np.random.uniform(), np.random.uniform(), np.random.uniform()])
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation_func = PointcloudRotate(rotation_angle, rotation_axis)

    trans_func = transforms.Compose([scale_func, rotation_func])

    return trans_func


def AddTransformsToBatchPoints(points, num_of_trans, max_rot_angle=2*np.pi):
    '''

    :param points:bn, num_of_points, 3
    :return: points: (num_of_trans, bn, num_of_points, 3)
             transform
    '''
    transfunc_list = []
    res_points = None
    for trans_i in range(0, num_of_trans):
        transf = GenPointcloudRandomTransformFunction(max_rot_angle)
        transfunc_list.append(transf)
        tpoints = transf(points)
        if res_points is None:
            res_points = tpoints[None, :, :, :]
        else:
            res_points = torch.cat((res_points, tpoints[None, :, :, :]), dim=0)

    return res_points, transfunc_list


class PointcloudRotateFuns(object):
    def __init__(self, rot_mats):
        '''
        :param rot_mats: bn, 3, 3
        '''

        self.rot_mats = rot_mats

    def __call__(self, points):
        '''

        :param points: bn, n , 3
        :return:
        '''
        if points.is_cuda is True:
            tmp_rot = self.rot_mats.cuda()
        else:
            tmp_rot = self.rot_mats
        transed_poitns = torch.transpose(torch.matmul(tmp_rot, torch.transpose(points, 1, 2)), 1, 2)
        return transed_poitns


def AddPCATransformsToBatchPoints(points, num_of_trans):
    trans_points_all = None
    rot_mats_all = None

    transfunc_list = []

    for bi in range(points.shape[0]):
        np_points = points[bi].cpu().numpy()
        pca_axis_raw = point_cloud_utils.compute_pca(np_points)
        rot_mats = None
        trans_points = None
        for ti in range(num_of_trans):
            tmp_idx = np.array([0, 1, 2])
            pca_axis = pca_axis_raw[tmp_idx, :]
            tmp_sign = np.random.randint(2, size=2)
            tmp_sign[tmp_sign == 0] = -1
            pca_axis[0, :] = pca_axis[0, :] * tmp_sign[0]
            pca_axis[1, :] = pca_axis[1, :] * tmp_sign[1]
            pca_axis[2, :] = np.cross(pca_axis[0, :], pca_axis[1, :])

            rot_mat = torch.from_numpy(pca_axis)
            if points.is_cuda:
                rot_mat = rot_mat.cuda()

            if rot_mats is None:
                rot_mats = rot_mat[None, :, :]
            else:
                rot_mats = torch.cat((rot_mats, rot_mat[None, :, :]), dim=0)

            tmp_trans_points = torch.transpose(torch.matmul(rot_mat, torch.transpose(points[bi], 0, 1)), 0, 1)

            if trans_points is None:
                trans_points = tmp_trans_points[None, :, :]
            else:
                trans_points = torch.cat((trans_points, tmp_trans_points[None, :, :]), dim=0)

        if trans_points_all is None:
            trans_points_all = trans_points[:, None, :, :]
        else:
            trans_points_all = torch.cat((trans_points_all, trans_points[:, None, :, :]), dim=1)

        if rot_mats_all is None:
            rot_mats_all = rot_mats[:, None, :, :]
        else:
            rot_mats_all = torch.cat((rot_mats_all, rot_mats[:, None, :, :]), dim=1)

    for ti in range(num_of_trans):
        trans_func = PointcloudRotateFuns(rot_mats_all[ti, :, :, :])
        transfunc_list.append(trans_func)

    return trans_points_all, rot_mats_all, transfunc_list








