from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import torch.utils.data as data
import os
import ntpath
import pickle
import re


def natural_sort_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def sort_strs_by_number(strs):
    return sorted(strs, key=natural_sort_key)


class ShapenetSegDataloader(data.Dataset):
    def __init__(self, data_root, category, preload_data=True, train=True):
        super().__init__()
        self.category = category
        self.data_root = data_root
        self.data_path = os.path.join(data_root, category)
        self.sub_dirs = [ntpath.basename(f.path) for f in os.scandir(self.data_path) if f.is_dir()]
        #self.sub_dirs = sort_strs_by_number([ntpath.basename(f.path) for f in os.scandir(self.data_path) if f.is_dir()])
        self.data_num = len(self.sub_dirs)
        self.train = train

        self.meta_data_list = None
        if preload_data:
            self.meta_data_list = []
            for i in range(self.data_num):
                meta_fname = os.path.join(self.data_path, self.sub_dirs[i], 'meta.pkl')
                with open(meta_fname, 'rb') as f:
                    meta_data = pickle.load(f)
                    self.meta_data_list.append(meta_data)

    def __getitem__(self, idx):
        if self.meta_data_list is None:
            meta_fname = os.path.join(self.data_path, self.sub_dirs[idx], 'meta.pkl')
            f = open(meta_fname, 'rb')
            meta_data = pickle.load(f)
        else:
            meta_data = self.meta_data_list[idx]

        points = meta_data['points']
        gt_points = meta_data['gt_points']
        gt_points_labels = meta_data['gt_points_labels'] - 1
        meta_data = {'points': points}

        if self.train is not True:
            meta_data['gt_points'] = gt_points
            meta_data['gt_points_labels'] = gt_points_labels
        return meta_data

    def __len__(self):
        return self.data_num