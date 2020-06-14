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

class bhcp_dataloader(data.Dataset):
    def __init__(self, data_root, category, is_pts_aligned=False, preload_data=True):
        super().__init__()
        self.data_path = os.path.join(data_root, category)
        self.sub_dirs = [ntpath.basename(f.path) for f in os.scandir(self.data_path) if f.is_dir()]
        self.data_num = len(self.sub_dirs)
        self.is_pts_aligned = is_pts_aligned

        self.meta_data_list = None
        if preload_data:
            self.meta_data_list = []
            for i in range(len(self.sub_dirs)):
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

        if self.is_pts_aligned:
            if 'points_aligned' in meta_data:
                points = meta_data['points_aligned']
            else:
                points = meta_data['points']
        else:
            points = meta_data['points']

        res = {}
        res['points'] = points
        if 'feat_pts' in meta_data: # the labeled feature points on the bhcp dataset for computing correspondence accuracy
            if self.is_pts_aligned:
                res['feat_pts'] = meta_data['feat_pts_aligned']
            else:
                res['feat_pts'] = meta_data['feat_pts']

        res['data_id'] = idx

        return res

    def __len__(self):
        return self.data_num






