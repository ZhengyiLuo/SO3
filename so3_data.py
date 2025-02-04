import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio
from lib.proj_utils import *
from lib.transformations import translation_matrix, quaternion_matrix, quaternion_from_matrix
import quaternion as qua

input_size = 128


class PoseDataset(data.Dataset):
    def __init__(self, mode, root, transforms=None):
        self.transforms = transforms
        self.root = root
        if mode == 'train':
            self.path = os.path.join(root,'dataset_config/train_data_list.txt')
        elif mode == 'test':
            self.path = os.path.join(root,'dataset_config/test_data_list.txt')

        self.list = []
        with open(self.path) as f:
            for input_line in f:
                self.list.append(input_line.strip())

        self.length = len(self.list)
        class_file = open(os.path.join(root,'dataset_config/classes.txt'))

        self.models = read_pointxyz(os.path.join(root,'models'))
        print(len(self.list))

    def __getitem__(self, index):

        try:
            img = np.array(Image.open('{0}/{1}-color.png'.format(self.root, self.list[index])))[:,:,:3]
            depth = np.load(('{0}/{1}-depth.npy'.format(self.root, self.list[index])))
            label = np.array(Image.open('{0}/{1}-label.png'.format(self.root, self.list[index])))
            meta = scio.loadmat('{0}/{1}-meta.mat'.format(self.root, self.list[index]))
        except Exception as e:
            print(e)
            print(self.list[index])
        obj = meta['cls_indexes'].flatten().astype(np.int32)
        model_id = meta['model_id'][0]
        model_points = self.models[model_id]
        idx = np.random.randint(0, len(obj))

        box2d = meta['box2d']
        box3d = meta['box3d']
        box2d_proj = meta['box2d_proj']

        cam = meta['intrinsic_matrix']

        pose = meta['poses'][:, :, idx]
        pose[:, 2] *= -1
        target_r = qua.as_float_array(qua.from_rotation_matrix(pose))

        # target_r= meta['poses'][:, :, idx][:, :3]

        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
#         target_pose =  meta['poses'][:, :, idx]
#         print(np.linalg.det(target_r))
        cam_scale = meta['factor_depth'][0][0]
        if self.transforms:
            img = self.transforms(img)

        # Project the 3D box coordinates into 2D image
        # print(boxes3d.shape)
        # print(boxes2d.shape)

        return img, \
               torch.from_numpy(depth.astype(np.float32)), \
               torch.from_numpy(box2d.astype(np.float32)), \
               torch.from_numpy(box2d_proj.astype(np.float32)), \
               torch.from_numpy(box3d.astype(np.float32)), \
               torch.from_numpy(label.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(pose.astype(np.float32)), \
               torch.from_numpy(cam.astype(np.float32)), \
               model_id

    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small
