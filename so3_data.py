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

def read_pointxyz(cat_dir):
    cld = {}
    for dr in [i for i in os.listdir(cat_dir) if i.isdigit()]:
        points = []
        with open(os.path.join(cat_dir, dr, "points.xyz"), "r") as f:
            for i in f:
                points.append([float(j) for j in i.strip().split(" ")])
        cld[dr] = np.array(points)
    return cld

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
        boxes = meta['box']
        cmin, rmin, cmax, rmax = [int(i) for i in boxes[0]]
        cam = meta['intrinsic_matrix']
        
        target_r, _ = np.linalg.qr(meta['poses'][:, :, idx][:, :3])
        target_r = qua.as_float_array(qua.from_rotation_matrix(target_r))
        # target_r= meta['poses'][:, :, idx][:, :3]

        target_t = np.array([meta['poses'][:, :, idx][:, 3:4].flatten()])
#         target_pose =  meta['poses'][:, :, idx]
#         print(np.linalg.det(target_r))
        cam_scale = meta['factor_depth'][0][0]
        if self.transforms:
            img = self.transforms(img)
        
        return img, \
               torch.from_numpy(depth.astype(np.float32)), \
               torch.from_numpy(boxes.astype(np.float32)), \
               torch.from_numpy(label.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(target_t.astype(np.float32)), \
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

        
def display_load_img(img, depth, boxes, label, cam,  pose_t, pose_r, model_pts):
    fig,axes = plt.subplots(2, 3,  dpi= 100)
    rt_mat = np.zeros((4, 4))
    rt_mat[:3,3] = pose_t
    rt_mat[-1,-1] = 1
    rt_mat[:3,:3] = pose_r
    print(rt_mat)

    point2d = project_to_img(cam, rt_mat[:3,:], model_pts)
    img = np.transpose(img.numpy(), (1,2,0))
    axes[0][0].imshow(img)
    
    axes[0][1].imshow(depth, cmap="gray")
    
    axes[0][2].imshow(label)
    axes[1][0].imshow(img)
    axes[1][0].scatter(np.asarray(point2d[0,:]), np.asarray(point2d[1,:]), s=20, alpha=0.05)
    rect1 = patches.Rectangle((boxes[0],boxes[1]),boxes[2] - boxes[0], boxes[3] - boxes[1],linewidth=1,edgecolor='r',facecolor='none')
    axes[1][1].add_patch(rect1)
    axes[1][1].imshow(img)
    plt.show()
