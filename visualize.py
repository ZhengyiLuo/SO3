import torchvision
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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
from so3_data import *
import cv2
from scipy.spatial.transform import Rotation
import matplotlib

GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def proj_pts_display(pts, cam, pose, img_org):

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout(pad=0)
    ax1.axis('square')
    pts2d = project_to_img(cam, pose, pts)
    ax1.scatter(pts2d[0,:], pts2d[1,:], c = (pts[:,2] * 100).astype(int))
    if np.max(img_org) > 1:
        img_org = img_org/256

    ax2.imshow(img_org)
    ax1.set_xlim(0, 224)
    ax1.set_ylim(0, 224)
    ax1.invert_yaxis()
    # plt.show()
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image_from_plot

def write_vid(vid_name, frames):
    out = cv2.VideoWriter(vid_name, cv2.VideoWriter_fourcc(*'FMP4'), 1, (640,480))
    for i in range(len(frames)):
        img = frames[i]
        out.write(img)
    out.release()


def visualize_model(rot_repr, model_dir, loader, points, vid_name = "res"):
    matplotlib.use("pdf")
    print("========================================================================")
    print("visualizing model....")
    from train import Net_Rot
    model = Net_Rot(rot_repr)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.npy")))
    model.to(GLOBAL_DEVICE)

    counter = 0
    frames = []
    for i, d in enumerate(loader, 0):
        img, depth, boxes2d, boxes2d_8, boxes3d, label, pose_r, pose_t, pose, cam,idx = d
        img, pose = img.to(GLOBAL_DEVICE), pose.to(GLOBAL_DEVICE)
        
        res = model(img)
        mats = rotReprToRotMat(res, rot_repr)

        for i in range(mats.shape[0]):
            mat = mats[i].detach().cpu().numpy()
            mat_4 = np.eye(4)
            mat_4[:3,:3] = mat
            
            mat_4[:3,3] = pose[i][:,3].detach().cpu().numpy()
            # print(mat, pose[i])
            img_org = np.transpose(img[i].detach().cpu().numpy(), (1,2,0))
            image_from_plot = proj_pts_display(points, cam[i], mat_4[:3,:4], img_org)
            frames.append(image_from_plot)
            counter += 1
            # print(image_from_plot.shape)
            # plt.imshow(image_from_plot)
            # plt.show()

        if counter > 50:
            break

    frames = np.array(frames)
    # np.save("frames_{}.npy".format(rot_repr), frames)
    write_vid(os.path.join(model_dir, "{}.mp4".format(vid_name)), frames)




if __name__ == "__main__":
    rot_repr = "quat"
    # dataset_root = "/hdd/zen/dev/6dof/6dof_data/so3/test/car_ycb/"
    dataset_root = "/hdd/zen/dev/6dof/6dof_data/so3/small/car_ycb/"
    # dataset_root = "/home/qiaog/courses/16720B-project/SO3/data/car_ycb"
    transform=transforms.Compose([transforms.ToTensor()])
    train_dataset = PoseDataset('train', dataset_root, transforms=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=1)

    test_dataset = PoseDataset('test', dataset_root, transforms=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)

    cld = read_pointxyz( os.path.join(dataset_root, "models"))    
    points = cld['0000']
    model_dir = "output/quat_small_matrot"
    visualize_model(rot_repr, model_dir, test_loader, points, vid_name = "test")
    visualize_model(rot_repr, model_dir, train_loader, points, vid_name = "train")
    