# %%writefile proj_utils.py
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.patches as patches
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import pywavefront
from PIL import Image
import os
import scipy.io as sio
from collections import OrderedDict
from lib.transformations import translation_matrix, quaternion_matrix, quaternion_from_matrix

camear_matrix = np.array([[1465.8411,    0.0000, 64.0000],[0.0000, 1465.8411, 64.0000],[0.0000,    0.0000,  1.0000]])
def get_meta(meta_file):
    trans = np.matrix([[ 1.,  1., -1.,  1.],
       [-1., -1.,  1., -1.],
       [-1., -1.,  1., -1.]])
    lines = []
    for i in meta_file:
        lines.append(i.strip())
    box = np.matrix([[float(j) for j in i.split(" ")] for i in lines[1:9]])
    rt = np.matrix([[float(j) for j in i.split(" ")] for i in lines[10:13]])
    sclae = float(lines[14])
    return box, np.multiply(trans, rt), sclae

def project_to_img(camear_matrix, rt, points, scale = 1):
    point3d = np.matrix.transpose(np.hstack((np.matrix(points), np.ones(len(points)).reshape(-1, 1))))
    point3d[:3,:] *= scale
    P = np.matmul(camear_matrix, rt)
    point3d = np.matmul(P, point3d)
    point3d[0:2,:] = np.divide(point3d[0:2,:], point3d[2,:], )
    point2d = point3d[:2,:]
    return point2d

def box_to_box(box2d):
    x_max = np.max(box2d[0,:])
    x_min = np.min(box2d[0,:])
    y_max = np.max(box2d[1,:])
    y_min = np.min(box2d[1,:])
    return x_min, y_min, x_max, y_max

def display_blender_annotation(camear_matrix, rt, points, box, img, scale = 1):
    fig,ax = plt.subplots(1)

    point3d = project_to_img(camear_matrix, rt, points, scale)
    ax.scatter(np.asarray(point3d[0 ,:]), np.asarray(point3d[1,:]), alpha=0.05)
    box2d = project_to_img(camear_matrix, rt, box, scale)
    ax.scatter(np.asarray(box2d[0 ,:]), np.asarray(box2d[1,:]), alpha=0.8)

    label = box_to_box(box2d)
    rect1 = patches.Rectangle((label[0],label[1]),label[2] - label[0], label[3] - label[1],linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect1)

    ax.imshow(img)
    plt.show()

def diplay_ycb(base_dir, models, index, scale = 1):
    value = loadmat(os.path.join(base_dir, "%06d-meta.mat" % index))
    box1 = open(os.path.join(base_dir,  "%06d-box.txt"% index), "r")
    img_color = plt.imread(os.path.join(base_dir, "%06d-color.png"% index))
    depth_img = plt.imread(os.path.join(base_dir, "%06d-depth.png"% index))
    semantic = plt.imread(os.path.join(base_dir, "%06d-label.png"% index))

    fig,axes = plt.subplots(2, 3,  dpi= 150)
    boxes = []
    for i in box1:
        boxes.append([float(j) for j in i.split()[1:]])
    axes[0][0].imshow(img_color)
    img = cv2.normalize(depth_img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    axes[0][1].imshow(img, cmap="gray")
    axes[0][2].imshow(semantic)

    for i in range(len(value["cls_indexes"])):
        index = value["cls_indexes"][i][0] - 1
        point2d = project_to_img(value["intrinsic_matrix"], value["poses"][:,:,i], np.array(models[index].vertices), scale)
        axes[1][0].imshow(img_color)
        axes[1][0].scatter(np.asarray(point2d[0,:]), np.asarray(point2d[1,:]))


    axes[1][1].imshow(img_color)
    for i in range(len(value["cls_indexes"])):
        label = boxes[i]
        rect1 = patches.Rectangle((label[0],label[1]),label[2] - label[0], label[3] - label[1],linewidth=1,edgecolor='r',facecolor='none')
        axes[1][1].add_patch(rect1)

        axes[1][1].scatter(value["center"][i][0], value["center"][i][1], s=50)
    x = value
    plt.show()
    return value, boxes, img_color, depth_img, semantic

def read_pointxyz(cat_dir):
    cld = {}
    for dr in [i for i in os.listdir(cat_dir) if i.isdigit()]:
        points = []
        with open(os.path.join(cat_dir, dr, "points.xyz"), "r") as f:
            for i in f:
                points.append([float(j) for j in i.strip().split(" ")])

        points = np.array(points)
        # points[:,2] *= -1
        cld[dr] = points
    return cld

def diplay_gen_ycb(cat, data_dir, models, model_index,  index, do_scale = False):
    base_dir = os.path.join(data_dir, "{}_ycb/data".format(cat), model_index)
    value = loadmat(os.path.join(base_dir, "%06d-meta.mat" % index))
    img_color = plt.imread(os.path.join(base_dir, "%06d-color.png"% index))
    semantic = plt.imread(os.path.join(base_dir, "%06d-label.png"% index))
    depth_img = np.load(os.path.join(base_dir,  "%06d-depth.npy"% index), "r")

    scale = value["scale"][0][0]
    fig,axes = plt.subplots(2, 3,  dpi= 150)
    boxes = value['box']
    axes[0][0].imshow(img_color)
    img = cv2.normalize(depth_img, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    axes[0][1].imshow(img, cmap="gray")
    axes[0][2].imshow(semantic)
    for i in range(len(value["cls_indexes"])):
        idx = value["cls_indexes"][i][0] - 1
        if do_scale:
            point2d = project_to_img(value["intrinsic_matrix"], value["poses"][:,:,i], models[value["model_id"][0]], scale)
        else :
            point2d = project_to_img(value["intrinsic_matrix"], value["poses"][:,:,i], models[value["model_id"][0]])
        axes[1][0].imshow(img_color)
        axes[1][0].scatter(np.asarray(point2d[0,:]), np.asarray(point2d[1,:]), s=20, alpha=0.05)


    axes[1][1].imshow(img_color)
    for i in range(len(value["cls_indexes"])):
        label = boxes[i]
        rect1 = patches.Rectangle((label[0],label[1]),label[2] - label[0], label[3] - label[1],linewidth=1,edgecolor='r',facecolor='none')
        axes[1][1].add_patch(rect1)

        axes[1][1].scatter(value["center"][i][0], value["center"][i][1], s=50)
    x = value
    plt.show()
    return value, boxes, img_color, depth_img, semantic

def display_load_img(img, depth, boxes, label, cam,  pose_t, pose_r, pose, model_pts):
    fig,axes = plt.subplots(2, 3,  dpi= 100)
    rt_mat = np.zeros((4, 4))
    rt_mat[:3,3] = pose_t
    rt_mat[-1,-1] = 1
    rt_mat[:3,:3] = pose[:3,:3]
    # rt_mat = np.matmul(translation_matrix(pose_t), quaternion_matrix(pose_r))

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
    # plt.gca().invert_yaxis()
    plt.show()
