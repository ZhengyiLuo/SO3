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
import quaternion as quat
from scipy.spatial.transform import Rotation
from lib.rot_tools import *


boxes3d_gt = np.array([[ 0.4068,  0.4068,  0.4068,  0.4068, -0.4068, -0.4068, -0.4068, -0.4068],
        [ 0.2905, -0.0115, -0.2905,  0.0115,  0.2905, -0.0115, -0.2905,  0.0115],
        [-0.0115,  0.2905,  0.0115, -0.2905, -0.0115,  0.2905,  0.0115, -0.2905]]).T

def normalizeEulerAngle(angles):
    for i in range(angles.shape[0]):
        while angles[i] > np.pi:
            angles[i] -= 2*np.pi
        while angles[i] < -np.pi:
            angles[i] += 2*np.pi
    return angles

# The input to the following function is a quaternion in numpy shape
# And the bounding boxes coordinates
def quatToRotRepr(quat, rot_repr, input = None):
    # First generate 
    rot = Rotation.from_quat(quat.reshape(-1))
    if rot_repr == "quat":
        return quat.reshape(-1).numpy()
    elif rot_repr == "mat":
        return quaternion_matrix(quat)[:3, :3].reshape(-1)
    elif rot_repr == "bbox":
        return input.reshape(-1).numpy()
    elif rot_repr == "rodr":
        rotvec = rot.as_rotvec().reshape(-1)
        return rotvec
    elif rot_repr == "euler":
        angles = rot.as_euler('xzy', degrees=False)
        angles = normalizeEulerAngle(angles)
        angles = angles.copy()
        return angles.reshape(-1)
    else:
        raise ValueError("Unknown rot_repr: %s" % rot_repr)

def rotReprToRotMat(inputs, rot_repr):
    
    if rot_repr == "quat":
        Rs = compute_rotation_matrix_from_quaternion(inputs)
    elif rot_repr == "mat":
        Rs = inputs.reshape((-1, 3,3))
    elif rot_repr == "rodr":
        Rs = compute_rotation_matrix_from_Rodriguez(inputs)
    elif rot_repr == "euler":
        Rs = compute_rotation_matrix_from_euler(inputs)
    elif rot_repr == "6dof":
        Rs = compute_rotation_matrix_from_ortho6d(inputs)
    else:
        raise ValueError("Unknown rot_repr: %s" % rot_repr)

    return Rs


# def rotReprToRotMat(input, rot_repr, cam=None, boxes3d=boxes3d_gt):
#     if rot_repr == "quat":
#         R = quaternion_matrix(input)[:3, :3]
#     elif rot_repr == "mat":
#         R = input.reshape((3,3))
#         # re-normalize the rotation matrix by QR decomposition
#         # R, _ = np.linalg.qr(R)
#     elif rot_repr == "bbox":
#         boxes2d = input.reshape(8,2).detach().cpu().numpy()
#         (success, rotation_vector, translation_vector) = cv2.solvePnP(boxes3d, boxes2d, cam.numpy(), np.zeros((4,1)))
#         # print(success)
#         # print(rotation_vector)
#         # print(translation_vector)
#         rot = Rotation.from_rotvec(rotation_vector.reshape(-1))
#         quat = rot.as_quat()
#         R = quaternion_matrix(quat)[:3, :3]
#     elif rot_repr == "rodr":
#         rot = Rotation.from_rotvec(input)
#         quat = rot.as_quat()
#         R = quaternion_matrix(quat)[:3, :3]
#     elif rot_repr == "euler":
#         R = compute_rotation_matrix_from_euler(input)
#     else:
#         raise ValueError("Unknown rot_repr: %s" % rot_repr)

#     T = np.eye(4)
#     T[:3, :3] = R
#     return T


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
    boxes = value['box2d']
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
