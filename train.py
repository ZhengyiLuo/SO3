import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from so3_data import *
from lib.transformations import translation_matrix, quaternion_matrix, quaternion_from_matrix
from scipy.spatial.transform import Rotation
import quaternion as qua

import matplotlib.pyplot as plt
import cv2

boxes3d_gt = np.array([[ 0.4068,  0.4068,  0.4068,  0.4068, -0.4068, -0.4068, -0.4068, -0.4068],
        [ 0.2905, -0.0115, -0.2905,  0.0115,  0.2905, -0.0115, -0.2905,  0.0115],
        [-0.0115,  0.2905,  0.0115, -0.2905, -0.0115,  0.2905,  0.0115, -0.2905]]).T

'''
Let the target given by DataLoader always be quaternion
WHen computing the loss, convert the ground truth to the suitable format
'''

def normalizeEulerAngle(angles):
    for i in range(angles.shape[0]):
        while angles[i] > np.pi:
            angles[i] -= 2*np.pi
        while angles[i] < -np.pi:
            angles[i] += 2*np.pi
    return angles

# The input to the following function is a quaternion in numpy shape
# And the bounding boxes coordinates
def quatToRotRepr(quat, rot_repr, boxes2d):
    rot = Rotation.from_quat(quat.reshape(-1))
    if rot_repr == "quat":
        return quat.reshape(-1).numpy()
    elif rot_repr == "mat":
        return quaternion_matrix(quat)[:3, :3].reshape(-1)
    elif rot_repr == "bbox":
        return boxes2d.reshape(-1).numpy()
    elif rot_repr == "rodr":
        rotvec = rot.as_rotvec().reshape(-1)
        return rotvec
    elif rot_repr == "euler":
        angles = rot.as_euler('xyz', degrees=False)
        angles = normalizeEulerAngle(angles)
        angles = angles.copy()
        return angles.reshape(-1)
    else:
        raise ValueError("Unknown rot_repr: %s" % rot_repr)

def rotReprToRotMat(input, rot_repr, cam=None, boxes3d=boxes3d_gt):
    if rot_repr == "quat":
        R = quaternion_matrix(input)[:3, :3]
    elif rot_repr == "mat":
        R = input.reshape((3,3))
        # re-normalize the rotation matrix by QR decomposition
        # R, _ = np.linalg.qr(R)
    elif rot_repr == "bbox":
        boxes2d = input.reshape(8,2).detach().cpu().numpy()
        (success, rotation_vector, translation_vector) = cv2.solvePnP(boxes3d, boxes2d, cam.numpy(), np.zeros((4,1)))
        # print(success)
        # print(rotation_vector)
        # print(translation_vector)
        rot = Rotation.from_rotvec(rotation_vector.reshape(-1))
        quat = rot.as_quat()
        R = quaternion_matrix(quat)[:3, :3]
    elif rot_repr == "rodr":
        rot = Rotation.from_rotvec(input)
        quat = rot.as_quat()
        R = quaternion_matrix(quat)[:3, :3]
    elif rot_repr == "euler":
        # first normalize the euler angles
        input = input.reshape(-1)
        input = normalizeEulerAngle(input)
        rot = Rotation.from_euler("xyz", input, degrees=False)
        quat = rot.as_quat()
        R = quaternion_matrix(quat)[:3, :3]
    else:
        raise ValueError("Unknown rot_repr: %s" % rot_repr)

    T = np.eye(4)
    T[:3, :3] = R
    return T

def main(args):
    dtype = torch.FloatTensor
    if args.use_gpu:
        dtype = torch.cuda.FloatTensor

    print("Will be save to:", args.save_path)

    # training dataset
    transform = T.Compose([
        # T.Scale(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    dataset_root = args.dataset_root
    train_dataset = PoseDataset('train', dataset_root, transforms=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_dset = PoseDataset('test', dataset_root, transforms=transform)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model
    model = torchvision.models.resnet18(pretrained=True)

    dim_output = DIM_OUTPUT[args.rot_repr]
    model.fc = nn.Linear(model.fc.in_features, dim_output)

    model.type(dtype)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # loss and optimizer
    loss_fn = nn.L1Loss().type(dtype)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)

    plot_x = []
    plot_y = [[], [], [], []]
    plot_name = ["Train distance", "Val distance", "Train loss", "Val loss"]

    print("Epoch 0, before training: ")
    # Check accuracy on the train and val sets.
    train_dist, train_loss = compute_distance_loss_avg(model, train_loader, dtype, rot_repr=args.rot_repr)
    val_dist, val_loss = compute_distance_loss_avg(model, val_loader, dtype, rot_repr=args.rot_repr)
    print('Train Distance: ', train_dist)
    print('Val Distance: ', val_dist)
    print('Train loss: ', train_loss.item())
    print('Val loss: ', val_loss.item())
    print()
    plot_y[0].append(train_dist)
    plot_y[1].append(val_dist)
    plot_y[2].append(train_loss.item())
    plot_y[3].append(val_loss.item())
    plot_x.append(0)

    for epoch in range(args.num_epochs1):
        # Run an epoch over the training data.
        print('Finetuning last layer: epoch %d / %d' % (epoch + 1, args.num_epochs1))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype, rot_repr=args.rot_repr)

        print("Training done, starting evaluation")
        # Check accuracy on the train and val sets.
        train_dist, train_loss = compute_distance_loss_avg(model, train_loader, dtype, rot_repr=args.rot_repr)
        val_dist, val_loss = compute_distance_loss_avg(model, val_loader, dtype, rot_repr=args.rot_repr)
        print('Train Distance: ', train_dist)
        print('Val Distance: ', val_dist)
        print('Train loss: ', train_loss.item())
        print('Val loss: ', val_loss.item())
        print()
        plot_y[0].append(train_dist)
        plot_y[1].append(val_dist)
        plot_y[2].append(train_loss.item())
        plot_y[3].append(val_loss.item())
        plot_x.append(1+epoch)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(args.num_epochs2):
        print('Training all: epoch %d / %d' % (epoch + 1, args.num_epochs2))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype, rot_repr=args.rot_repr)

        print("Training done, starting evaluation")
        # Check accuracy on the train and val sets.
        train_dist, train_loss = compute_distance_loss_avg(model, train_loader, dtype, rot_repr=args.rot_repr)
        val_dist, val_loss = compute_distance_loss_avg(model, val_loader, dtype, rot_repr=args.rot_repr)
        print('Train Distance: ', train_dist)
        print('Val Distance: ', val_dist)
        print('Train loss: ', train_loss.item())
        print('Val loss: ', val_loss.item())
        print()
        plot_y[0].append(train_dist)
        plot_y[1].append(val_dist)
        plot_y[2].append(train_loss.item())
        plot_y[3].append(val_loss.item())
        plot_x.append(1+epoch+args.num_epochs1)

    # Save the model
    torch.save(model.state_dict(), args.save_path+".npy")

    # Save the log
    np.savez(args.save_path + "_log.npz", plot_x=plot_x, plot_y=plot_y, plot_name=plot_name)

    # Visualize and save the progress
    fig, ax1 = plt.subplots(dpi=300)
    ax1.set_xlabel("Epoch")
    ax1.plot(plot_x, plot_y[0], label=plot_name[0])
    ax1.plot(plot_x, plot_y[1], label=plot_name[1])
    ax1.set_ylim([0.0, 1.0])
    ax1.set_ylabel("Average distance")
    plt.legend(loc="lower left")

    ax2 = ax1.twinx()
    ax2.plot(plot_x, plot_y[2], ':', label=plot_name[2])
    ax2.plot(plot_x, plot_y[3], ':', label=plot_name[3])
    ax2.set_ylabel("Average loss")
    plt.legend(loc="upper right")

    # for y, name in zip(plot_y, plot_name):
    #     plt.plot(plot_x, y, label = name)
    plt.savefig(args.save_path + ".png")


def run_epoch(model, loss_fn, loader, optimizer, dtype, rot_repr):
    """
    Train the model for one epoch.
    """
    model.train()
    for i, data in enumerate(loader, 0):
        img, depth, boxes2d, boxes3d, label, pose_r, pose_t, pose, cam,idx= data
        x_var = Variable(img.type(dtype))

        # convert the ground truth quaternion to desired rot_repr
        target = []
        for quat, b2d in zip(pose_r, boxes2d):
            target.append(quatToRotRepr(quat, rot_repr, b2d))
        target = np.stack(target, axis=0)
        target = torch.from_numpy(target.astype(np.float32))
        y_var = Variable(target.type(dtype).float())

        # Run the model forward to compute scores and loss.
        scores = model(x_var)

        loss = loss_fn(scores, y_var)
        # Run the model backward and take a step using the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def compute_distance_loss_avg(model, loader, dtype, rot_repr):
    """
    Check the accuracy of the model.
    """
    # Set the model to eval mode
    model.eval()
    total_distance, total_loss, num_samples = 0.0, 0.0, 0.0
    avg_dists = []
    for i, data in enumerate(loader, 0):
        img, depth, boxes2d, boxes3d, label, pose_r, pose_t, pose, cam,idx= data
        x_var = Variable(img.type(dtype))
        y_var = Variable(pose_r.type(dtype).float())

        # feed into the network
        scores = model(x_var)
        preds = scores.data.cpu()
        for i in range(preds.shape[0]):
            # compute the average distance over all points
            rot1 = rotReprToRotMat(preds[i], rot_repr, cam=cam[i])
            rot2 = quaternion_matrix(pose_r[i])
            dist = comp_rotation(points, rot1, rot2)
            avg_dists.append(dist)

            # compute the loss on the network direct output
            target = quatToRotRepr(pose_r[i], rot_repr, boxes2d[i])
            target = torch.from_numpy(target).type(torch.float32)
            total_loss += torch.abs(preds[i] - target).sum()
            num_samples += 1

    # Return the fraction of datapoints that were correctly classified.
    avg_dist = np.sum(avg_dists)/len(avg_dists)
    avg_loss = total_loss / num_samples
    return avg_dist, avg_loss

def comp_rotation(points, rot1, rot2):
    pt1_proj = rot1.dot(points)
    pt2_proj = rot2.dot(points)
    distance = np.sum(np.linalg.norm(pt1_proj - pt2_proj, axis=0)) / points.shape[1]
    return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_epochs1', default=10, type=int)
    parser.add_argument('--num_epochs2', default=30, type=int)
    parser.add_argument('--dataset_root', default="data/car_ycb_big", type=str)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--rot_repr", type=str, default="quat", choices=["quat", "mat", "bbox", "rodr", "euler"],
                        help="The type of rotation representation the network output")
    parser.add_argument('--save_path', default="output/quaternion", type=str)

    DIM_OUTPUT = {
            "quat": 4,
            "mat": 9,
            "bbox": 16,
            "rodr": 3,
            "euler": 3
    }

    cat = "car"
    # data_dir = "/hdd/zen/dev/6dof/6dof_data/"
    data_dir = "/home/qiaog/courses/16720B-project/SO3/data"
    points_cld = read_pointxyz(os.path.join(data_dir, cat +"_ycb_big", "models"))
    points = np.matrix.transpose(np.hstack((np.matrix(points_cld["0000"]), np.ones(len(points_cld["0000"])).reshape(-1, 1))))

    args = parser.parse_args()
    main(args)
