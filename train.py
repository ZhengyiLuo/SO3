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
from lib.proj_utils import *
from lib.rot_tools import *
import quaternion as qua

import matplotlib.pyplot as plt
import matplotlib
import cv2
from visualize import *

DIM_OUTPUT = {
            "quat": 4,
            "mat": 9,
            "bbox": 16,
            "rodr": 3,
            "euler": 3, 
            "6dof": 6
    }

'''
Let the target given by DataLoader always be quaternion
WHen computing the loss, convert the ground truth to the suitable format
'''

class Net_Rot(nn.Module):
    def __init__(self, rot_repr):
        super(Net_Rot, self).__init__()
        dim_output = DIM_OUTPUT[rot_repr]
        self.ResNet = torchvision.models.resnet18(pretrained=True)
        self.ResNet.fc = nn.Linear(self.ResNet.fc.in_features, dim_output)
        self.rot_repr = rot_repr

    def forward(self, x):
        x = self.ResNet(x)
        return x

    def freeze_prev_layers(self):
        for param in self.ResNet.parameters():
            param.requires_grad = False
        for param in self.ResNet.fc.parameters():
            param.requires_grad = True

    def unfreeze_layers(self):
        for param in self.ResNet.parameters():
            param.requires_grad = True

class geodesic_loss_R(nn.Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.eps = 1e-6

    # batch geodesic loss for rotation matrices
    def bgdR(self,Rgts,Rps):
        Rds = torch.bmm(Rgts.permute(0,2,1),Rps)
        Rt = torch.sum(Rds[:,torch.eye(3).byte()],1) #batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5*(Rt-1), -1+self.eps, 1-self.eps)
        return torch.acos(theta)

    def forward(self, ypred, ytrue):
        theta = self.bgdR(ypred,ytrue)
        if self.reduction == 'mean':
            return torch.mean(theta)
        else:
            return theta

def main(args, model_points):
    matplotlib.use('pdf')
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
    dataset_root = args.data_dir
    train_dataset = PoseDataset('train', dataset_root, transforms=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    val_dset = PoseDataset('test', dataset_root, transforms=transform)
    val_loader = torch.utils.data.DataLoader(val_dset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model
    model = Net_Rot(args.rot_repr)

    model.type(dtype)
    model.freeze_prev_layers()
    
    # loss and optimizer
    # loss_fn = nn.L2Loss().type(dtype)
    loss_fn = nn.MSELoss().type(dtype)
    optimizer = torch.optim.Adam(model.ResNet.fc.parameters(), lr=1e-3)

    plot_x = []
    plot_y = [[], [], [], []]
    plot_name = ["Train distance", "Val distance", "Train loss", "Val loss"]

    print("========================================================================")
    print("Epoch 0, before training: ")
    # Check accuracy on the train and val sets.
    
    train_dist, train_loss = run_epoch(model, loss_fn, train_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, model_points = model_points, train= False)
    val_dist, val_loss = run_epoch(model, loss_fn, val_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, model_points = model_points, train= False)
    print('Train Distance: ', train_dist)
    print('Val Distance: ', val_dist)
    print('Train loss: ', train_loss)
    print('Val loss: ', val_loss)
    print()
    plot_y[0].append(train_dist)
    plot_y[1].append(val_dist)
    plot_y[2].append(train_loss)
    plot_y[3].append(val_loss)
    plot_x.append(0)

    for epoch in range(args.num_epochs1):
        # Run an epoch over the training data.
        print('Finetuning last layer: epoch %d / %d' % (epoch + 1, args.num_epochs1))
        run_epoch(model, loss_fn, train_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, optimizer= optimizer)

        print("Training done, starting evaluation")
        # Check accuracy on the train and val sets.
        train_dist, train_loss = run_epoch(model, loss_fn, train_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, model_points = model_points, train= False)
        val_dist, val_loss = run_epoch(model, loss_fn, val_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, model_points = model_points, train= False)
        print('Train Distance: ', train_dist)
        print('Val Distance: ', val_dist)
        print('Train loss: ', train_loss)
        print('Val loss: ', val_loss)
        print()
        plot_y[0].append(train_dist)
        plot_y[1].append(val_dist)
        plot_y[2].append(train_loss)
        plot_y[3].append(val_loss)
        plot_x.append(1+epoch)

    model.unfreeze_layers()
    print("========================================================================")
    print("start training all!!!!!!!!")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(args.num_epochs2):
        print('Training all: epoch %d / %d' % (epoch + 1, args.num_epochs2))
        run_epoch(model, loss_fn, train_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, optimizer= optimizer)

        print("Training done, starting evaluation")
        # Check accuracy on the train and val sets.
        train_dist, train_loss = run_epoch(model, loss_fn, train_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, model_points = model_points, train= False)
        val_dist, val_loss = run_epoch(model, loss_fn, val_loader, dtype, rot_repr=args.rot_repr, loss_type=args.loss_type, model_points = model_points, train= False)
        print('Train Distance: ', train_dist)
        print('Val Distance: ', val_dist)
        print('Train loss: ', train_loss)
        print('Val loss: ', val_loss)
        print()
        plot_y[0].append(train_dist)
        plot_y[1].append(val_dist)
        plot_y[2].append(train_loss)
        plot_y[3].append(val_loss)
        plot_x.append(1+epoch+args.num_epochs1)

    # Save the model
    torch.save(model.state_dict(), os.path.join(args.save_path, "model.npy"))

    # Save the log
    np.savez(os.path.join(args.save_path,"log.npy"), plot_x=plot_x, plot_y=plot_y, plot_name=plot_name)

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
    plt.savefig(os.path.join(args.save_path,"model.png"))

    ### Visualization, no transform
    transform=transforms.Compose([transforms.ToTensor()])
    train_dataset = PoseDataset('train', dataset_root, transforms=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=1)

    test_dataset = PoseDataset('test', dataset_root, transforms=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=1)


    visualize_model(args.rot_repr, args.save_path, train_loader, model_points, vid_name = "train")
    visualize_model(args.rot_repr, args.save_path, test_loader, model_points, vid_name = "test")


def calc_loss(model_output, loss_fn, rot_repr, dtype, gts, loss_type = "regression"):
    ## regression loss
    pose_r, pose_t, pose, boxes2d, boxes2d_proj, boxes3d = gts
    if loss_type == "regression":
        target = []
        for quat, b2d in zip(pose_r, boxes2d_proj):
            quat = quat.cpu()
            target.append(quatToRotRepr(quat, rot_repr, b2d))
        target = np.stack(target, axis=0)
        target = torch.from_numpy(target.astype(np.float32))
        y_var = Variable(target.type(dtype).float())
        loss = loss_fn(model_output, y_var)

    elif loss_type == "matrot":
        output_mat = rotReprToRotMat(model_output, rot_repr)
        loss = loss_fn(output_mat, pose[:,:,:3]) # only comparing rotation part

    return loss


def run_epoch(model, loss_fn, loader, dtype, rot_repr, loss_type = "regression", train = True, model_points = None, optimizer = None):
    """
    Train the model for one epoch.
    """
    total_distance, total_loss, num_samples = 0.0, 0.0, 0.0
    if train:
        model.train()
    else: 
        model.eval()
        avg_dists = []
    
    for i, data in enumerate(loader, 0):
        img, depth, boxes2d, boxes2d_proj, boxes3d, label, pose_r, pose_t, pose, cam,idx = data
        
        x_var = Variable(img.type(dtype))
        scores = model(x_var)

        gts = (pose_r.type(dtype), pose_t.type(dtype), pose.type(dtype), boxes2d.type(dtype), boxes2d_proj.type(dtype), boxes3d.type(dtype))
        loss = calc_loss(scores, loss_fn, rot_repr, dtype, gts, loss_type = loss_type)
        
        total_loss += loss.item()
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            preds_rot = rotReprToRotMat(scores, rot_repr)
            for i in range(preds_rot.shape[0]):
                rot1 = preds_rot[i].detach().cpu().numpy()
                rot1_4 = np.eye(4)
                rot1_4[:3,:3] = rot1
                rot2 = quaternion_matrix(pose_r[i])
                dist = compare_rotation(model_points, rot1_4[:3,:4], rot2[:3,:4])
                avg_dists.append(dist)
                num_samples += 1

    avg_loss = total_loss / len(loader) # Divide by unber of epoches for 
    
    if train:
        return avg_loss
    else:
        avg_dist = np.sum(avg_dists)/len(avg_dists)
        return avg_dist, avg_loss

def compare_rotation(points, rot1, rot2):
    points = np.matrix.transpose(np.hstack((np.matrix(points), np.ones(len(points)).reshape(-1, 1))))
    pt1_proj = rot1.dot(points)
    pt2_proj = rot2.dot(points)

    distance = np.sum(np.linalg.norm(pt1_proj - pt2_proj, axis=0)) / points.shape[1]
    return distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_epochs1', default=20, type=int)
    parser.add_argument('--num_epochs2', default=50, type=int)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--rot_repr", type=str, default="quat", choices=["quat", "mat", "bbox", "rodr", "euler", "6dof"],
                        help="The type of rotation representation the network output")
    parser.add_argument('--save_path', default="output/quaternion", type=str)
    parser.add_argument('--data_dir', default="/hdd/zen/dev/6dof/6dof_data/so3/big/car_ycb/", type=str)
    parser.add_argument('--loss_type', default="regression", type=str)

    args = parser.parse_args()

    data_dir = args.data_dir
    # data_dir = "/home/qiaog/courses/16720B-project/SO3/data"
    points_cld = read_pointxyz(os.path.join(data_dir, "models"))
    # model_points = np.matrix.transpose(np.hstack((np.matrix(points_cld["0000"]), np.ones(len(points_cld["0000"])).reshape(-1, 1))))
    model_points = points_cld["0000"]
    
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    
    main(args, model_points)
