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

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=10, type=int)
parser.add_argument('--dataset_root', default="data/car_ycb", type=str)
parser.add_argument('--use_gpu', action='store_true')
parser.add_argument("--rot_repr", type=str, default="quat", choices=["quat", "mat", "bbox", "rodr"], 
                    help="The type of rotation representation the network output")
parser.add_argument('--save_path', default="output/quaternion.npy", type=str)

DIM_OUTPUT = {
        "quat": 4,
        "mat": 9,
        "bbox": 24,
        "rodr": 3,
}

cat = "car"
data_dir = "/hdd/zen/dev/6dof/6dof_data/"
points_cld = read_pointxyz(os.path.join(data_dir, cat +"_ycb", "models"))
points = np.matrix.transpose(np.hstack((np.matrix(points_cld["0001"]), np.ones(len(points_cld["0001"])).reshape(-1, 1))))

def main(args):
    dtype = torch.FloatTensor
    if args.use_gpu:
        dtype = torch.cuda.FloatTensor

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

    for epoch in range(args.num_epochs1):
        # Run an epoch over the training data.
        print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs1))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype)

        # Check accuracy on the train and val sets.
        train_acc = check_accuracy(model, train_loader, dtype)
        val_acc = check_accuracy(model, val_loader, dtype)
        print('Train Distance: ', train_acc)
        print('Val Distance: ', val_acc)
        print()

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(args.num_epochs2):
        print('Starting epoch %d / %d' % (epoch + 1, args.num_epochs2))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype)

        train_acc = check_accuracy(model, train_loader, dtype)
        val_acc = check_accuracy(model, val_loader, dtype)
        print('Train Distance: ', train_acc)
        print('Val Distance: ', val_acc)
        print()

    torch.save(model.state_dict(), args.save_path)


def run_epoch(model, loss_fn, loader, optimizer, dtype):
    """
    Train the model for one epoch.
    """
    model.train()
    for i, data in enumerate(loader, 0):
        img, depth, boxes, label, pose_r, pose_t, pose, cam,idx= data
        x_var = Variable(img.type(dtype))
        y_var = Variable(pose_r.type(dtype).float())

        # Run the model forward to compute scores and loss.
        scores = model(x_var)
        loss = loss_fn(scores, y_var)
        # Run the model backward and take a step using the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def check_accuracy(model, loader, dtype):
    """
    Check the accuracy of the model.
    """
    # Set the model to eval mode
    model.eval()
    num_correct, num_samples = 0, 0
    avg_dists = []
    for i, data in enumerate(loader, 0):
        img, depth, boxes, label, pose_r, pose_t, pose, cam,idx= data
        # Cast the image data to the correct type and wrap it in a Variable. At
        # test-time when we do not need to compute gradients, marking the Variable
        # as volatile can reduce memory usage and slightly improve speed.
        x_var = Variable(img.type(dtype))
        y_var = Variable(pose_r.type(dtype).float())

        # Run the model forward, and compare the argmax score with the ground-truth
        # category.
        scores = model(x_var)
        preds = scores.data.cpu()
        for i in range(preds.shape[0]):
            rot1 = quaternion_matrix(preds[i])
            rot2 = quaternion_matrix(pose_r[i])
            dist = comp_rotation(points, rot1, rot2)
            avg_dists.append(dist)


    # Return the fraction of datapoints that were correctly classified.
    avg_dist = np.sum(avg_dists)/len(avg_dists)
    return avg_dist

def comp_rotation(points, rot1, rot2):
    pt1_proj = rot1.dot(points)
    pt2_proj = rot2.dot(points)
    distance = np.sum(np.linalg.norm(pt1_proj - pt2_proj, axis=0)) / points.shape[1]
    return distance 


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
