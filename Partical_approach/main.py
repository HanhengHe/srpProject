import torch
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import argparse
import math

from Partical_approach import data_loader

# Training settings
parser = argparse.ArgumentParser(description='PyTorch DAAN')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',  # origin: 200
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=233, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--l2_decay', type=float, default=5e-4,
                    help='the L2  weight decay')
parser.add_argument('--save_path', type=str, default="./tmp/origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default="D:\\workspace\\PycharmProjects\\data\\OfficeHomeDataset\\",
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="Clipart",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="Product",
                    help='the name of the test dir')
parser.add_argument('--diff_lr', type=bool, default=True,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--num_class', default=65, type=int,
                    help='the number of classes')
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def load_data():
    source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
    return source_train_loader, target_train_loader, target_test_loader
