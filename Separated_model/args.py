import argparse
import torch

# Training settings

parser = argparse.ArgumentParser(description='PyTorch Domain Adapt')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--global_epochs', type=int, default=5, metavar='N',
                    help='number of global epochs to train (default: 10)')
parser.add_argument('--local_epochs', type=int, default=5, metavar='N',
                    help='number of local epochs to train (default: 10)')
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
parser.add_argument('--save_path', type=str, default=".\\tmp\\origin_",
                    help='the path to save the model')
parser.add_argument('--root_path', type=str, default="D:\\workspace\\PycharmProjects\\data\\OfficeHomeDataset\\",
                    help='the path to load the data')
parser.add_argument('--source_dir', type=str, default="Clipart",
                    help='the name of the source dir')
parser.add_argument('--test_dir', type=str, default="Product",
                    help='the name of the test dir')
parser.add_argument('--gamma', type=int, default=1,
                    help='the fc layer and the sharenet have different or same learning rate')
parser.add_argument('--num_class', default=65, type=int,
                    help='the number of classes')
parser.add_argument('--global_D_step', default=5, type=int,
                    help='number discriminator step')
parser.add_argument('--local_C_step', default=5, type=int,
                    help='number discriminator step')
parser.add_argument('--local_D_step', default=5, type=int,
                    help='number discriminator step')
parser.add_argument('--gpu', default=0, type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
