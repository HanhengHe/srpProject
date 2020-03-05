import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import argparse
import math

from Partical_approach import data_loader
from Partical_approach.GobalAdapt import FeatureExtractor, MixturePost

import tqdm

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Domain Adapt')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
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


def train(epoch_, feS_model, feT_model, mp_model, sourceLoader, targetLoader):
    # total_progress_bar = tqdm.tqdm(desc='Train iter', total=args.epochs)
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch_ - 1) / args.epochs), 0.75)

    feS_optimizer = optim.SGD(feS_model.parameters(), lr=LEARNING_RATE,
                              momentum=args.momentum, weight_decay=args.l2_decay)
    feT_optimizer = optim.SGD(feT_model.parameters(), lr=LEARNING_RATE,
                              momentum=args.momentum, weight_decay=args.l2_decay)
    mp_optimizer = optim.SGD(mp_model.parameters(), lr=LEARNING_RATE,
                             momentum=args.momentum, weight_decay=args.l2_decay)

    len_dataLoader = len(sourceLoader)

    feS_model.train()
    feT_model.train()
    mp_model.train()

    label_source = torch.from_numpy(np.array([[0, 0]]*args.batch_size)).float().to(DEVICE)
    label_target = torch.from_numpy(np.array([[1, 1]]*args.batch_size)).float().to(DEVICE)

    for batch_idx, (source_data, source_label) in tqdm.tqdm(enumerate(sourceLoader),
                                                            total=len_dataLoader,
                                                            desc='Train epoch = {}'.format(epoch_),
                                                            ncols=80,
                                                            leave=False):
        p = float(batch_idx + 1 + epoch_ * len_dataLoader) / args.epochs / len_dataLoader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        feS_optimizer.zero_grad()
        feT_optimizer.zero_grad()
        mp_optimizer.zero_grad()

        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        for t_data, t_label in targetLoader:
            target_data, target_label = t_data.to(DEVICE), t_label.to(DEVICE)
            break

        source_out = feS_model(source_data)
        target_out = feT_model(target_data)

        mixturePost.set_alpha(alpha)

        source_outC, source_outD = mixturePost(source_out)
        _, target_outD = mixturePost(target_out)
        lossC = F.cross_entropy(source_outC, source_label)
        lossD = F.binary_cross_entropy(source_outD, label_source)
        lossD += F.binary_cross_entropy(target_outD, label_target)
        # ??
        loss = (1-alpha) * lossC + alpha * lossD
        loss.backward()
        feS_optimizer.step()
        feT_optimizer.step()
        mp_optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                '\nTotal Loss: {:.6f},  Classifier Loss: {:.6f},  Discriminator Loss: {:.6f}'.format(
                    loss.item(), lossC.item(), lossD.item()))
        # total_progress_bar.update(1)


def test(feT_model, mp_model, testLoader):
    feT_model.eval()
    mp_model.eval()
    test_loss = 0
    correct_ = 0
    with torch.no_grad():
        for data, target in testLoader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            out = feT_model(data)
            out, _ = mp_model(out)
            test_loss += F.nll_loss(out, target,
                                    size_average=False).item()  # sum up batch loss
            pred = out.data.max(1)[1]  # get the index of the max log-probability
            correct_ += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(testLoader.dataset)
        print(args.test_dir, '\nTest set: Average loss on classifier: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct_, len(testLoader.dataset),
            100. * correct_ / len(testLoader.dataset)))
    return correct_


if __name__ == '__main__':

    correct = 0

    featureExtractor_S = FeatureExtractor(base_net='ResNet50').to(DEVICE)
    featureExtractor_T = FeatureExtractor(base_net='ResNet50').to(DEVICE)
    mixturePost = MixturePost(num_classes=args.num_class).to(DEVICE)

    for epoch in range(1, args.epochs + 1):
        source_loader, target_loader, test_loader = load_data()

        train(epoch, featureExtractor_S, featureExtractor_T, mixturePost,
              source_loader, target_loader)
        t_correct = test(featureExtractor_T, mixturePost, test_loader)
        if t_correct > correct:
            correct = t_correct
        print(args.source_dir, "to", args.test_dir, end='\t')
        print("%s max correct:" % args.test_dir, correct)
