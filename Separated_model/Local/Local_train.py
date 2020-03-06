import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math

from Separated_model.Global.GobalAdapt import FeatureExtractor
from Separated_model.Local.LocalAdapt import LocalAutoEncoder, LocalAlign, LocalClassifier
from Separated_model.functions import load_data
from Separated_model.args import args

import tqdm

DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def train_local(epoch_, fExtractor_S, fExtractor_T, lAlign_S, lAlign_T, lDiscriminator, lClassifier, source_loader,
                target_loader):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch_ - 1) / args.local_epochs), 0.75)

    laS_optimizer = optim.SGD(lAlign_S.parameters(), lr=LEARNING_RATE,
                              momentum=args.momentum, weight_decay=args.l2_decay)
    laT_optimizer = optim.SGD(lAlign_T.parameters(), lr=LEARNING_RATE,
                              momentum=args.momentum, weight_decay=args.l2_decay)
    lD_optimizer = optim.SGD(lDiscriminator.parameters(), lr=LEARNING_RATE,
                             momentum=args.momentum, weight_decay=args.l2_decay)
    lC_optimizer = []
    for _ in range(len(lDiscriminator)):
        optimizer = optim.SGD(lDiscriminator[i].parameters(), lr=LEARNING_RATE,
                              momentum=args.momentum, weight_decay=args.l2_decay)
        lC_optimizer.append(optimizer)

    len_dataLoader = len(source_loader)

    for parm in fExtractor_S.parameters():
        parm.requires_grad = False

    for parm in fExtractor_T.parameters():
        parm.requires_grad = False

    lAlign_S.train()
    lAlign_T.train()
    lDiscriminator.train()
    lClassifier.train()

    for batch_idx, (source_data, source_label) in tqdm.tqdm(enumerate(source_loader),
                                                            total=len_dataLoader,
                                                            desc='Train epoch = {}'.format(epoch_),
                                                            ncols=80,
                                                            leave=False):
        p = float(batch_idx + 1 + epoch_ * len_dataLoader) / args.global_epochs / len_dataLoader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        for t_data, _ in target_loader:
            target_data = t_data.to(DEVICE)
            break

        source_in = fExtractor_S(source_data)
        target_in = fExtractor_T(target_data)

        # train a classifier first
        for _ in range(args.local_C_step):
            lC_optimizer.zero_grad()

            source_out = lAlign_S(source_in)

            source_outC = lClassifier(source_out)
            lossC = F.cross_entropy(source_outC, source_label)
            lossC.backward()
            lC_optimizer.step()

        print('\nstep Classifier, Loss: {:.6f}'.format(lossC.item()), end='')

        # build fake label on target domain
        target_out = lAlign_T(target_in)
        target_label = lClassifier(target_out).data.max(1)[1]

        # Discriminator
        for _ in range(args.local_D_step):
            laS_optimizer.zero_grad()
            laT_optimizer.zero_grad()
            lD_optimizer.zero_grad()

            source_out = lAlign_S(source_in)
            target_out = lAlign_T(target_in)

            lDiscriminator.set_alpha(alpha)

            # SGD here !!! Problems here
            for index in range(len(source_label)):
                lC_optimizer[source_label[:, index].item()].zero_grad()
                lDiscriminator[source_label[:, index].item()].set_alpha(alpha)
                loss_index = lDiscriminator[source_label[:, index].item()](source_out[:, index])
                loss_index = F.cross_entropy(loss_index, torch.from_numpy(np.array([[0, 0]])).float().to(DEVICE))
                loss_index.backward()
                lC_optimizer[source_label[:, index].item()].step()

            for index in range(len(target_label)):
                lC_optimizer[target_label[:, index].item()].zero_grad()
                lDiscriminator[target_label[:, index].item()].set_alpha(alpha)
                loss_index = lDiscriminator[target_label[:, index].item()](target_out[:, index])
                loss_index = F.cross_entropy(loss_index, torch.from_numpy(np.array([[1, 1]])).float().to(DEVICE))
                loss_index.backward()
                lC_optimizer[target_label[:, index].item()].step()

        # Align autoEncoder
        laS_optimizer.zero_grad()
        laT_optimizer.zero_grad()
        lD_optimizer.zero_grad()

        source_out = lAlign_S(source_in)
        target_out = lAlign_T(target_in)

        total_Loss_S = 0
        total_Loss_T = 0

        # SGD here !!! Problems here
        for index in range(len(source_label)):
            lC_optimizer[source_label[:, index].item()].zero_grad()
            laS_optimizer.zero_grad()
            loss_index = lDiscriminator[source_label[:, index].item()](source_out[:, index])
            loss_index_C = lClassifier(source_out[:, index])
            loss_index = F.cross_entropy(loss_index, torch.from_numpy(np.array([[0, 0]])).float().to(DEVICE))
            loss_index_C = F.cross_entropy(loss_index_C, source_label[:, index].item())
            loss = loss_index_C + loss_index
            total_Loss_S += loss.item()
            loss.backward()
            laS_optimizer.step()


        for index in range(len(target_label)):
            lC_optimizer[source_label[:, index].item()].zero_grad()
            laT_optimizer.zero_grad()
            loss_index = lDiscriminator[target_label[:, index].item()](target_out[:, index])
            loss_index = F.cross_entropy(loss_index, torch.from_numpy(np.array([[1, 1]])).float().to(DEVICE))
            total_Loss_T += loss_index.item()
            loss_index.backward()
            laT_optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('\tStep local align, Total Loss on source: {:.6f}, Total Loss on target: {:.6f}'.format(
                    total_Loss_S.item(), total_Loss_T.item()))


def test_local(feT_model, laT_model, lC_model, test_loader):
    feT_model.eval()
    laT_model.eval()
    lC_model.eval()
    test_loss = 0
    correct_ = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            out = feT_model(data)
            out = laT_model(out)
            out = lC_model(out)
            test_loss += F.nll_loss(out, labels).item()  # sum up batch loss
            pred = out.data.max(1)[1]  # get the index of the max log-probability
            correct_ += pred.eq(labels.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print(args.test_dir, '\nTest set: Average loss on classifier: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct_, len(test_loader.dataset),
            100. * correct_ / len(test_loader.dataset)))
    return correct_


def Local_align(fExtractor_S, fExtractor_T, lAlign_S, lAlign_T, lDiscriminator, lClassifier):
    correct = 0

    for epoch_ in range(1, args.local_epochs + 1):
        source_loader, target_loader, test_loader = load_data()

        train_local(epoch_, fExtractor_S, fExtractor_T, lAlign_S, lAlign_T, lDiscriminator, lClassifier,
                    source_loader, target_loader)

        t_correct = test_local(fExtractor_T, lAlign_T, lClassifier, test_loader)  # !
        if t_correct > correct:
            correct = t_correct
        print(args.source_dir, "to", args.test_dir, end='\t')
        print("%s max correct:" % args.test_dir, correct)


if __name__ == '__main__':

    featureExtractor_S = FeatureExtractor(base_net='ResNet50')
    featureExtractor_T = FeatureExtractor(base_net='ResNet50')

    featureExtractor_S.load_state_dict(
        torch.load("D:\\workspace\\PycharmProjects\\srpProject\\Separated_model\\model\\featureExtractor_S"))
    featureExtractor_T.load_state_dict(
        torch.load("D:\\workspace\\PycharmProjects\\srpProject\\Separated_model\\model\\featureExtractor_T"))

    featureExtractor_S = featureExtractor_S.to(DEVICE)
    featureExtractor_T = featureExtractor_T.to(DEVICE)

    localAlign_S = LocalAutoEncoder().to(DEVICE)
    localAlign_T = LocalAutoEncoder().to(DEVICE)
    localDiscriminator = LocalAlign().to(DEVICE)
    localClassifier = {}
    for i in range(len(args.num_class)):
        localClassifier[i] = LocalClassifier(num_classes=65).to(DEVICE)

    Local_align(featureExtractor_S, featureExtractor_T, localAlign_S, localAlign_T, localDiscriminator, localClassifier)
