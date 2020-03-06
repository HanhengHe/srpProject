import torch
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import math

from Separated_model.functions import load_data
from Separated_model.Global.GobalAdapt import FeatureExtractor, GlobalAlign, GlobalClassifier
from Separated_model.args import args

import tqdm

DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
torch.cuda.manual_seed(args.seed)


def train_global(epoch_, feS_model, feT_model, gD_model, gC_model, sourceLoader, targetLoader):
    LEARNING_RATE = args.lr / math.pow((1 + 10 * (epoch_ - 1) / args.global_epochs), 0.75)

    feS_optimizer = optim.SGD(feS_model.parameters(), lr=LEARNING_RATE,
                              momentum=args.momentum, weight_decay=args.l2_decay)
    feT_optimizer = optim.SGD(feT_model.parameters(), lr=LEARNING_RATE,
                              momentum=args.momentum, weight_decay=args.l2_decay)
    ga_optimizer = optim.SGD(gD_model.parameters(), lr=LEARNING_RATE,
                             momentum=args.momentum, weight_decay=args.l2_decay)
    gc_optimizer = optim.SGD(gC_model.parameters(), lr=LEARNING_RATE,
                             momentum=args.momentum, weight_decay=args.l2_decay)

    len_dataLoader = len(sourceLoader)

    feS_model.train()
    feT_model.train()
    gD_model.train()
    gC_model.train()

    label_source = torch.from_numpy(np.array([[0, 0]] * args.batch_size)).float().to(DEVICE)
    label_target = torch.from_numpy(np.array([[1, 1]] * args.batch_size)).float().to(DEVICE)

    for batch_idx, (source_data, source_label) in tqdm.tqdm(enumerate(sourceLoader),
                                                            total=len_dataLoader,
                                                            desc='Train epoch = {}'.format(epoch_),
                                                            ncols=80,
                                                            leave=False):
        p = float(batch_idx + 1 + epoch_ * len_dataLoader) / args.global_epochs / len_dataLoader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        source_data, source_label = source_data.to(DEVICE), source_label.to(DEVICE)
        for t_data, _ in targetLoader:
            target_data = t_data.to(DEVICE)
            break

        # Discriminator
        for _ in range(args.global_D_step):
            feS_optimizer.zero_grad()
            feT_optimizer.zero_grad()
            ga_optimizer.zero_grad()
            gc_optimizer.zero_grad()

            source_out = feS_model(source_data)
            target_out = feT_model(target_data)

            gD_model.set_alpha(alpha)

            source_outD = gD_model(source_out)
            source_outC = gC_model(source_out)
            target_outD = gD_model(target_out)
            lossC = F.cross_entropy(source_outC, source_label)
            lossD = F.binary_cross_entropy(source_outD, label_source)
            lossD += F.binary_cross_entropy(target_outD, label_target)
            # ??
            loss = lossC + lossD
            loss.backward()
            ga_optimizer.step()
            gc_optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(
                '\nTotal Loss:{:.6f}, Classifier Loss: {:.6f}, Discriminator Loss: {:.6f}'.format(
                    loss.item(), lossC.item(), lossD.item()))

        # Feature extractor
        feS_optimizer.zero_grad()
        feT_optimizer.zero_grad()
        ga_optimizer.zero_grad()
        gc_optimizer.zero_grad()

        source_out = feS_model(source_data)
        target_out = feT_model(target_data)

        gD_model.set_alpha(alpha)

        source_outD = gD_model(source_out)
        source_outC = gC_model(source_out)
        lossC = F.cross_entropy(source_outC, source_label)
        target_outD = gD_model(target_out)

        lossD = F.binary_cross_entropy(source_outD, label_source)
        lossD += F.binary_cross_entropy(target_outD, label_target)
        # ??
        # loss = alpha * lossC + (1 - alpha) * lossD
        loss = lossC + lossD
        loss.backward()
        feS_optimizer.step()
        feT_optimizer.step()


def test_global(feT_model, gC_model, testLoader):
    feT_model.eval()
    gC_model.eval()
    test_loss = 0
    correct_ = 0
    with torch.no_grad():
        for data, labels in testLoader:
            data, labels = data.to(DEVICE), labels.to(DEVICE)
            out = feT_model(data)
            out = gC_model(out)
            test_loss += F.nll_loss(out, labels).item()  # sum up batch loss
            pred = out.data.max(1)[1]  # get the index of the max log-probability
            correct_ += pred.eq(labels.data.view_as(pred)).cpu().sum()

        test_loss /= len(testLoader.dataset)
        print(args.test_dir, '\nTest set: Average loss on classifier: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct_, len(testLoader.dataset),
            100. * correct_ / len(testLoader.dataset)))
    return correct_


def Global_align(lAlign_S, lAlign_T, gDiscriminator, gClassifier):
    correct = 0

    for epoch_ in range(1, args.global_epochs + 1):

        source_loader, target_loader, test_loader = load_data()

        # train
        train_global(epoch_, lAlign_S, lAlign_T, gDiscriminator, gClassifier,
                     source_loader, target_loader)

        # test
        t_correct = test_global(lAlign_T, gClassifier, test_loader)  # !
        if t_correct > correct:
            correct = t_correct
        print(args.source_dir, "to", args.test_dir, end='\t')
        print("%s max correct:" % args.test_dir, correct)


if __name__ == '__main__':
    featureExtractor_S = FeatureExtractor(base_net='ResNet50').to(DEVICE)
    featureExtractor_T = FeatureExtractor(base_net='ResNet50').to(DEVICE)
    globalDiscriminator = GlobalAlign().to(DEVICE)
    globalClassifier = GlobalClassifier(num_classes=args.num_class).to(DEVICE)

    Global_align(featureExtractor_S, featureExtractor_T, globalDiscriminator, globalClassifier)

    torch.save(featureExtractor_S.state_dict(),
               "D:\\workspace\\PycharmProjects\\srpProject\\Separated_model\\model\\featureExtractor_S")
    torch.save(featureExtractor_T.state_dict(),
               "D:\\workspace\\PycharmProjects\\srpProject\\Separated_model\\model\\featureExtractor_T")
