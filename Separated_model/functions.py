import torch
import torch.nn as nn
from torch.autograd import Function

from Separated_model import data_loader
from Separated_model.args import args


class GradientReverseLayerF(Function):
    def __init__(self):
        super(GradientReverseLayerF, self).__init__()

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}


def load_data():
    source_train_loader = data_loader.load_training(args.root_path, args.source_dir, args.batch_size, kwargs)
    target_train_loader = data_loader.load_training(args.root_path, args.test_dir, args.batch_size, kwargs)
    target_test_loader = data_loader.load_testing(args.root_path, args.test_dir, args.batch_size, kwargs)
    return source_train_loader, target_train_loader, target_test_loader
