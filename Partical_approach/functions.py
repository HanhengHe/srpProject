import torch
import torch.nn as nn
from torch.autograd import Function


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
