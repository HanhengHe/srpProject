import torch.nn as nn

import Partical_approach.backbone as backbone
from Partical_approach.functions import GradientReverseLayerF


class FeatureExtractor(nn.Module):

    def __init__(self, base_net='ResNet50'):
        super(FeatureExtractor, self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()

    def forward(self, tensor):
        return self.sharedNet(tensor)


class MixturePost(nn.Module):
    def __init__(self, num_classes=65):
        super().__init__()

        self.classes = num_classes
        self.alpha = 0.0

        # classifier
        self.classifier = nn.Sequential()
        self.classifier.add_module('fc1', nn.Linear(2048, 1024))
        self.classifier.add_module('relu1', nn.ReLU(True))
        self.classifier.add_module('dpt1', nn.Dropout())

        self.classifier.add_module('fc2', nn.Linear(1024, 256))
        self.classifier.add_module('relu2', nn.ReLU(True))
        self.classifier.add_module('dpt2', nn.Dropout())

        self.classifier.add_module('fc3', nn.Linear(256, num_classes))
        self.classifier.add_module('classifier_out', nn.Softmax(dim=1))

        # discriminator
        self.discriminator = nn.Sequential()
        self.discriminator.add_module('fc1', nn.Linear(2048, 1024))
        self.discriminator.add_module('relu1', nn.ReLU(True))
        self.discriminator.add_module('dpt1', nn.Dropout())

        self.discriminator.add_module('fc2', nn.Linear(1024, 256))
        self.discriminator.add_module('relu2', nn.ReLU(True))
        self.discriminator.add_module('dpt2', nn.Dropout())

        self.discriminator.add_module('fc3', nn.Linear(256, 2))
        self.discriminator.add_module('discriminator_out', nn.Sigmoid())

    def set_alpha(self, alpha_new):
        self.alpha = alpha_new

    def forward(self, X, alpha=0.0):
        dis_out = GradientReverseLayerF.apply(X, alpha)
        return self.classifier(X), self.discriminator(dis_out)
