import torch.nn as nn

from Partical_approach.functions import GradientReverseLayerF


class LocalAutoEncoder(nn.Module):
    def __init__(self):
        super(LocalAutoEncoder, self).__init__()

        self.autoEncoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(),
        )  # ??

    def forward(self, X):
        return self.autoEncoder(X)


class LocalDiscriminator(nn.Module):
    def __init__(self, num_classes=65):
        super(LocalDiscriminator, self).__init__()

        self.classes = num_classes
        self.alpha = 0.0

        # local domain discriminator
        self.dci = {}
        for i in range(num_classes):
            self.dci[i] = nn.Sequential()
            self.dci[i].add_module('fc1', nn.Linear(256, 1024))
            self.dci[i].add_module('relu1', nn.ReLU(True))
            self.dci[i].add_module('dpt1', nn.Dropout())
            self.dci[i].add_module('fc2', nn.Linear(1024, 512))
            self.dci[i].add_module('relu2', nn.ReLU(True))
            self.dci[i].add_module('dpt2', nn.Dropout())
            self.dci[i].add_module('fc3', nn.Linear(512, 2))
            self.dcis.add_module('dci_' + str(i), self.dci[i])

    def set_alpha(self, alpha_new):
        self.alpha = alpha_new

    def forward(self, X, class_num):

        reverse_feature = GradientReverseLayerF.apply(X, self.alpha)

        return self.dci[class_num](reverse_feature)
