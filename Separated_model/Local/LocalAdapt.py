import torch.nn as nn

from Separated_model.functions import GradientReverseLayerF


class LocalAutoEncoder(nn.Module):
    def __init__(self):
        super(LocalAutoEncoder, self).__init__()

        self.autoEncoder = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
        )  # ??

    def forward(self, X):
        return self.autoEncoder(X)


class LocalAlign(nn.Module):
    def __init__(self):
        super(LocalAlign, self).__init__()

        self.alpha = 0.0

        # local domain discriminator
        self.dci = nn.Sequential()
        self.dci.add_module('fc1', nn.Linear(512, 1024))
        self.dci.add_module('relu1', nn.ReLU(True))
        self.dci.add_module('dpt1', nn.Dropout())
        self.dci.add_module('fc2', nn.Linear(1024, 512))
        self.dci.add_module('relu2', nn.ReLU(True))
        self.dci.add_module('dpt2', nn.Dropout())
        self.dci.add_module('fc3', nn.Linear(512, 2))


    def set_alpha(self, alpha_new):
        self.alpha = alpha_new

    def forward(self, X):
        reverse_feature = GradientReverseLayerF.apply(X, self.alpha)

        return self.dci(reverse_feature)


class LocalClassifier(nn.Module):
    def __init__(self, num_classes=65):
        super().__init__()

        self.classes = num_classes
        self.alpha = 0.0

        # classifier
        self.classifier = nn.Sequential()

        self.classifier.add_module('fc1', nn.Linear(512, 256))
        self.classifier.add_module('relu1', nn.ReLU(True))
        self.classifier.add_module('dpt1', nn.Dropout())

        self.classifier.add_module('fc2', nn.Linear(256, num_classes))
        self.classifier.add_module('classifier_out', nn.Softmax(dim=1))

    def forward(self, X):
        return self.classifier(X)
