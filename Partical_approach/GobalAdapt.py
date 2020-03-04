import torch.nn as nn

import Partical_approach.backbone as backbone
from Partical_approach.functions import ReverseLayerF, Replacement


class GobalAdapt(nn.Module):

    def __init__(self, num_classes=65, base_net='ResNet50'):
        super(GobalAdapt, self).__init__()
        self.sharedNet = backbone.network_dict[base_net]()
        self.bottleneck = nn.Linear(2048, 256)
        self.source_fc = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classes = num_classes

        # global domain discriminator
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('fc1', nn.Linear(256, 1024))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dpt1', nn.Dropout())
        self.domain_classifier.add_module('fc2', nn.Linear(1024, 1024))
        self.domain_classifier.add_module('relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dpt2', nn.Dropout())
        self.domain_classifier.add_module('fc3', nn.Linear(1024, 2))


    def forward(self, source, target, s_label, DEV, alpha=0.0):
        source_share = self.sharedNet(source)
        source_share = self.bottleneck(source_share)
        source = self.source_fc(source_share)
        p_source = self.softmax(source)

        a = p_source.cpu()

        target = self.sharedNet(target)
        target = self.bottleneck(target)
        t_label = self.source_fc(target)
        s_out = []
        t_out = []

        if self.training is True:
            # RevGrad
            s_reverse_feature = ReverseLayerF.apply(source_share, alpha)
            t_reverse_feature = ReverseLayerF.apply(target, alpha)
            s_domain_output = self.domain_classifier(s_reverse_feature)
            t_domain_output = self.domain_classifier(t_reverse_feature)


        else:
            s_domain_output = 0
            t_domain_output = 0
            s_out = [0] * self.classes
            t_out = [0] * self.classes
        return source, s_domain_output, t_domain_output, s_out, t_out,
