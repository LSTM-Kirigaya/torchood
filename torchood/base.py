import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleCNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.resnet = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
        if in_dim != 3:
            ori_conv1 = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=ori_conv1.out_channels,
                                          kernel_size=ori_conv1.kernel_size,
                                          stride=ori_conv1.stride, padding=ori_conv1.padding)
        feature_dim = self.resnet.fc.out_features

        self.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)

    def forward(self, inputs):
        feature = self.resnet(inputs)
        logits = self.fc(feature)
        prob = F.softmax(logits, dim=1)

        return feature, logits, prob

    def criterion(self, feature, logits, label):
        pred = torch.argmax(logits, dim=1)
        target = torch.argmax(label, dim=1)

        loss_ce = F.cross_entropy(logits, target)
        return loss_ce