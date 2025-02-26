import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Callable

class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim, init_weight=True):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.feat_dim,self.n_classes).cuda(), requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers)

    def forward(self, x):   
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, - dist



class PrototypeNetwork(nn.Module):
    r"""Prototype Networks (PrototypeNet)
    Pytorch implementation for [CVPR 2018: Rbust classification with convolutional prototype learning](https://arxiv.org/pdf/1805.03438)
    
    Reference Repo: https://github.com/liyiheng123/Robust-Classification-with-Convolutional-Prototype-Learning-Pytorch
    """
    def __init__(
        self,
        model: nn.Module,
        # output dim of model
        output_dim=1000,
        num_hidden_units=2,
        num_classes=10,
        scale=2
    ):
        super().__init__()
        self.model = model
        self.scale = scale
        self.num_classes = num_classes
        
        self.dce = dce_loss(num_classes, num_hidden_units)
        
        self.ip = nn.Sequential(
            nn.PReLU(),
            nn.Linear(output_dim, 2)
        )


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _, _ = self.model(x)
        features = self.ip(x)
        centers, distance = self.dce(features)
        outputs = F.log_softmax(self.scale * distance, dim=1)
        
        return features, centers, distance, outputs

    def regularization(self, features: torch.Tensor, centers: torch.Tensor, labels: torch.Tensor):
        distance = features - torch.t(centers)[labels]
        distance = torch.sum(torch.pow(distance,2),1, keepdim=True)
        distance = torch.sum(distance, 0, keepdim=True) / features.shape[0]
        return distance


    def criterion(self, features, centers, outputs, labels, reg: float = .001):
        if labels.max() >= self.num_classes:
            raise ValueError(f'label\'s value must in range of [0, {self.num_classes - 1}], but max(label) = {labels.max()}')
        
        loss_1 = F.nll_loss(outputs, labels)        
        loss_2 = self.regularization(features, centers, labels)
        loss = loss_1 + reg * loss_2
        return loss
