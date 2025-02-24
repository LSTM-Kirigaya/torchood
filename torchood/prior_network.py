import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Dirichlet
from torch.distributions.kl import kl_divergence


class PriorNetwork(nn.Module):
    r"""Predictive Uncertainty Estimation via Prior Networks(Prior Networks)
    Pytorch implementation for [NeurIPS 2018: Predictive Uncertainty Estimation via Prior Networks](https://papers.nips.cc/paper_files/paper/2018/hash/3ea2db50e62ceefceaf70a9d9a56a6f4-Abstract.html)
    
    Reference Repo: https://github.com/gtegner/PriorNetworks
    """
    def __init__(self, model: nn.Module, num_class: int, a0 = 0.2):
        super().__init__()
        self.num_class = num_class
        h = 50
        self.model = model

        self.eps = 0.01
        self.a0 = a0
        self.eps_var = 1e-6

    def forward(self, x: torch.Tensor):
        r"""inference function of PriorNetwork
    
        Args:
            x (Tensor): input
        
        Returns:
            mean (Tensor): probs for classification.
            alpha (Tensor): Dirichlet parameter, for uncertainty modeling.
            precision (Tensor): sum of `alpha`.
        
        """
        # x = F.relu(self.fc1(x))
        # logits = self.fc2(x)
        
        _, logits, _ = self.model(x)
        alpha = torch.exp(logits)
        precision = torch.sum(alpha, 1)
        mean = F.softmax(logits, 1)

        return mean, alpha, precision

    def kl_loss(self, y_precision, y_alpha, precision, alpha):
        loss = torch.lgamma(y_precision + self.eps_var) - torch.sum(torch.lgamma(y_alpha + self.eps_var), 1) \
            - torch.lgamma(precision + self.eps_var) \
            + torch.sum(torch.lgamma(alpha + self.eps_var), 1)

        l2 = torch.sum((y_alpha - alpha)
                       * (torch.digamma(y_alpha + self.eps_var) - torch.digamma(alpha + self.eps_var)), 1)

        return loss + l2

    
    def criterion(self, alpha_id: torch.Tensor, precision_id: torch.Tensor, alpha_ood: torch.Tensor, precision_ood: torch.Tensor, label: torch.Tensor):
        r"""Loss function of PriorNetwork
    
        Args:
        
        Returns:
        
        """

        kl_id = self.id_loss(alpha_id, precision_id, label)
        kl_ood = self.ood_loss(alpha_ood, precision_ood)
        
        return torch.mean(kl_id + kl_ood)

    def id_loss(self, alpha_id, precision_id, label):
        precision_id = precision_id.unsqueeze(1)
        sample_num = alpha_id.shape[0]
        
        # KL IN
        y_onehot = torch.FloatTensor(sample_num, self.num_class).to('cuda')
        y_onehot.zero_()        
        y_onehot.scatter_(1, label.view(-1, 1), 1)

        mu_c = y_onehot * (1 - self.eps * (self.num_class - 1))
        mu_c += self.eps * torch.ones_like(y_onehot)

        target_in_alpha = mu_c * self.a0
        target_in_precision = self.a0 * torch.ones((sample_num, 1)).to('cuda')
        
        kl_id = self.kl_loss(
            target_in_precision,
            target_in_alpha,
            precision_id,
            alpha_id
        )

        return kl_id

    def ood_loss(self, alpha_ood, precision_ood):
        precision_ood = precision_ood.unsqueeze(1)
        sample_num = alpha_ood.shape[0]
        
        # KL OUT
        target_out_alpha = torch.ones((sample_num, self.num_class)).float().to('cuda')
        target_out_precision = self.num_class * torch.ones(sample_num, 1).to('cuda')

        kl_ood = self.kl_loss(
            target_out_precision,
            target_out_alpha,
            precision_ood,
            alpha_ood
        )
        return kl_ood