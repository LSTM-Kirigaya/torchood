
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet

class PosteriorNetwork(nn.Module):
    r"""Posterior Network (PostNet)
    Pytorch implementation for [NeurIPS 2020: Posterior Network: Uncertainty Estimation without OOD Samples via Density-Based Pseudo-Counts](https://proceedings.neurips.cc/paper/2020/hash/0eac690d7059a8de4b48e90f14510391-Abstract.html)
    
    Reference Repo: https://github.com/gtegner/PriorNetworks
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the Posterior Network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            feature (torch.Tensor): Backbone feature map.
            logits (torch.Tensor): Classifier logits.
            probs (torch.Tensor): Softmax probabilities.
        """
        # Forward pass through the user-defined model
        feature, logits, probs = self.model(x)

        return feature, logits, probs

    def criterion(self, logits: torch.Tensor, target: torch.Tensor, regr: float = 1e-5):
        """Loss function for Posterior Network (UCE loss).

        Args:
            logits (torch.Tensor): Classifier logits.
            target (torch.Tensor): Target labels.
            regr (float): Regularization factor.

        Returns:
            loss (torch.Tensor): Computed loss.
        """
        alpha = torch.exp(logits)  # Dirichlet parameters
        alpha_0 = alpha.sum(dim=1, keepdim=True)  # Sum of alpha (precision)
        
        # UCE loss
        uce_loss = torch.sum(F.one_hot(target, num_classes=logits.size(-1)) * (torch.digamma(alpha_0) - torch.digamma(alpha)))
        
        # Entropy regularization
        entropy_reg = Dirichlet(alpha).entropy()
        loss = uce_loss - regr * entropy_reg.sum()

        return loss
