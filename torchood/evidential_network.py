import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

def _default_evidence_builder(feature: torch.Tensor):
    return torch.exp(feature)


class EvidenceNeuralNetwork(nn.Module):
    r"""Evidential Neural Network (ENN)
    Pytorch implementation for [NeurIPS 2018: Evidential Deep Learning to Quantify Classification Uncertainty](https://arxiv.org/pdf/1806.01768)
    
    Args:
        model (nn.Module) : A user defined classifier.
        evidence_builder (Callable[[torch.Tensor], torch.Tensor], optional) : Ground truth class indices or class probabilities;
            see Shape section below for supported shapes.
        alpha_kl (int, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        
    """
    def __init__(
        self,
        model: nn.Module,
        num_class: int,
        evidence_builder: Callable[[torch.Tensor], torch.Tensor] = None,
        alpha_kl: int = 0
    ) -> None:
        super().__init__()
        
        self.model = model
        self.alpha_kl = alpha_kl
        self.num_class = num_class
        if evidence_builder is None:
            self.evidence_builder = _default_evidence_builder
        else:
            self.evidence_builder = evidence_builder        
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""inference of enn

        Returns:
            tuple[Tensor, Tensor, Tensor]: out_feature, evidence, prob
        """
        # acquire output feature of classifier
        feature, logits, probs = self.model(x)
        
        # by default, evidence is the usually modeling by f(x), where
        # x is the output feature of classifier and f is any function whose output value is non-negative.
        evidence = torch.exp(logits)
        alpha = evidence + 1
        prob = F.normalize(alpha, p=1, dim=1)
        
        return feature, evidence, prob
    
    def criterion(self, evidence: torch.Tensor, label: torch.Tensor):
        r"""loss function of enn
        
        .. math::
            \mathcal L(\theta) = \sum_{i=1}^N \mathcal L_i(\theta) +\lambda_t \sum_{i=1}^N \mathrm{KL}\left(D(p_i|\tilde{\alpha}_i) || D(p_i | \bold 1)\right).
        
        where
        
        .. math::
            \lambda_t = \min(1, t / 10).

        Args:
            evidence (Tensor): evidence
            label (Tensor): label
        """
        
        alpha = evidence + 1
        # prob = F.normalize(alpha, dim=1)

        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.num_class)
        label = nn.functional.one_hot(label, num_classes=self.num_class)
        
        loss_ece = torch.sum(label * (torch.digamma(alpha_0) - torch.digamma(alpha)), dim=1)
        loss_ece = torch.mean(loss_ece)

        if self.alpha_kl > 0:
            tilde_alpha = label + (1 - label) * alpha
            uncertainty_alpha = torch.ones_like(tilde_alpha).cuda()
            estimate_dirichlet = torch.distributions.Dirichlet(tilde_alpha)
            uncertainty_dirichlet = torch.distributions.Dirichlet(uncertainty_alpha)
            kl = torch.distributions.kl_divergence(estimate_dirichlet, uncertainty_dirichlet)
            loss_kl = torch.mean(kl)
        else:
            loss_kl = 0
        return loss_ece + self.alpha_kl * loss_kl

    def predict(self, *args) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """make prediction of uncertainty
        
        """
        with torch.no_grad():
            evidence, prob = self.forward(*args)
            alpha = evidence + 1
            S = alpha.sum(dim=1)
            u = self.out_dim / S
            
        return prob, u

class EvidenceReconciledNeuralNetwork(EvidenceNeuralNetwork):
    """Evidence Reconciled Neural Network(ERNN)
    Pytorch implementation for [MICCAI 2023: Evidence Reconciled Neural Network for Out-of-Distribution Detection in Medical Images](https://conferences.miccai.org/2023/papers/249-Paper2401.html)
    
    Args:
        model (nn.Module) : A user defined classifier.
        evidence_builder (Callable[[torch.Tensor], torch.Tensor], optional) : Ground truth class indices or class probabilities;
            see Shape section below for supported shapes.
        alpha_kl (int, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
    """
    def __init__(
        self,
        model: nn.Module,
        num_class: int,
        evidence_builder: Callable[[torch.Tensor], torch.Tensor] = None,
        alpha_kl: int = 0
    ) -> None:
        super().__init__(model, num_class, evidence_builder, alpha_kl)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""inference of enn

        Returns:
            tuple[Tensor, Tensor, Tensor]: out_feature, evidence, prob
        """
        
        feature, logits, probs = self.model(x)

        evidence = torch.exp(logits)
        em_evidence = evidence - torch.min(evidence, dim=1, keepdim=True).values
        prob = F.normalize(em_evidence + 1, p=1, dim=1)
        return feature, em_evidence, prob

class RedundancyRemovingEvidentialNeuralNetwork(EvidenceNeuralNetwork):
    """Redundancy Removing Evidential Neural Network(R2ENN)

    Args:
        EvidenceNeuralNetwork (_type_): _description_
    """
    def __init__(self, model: nn.Module, num_class: int, evidence_builder = None, alpha_kl = 0):
        super().__init__(model, evidence_builder, alpha_kl)

        self.num_class = num_class
        self.mathcal_R = nn.Sequential(
            nn.Linear(num_class, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor):
        feature, logits, probs = self.model(x)
                
        evidence = torch.exp(logits)
        redundant = self.mathcal_R(evidence)
         
        em_evidence = evidence - redundant * torch.min(evidence, dim=1, keepdim=True).values
        prob = F.normalize(em_evidence + 1, p=1, dim=1)
        return feature, em_evidence, prob

    def criterion(self, evidence: torch.Tensor, label: torch.Tensor):
        """
        evicential cross_entropy for ENN
        """
        alpha = evidence + 1
        # prob = F.normalize(alpha, dim=1)

        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.num_class)
        
        label = nn.functional.one_hot(label, num_classes=self.num_class)
        loss_ece = torch.sum(label * (torch.digamma(alpha_0) - torch.digamma(alpha)), dim=1)
        loss_ece = torch.mean(loss_ece)

        if self.alpha_kl > 0:
            tilde_alpha = label + (1 - label) * alpha
            uncertainty_alpha = torch.ones_like(tilde_alpha).cuda()
            estimate_dirichlet = torch.distributions.Dirichlet(tilde_alpha)
            uncertainty_dirichlet = torch.distributions.Dirichlet(uncertainty_alpha)
            kl = torch.distributions.kl_divergence(estimate_dirichlet, uncertainty_dirichlet)
            loss_kl = torch.mean(kl)
        else:
            loss_kl = 0
        return loss_ece + self.alpha_kl * loss_kl
