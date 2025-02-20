import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Callable

def __default_evidence_builder(feature: torch.Tensor):
    return torch.exp(feature)



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
        features_square=torch.sum(torch.pow(x,2),1, keepdim=True)
        centers_square=torch.sum(torch.pow(self.centers,2),0, keepdim=True)
        features_into_centers=2*torch.matmul(x, (self.centers))
        dist=features_square+centers_square-features_into_centers

        return self.centers, - dist



class PrototypeNetworks(nn.Module):
    r"""Robust Classification with Convolutional Prototype Learning (Prototype Networks)
    Pytorch implementation for [CVPR 2018: Rbust classification with convolutional prototype learning](https://arxiv.org/pdf/1805.03438)
    
    Reference Repo: https://github.com/liyiheng123/Robust-Classification-with-Convolutional-Prototype-Learning-Pytorch
    """
    def __init__(
        self,
        model: nn.Module,
        num_hidden_units=2,
        num_classes=10,
        scale=2
    ):
        super().__init__()
        self.model = model
        self.scale = scale

        self.preluip_1 = nn.PReLU()
        self.ip_1 = nn.Linear(128 * 3 * 3, 2)
        
        self.dce = dce_loss(
            num_classes=num_classes,
            num_hidden_units=num_hidden_units
        )


    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.model(x)
        x = x.view(-1, 128 * 3 * 3)

        features = self.preluip_1(self.ip_1(x))
        centers, distance = self.dce(features)
        outputs = F.log_softmax(self.scale * x, dim=1)
        return features, centers, distance, outputs

    def regularization(self, features, centers, labels):
        distance = features - torch.t(centers)[labels]
        distance = torch.sum(torch.pow(distance,2),1, keepdim=True)
        distance = torch.sum(distance, 0, keepdim=True) / features.shape[0]
        return distance


    def criterion(self, features, centers, outputs, labels, reg: float):
        loss_1 = F.nll_loss(outputs, labels)
        loss_2 = self.regularization(features, centers, labels)
        loss = loss_1 + reg * loss_2
        return loss


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
        evidence_builder: Callable[[torch.Tensor], torch.Tensor] = None,
        alpha_kl: int = 0
    ) -> None:
        super().__init__()
        
        self.model = model
        self.alpha_kl = alpha_kl
        if evidence_builder is None:
            self.evidence_builder = __default_evidence_builder
        else:
            self.evidence_builder = evidence_builder        
    
    def forward(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""inference of enn

        Returns:
            tuple[Tensor, Tensor, Tensor]: out_feature, evidence, prob
        """
        # acquire output feature of classifier
        feature = self.model(*args)
        # by default, evidence is the usually modeling by f(x), where
        # x is the output feature of classifier and f is any function whose output value is non-negative.
        evidence = self.evidence_builder(feature)
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

        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.out_dim)
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
        evidence_builder: Callable[[torch.Tensor], torch.Tensor] = None,
        alpha_kl: int = 0
    ) -> None:
        super().__init__(model, evidence_builder, alpha_kl)
    
    def forward(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""inference of enn

        Returns:
            tuple[Tensor, Tensor, Tensor]: out_feature, evidence, prob
        """
        
        feature = self.model(*args)
        evidence = self.evidence_builder(feature)
        em_evidence = evidence - torch.min(evidence, dim=1, keepdim=True).values
        prob = F.normalize(em_evidence + 1, p=1, dim=1)
        return feature, em_evidence, prob

class RedundancyRemovingEvidentialNeuralNetwork(EvidenceNeuralNetwork):
    """Redundancy Removing Evidential Neural Network(R2ENN)

    Args:
        EvidenceNeuralNetwork (_type_): _description_
    """
    def __init__(self, model, evidence_builder = None, alpha_kl = 0):
        super().__init__(model, evidence_builder, alpha_kl)


