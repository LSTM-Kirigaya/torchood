import torch
import torch.nn as nn
import torch.nn.functional as F

class HODLoss(nn.Module):
    def __init__(self, num_inlier_classes, num_outlier_classes, lambda_coarse=0.1):
        super(HODLoss, self).__init__()
        self.num_inlier_classes = num_inlier_classes
        self.num_outlier_classes = num_outlier_classes
        self.lambda_coarse = lambda_coarse

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # logits: [batch_size, num_inlier_classes + num_outlier_classes]
        # labels: [batch_size], where labels >= num_inlier_classes are outliers

        # Fine-grained loss (cross-entropy for inliers and outliers)
        fine_loss = F.cross_entropy(logits, labels)

        # Coarse-grained loss (binary classification: inlier vs outlier)
        # inlier_probs = torch.sum(F.softmax(logits, dim=1)[:, :self.num_inlier_classes], dim=1)
        outlier_probs = torch.sum(F.softmax(logits, dim=1)[:, self.num_inlier_classes:], dim=1)

        # Coarse labels: 0 for inliers, 1 for outliers
        coarse_labels = (labels >= self.num_inlier_classes).float()
        coarse_loss = F.binary_cross_entropy(outlier_probs, coarse_labels)

        # Combined loss
        total_loss = fine_loss + self.lambda_coarse * coarse_loss

        return total_loss

class HODDetector(nn.Module):
    r"""HOD Loss based OOD
    Pytorch implementation for [MIA 2022: Does Your Dermatology Classifier Know What It Doesn't Know? Detecting the Long-Tail of Unseen Conditions](https://arxiv.org/abs/2104.03829v1?_hsenc=p2ANqtz-8g4edUHChpOYtecmHQ5BQyfEE8FtzPSe_cJRHy1k8R7cmRVuF8pSRpajGRW-eMezmU2aM-ohNGWabNKag6LX4x0zXjEQ&_hsmi=167784826&utm_source=pocket_mylist)
    
    """
    def __init__(self, model, num_inlier_classes, num_outlier_classes, lambda_coarse=0.1):
        super(HODDetector, self).__init__()
        self.model = model
        self.num_inlier_classes = num_inlier_classes
        self.num_outlier_classes = num_outlier_classes
        self.lambda_coarse = lambda_coarse
        self.hod_loss = HODLoss(num_inlier_classes, num_outlier_classes, lambda_coarse)

    def forward(self, x: torch.Tensor):
        # Forward pass through the model
        feature, logits, probs = self.model(x)
        return feature, logits, probs

    def criterion(self, logits: torch.Tensor, labels: torch.Tensor):
        return self.hod_loss(logits, labels)

    def get_ood_score(self, probs: torch.Tensor):
        # OOD score is the sum of outlier probabilities
        outlier_probs = torch.sum(probs[:, self.num_inlier_classes:], dim=1)
        return outlier_probs
