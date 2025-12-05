"""
Loss functions for RLD training.

Implements knowledge distillation loss to avoid DQGAN's conflicting
gradient problem. Single clean objective from teacher model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation loss for ranking.

    Uses KL divergence between teacher and student score distributions
    with temperature scaling for better gradient flow.

    Args:
        temperature: Temperature for softening distributions (default: 1.0)
        lambda_kd: Weight for KD loss (default: 1.0)
        lambda_bce: Weight for BCE smoothing loss (default: 0.3)
    """

    def __init__(
        self,
        temperature: float = 1.0,
        lambda_kd: float = 1.0,
        lambda_bce: float = 0.3
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_kd = lambda_kd
        self.lambda_bce = lambda_bce

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Compute combined KD + BCE loss.

        Args:
            student_logits: Student model scores [batch_size, num_docs]
            teacher_logits: Teacher model scores [batch_size, num_docs]
            labels: Ground truth labels [batch_size, num_docs]

        Returns:
            Combined loss scalar
        """
        # KL divergence loss with temperature
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        kd_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)  # Scale by T^2 as per Hinton et al.

        # BCE loss for ground truth smoothing
        bce_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            labels.float()
        )

        # Combined loss
        total_loss = self.lambda_kd * kd_loss + self.lambda_bce * bce_loss

        return total_loss, {
            'kd_loss': kd_loss.item(),
            'bce_loss': bce_loss.item(),
            'total_loss': total_loss.item()
        }


class ListwiseRankingLoss(nn.Module):
    """
    Listwise ranking loss (alternative to KD when no teacher available).

    Uses ListMLE (List Maximum Likelihood Estimation) for direct
    optimization of ranking quality.
    """

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute ListMLE loss.

        Args:
            logits: Model scores [batch_size, num_docs]
            labels: Relevance labels [batch_size, num_docs]

        Returns:
            Loss scalar
        """
        # Sort documents by relevance
        sorted_labels, sorted_indices = torch.sort(labels, descending=True, dim=-1)

        # Get scores in relevance order
        batch_indices = torch.arange(logits.size(0)).unsqueeze(1).expand_as(sorted_indices)
        sorted_logits = logits[batch_indices, sorted_indices]

        # Compute ListMLE loss
        max_logits = sorted_logits.max(dim=-1, keepdim=True)[0]
        normalized_logits = sorted_logits - max_logits

        # Cumulative log-sum-exp from bottom to top
        exp_logits = torch.exp(normalized_logits)
        cumsum_exp = torch.cumsum(exp_logits.flip(dims=[-1]), dim=-1).flip(dims=[-1])

        # Only consider relevant documents
        mask = (sorted_labels > 0).float()
        loss = -torch.sum(
            mask * (normalized_logits - torch.log(cumsum_exp + 1e-10)),
            dim=-1
        ) / (torch.sum(mask, dim=-1) + 1e-10)

        return loss.mean()


class AdaptiveTemperatureLoss(nn.Module):
    """
    Knowledge Distillation with learnable temperature.

    The temperature automatically adapts to dataset difficulty,
    avoiding manual tuning.

    Args:
        initial_temperature: Starting temperature (default: 1.0)
        min_temperature: Minimum allowed temperature (default: 0.1)
        max_temperature: Maximum allowed temperature (default: 10.0)
        lambda_kd: Weight for KD loss
        lambda_bce: Weight for BCE loss
    """

    def __init__(
        self,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.1,
        max_temperature: float = 10.0,
        lambda_kd: float = 1.0,
        lambda_bce: float = 0.3
    ):
        super().__init__()

        # Learnable temperature parameter (in log space for stability)
        self.log_temperature = nn.Parameter(
            torch.tensor(float(initial_temperature)).log()
        )
        self.min_temp = min_temperature
        self.max_temp = max_temperature

        self.lambda_kd = lambda_kd
        self.lambda_bce = lambda_bce

    @property
    def temperature(self):
        """Get current temperature value (clamped)"""
        temp = self.log_temperature.exp()
        return torch.clamp(temp, self.min_temp, self.max_temp)

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Compute adaptive temperature KD loss.

        Args:
            student_logits: Student scores [batch_size, num_docs]
            teacher_logits: Teacher scores [batch_size, num_docs]
            labels: Ground truth labels [batch_size, num_docs]

        Returns:
            Loss scalar and metrics dict
        """
        temp = self.temperature

        # KL divergence with adaptive temperature
        student_probs = F.log_softmax(student_logits / temp, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1)

        kd_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (temp ** 2)

        # BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            student_logits,
            labels.float()
        )

        # Combined loss
        total_loss = self.lambda_kd * kd_loss + self.lambda_bce * bce_loss

        return total_loss, {
            'kd_loss': kd_loss.item(),
            'bce_loss': bce_loss.item(),
            'total_loss': total_loss.item(),
            'temperature': temp.item()
        }


class MarginRankingLoss(nn.Module):
    """
    Margin-based ranking loss for pairwise document comparison.

    Ensures relevant documents score higher than irrelevant ones
    by a margin.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Compute margin ranking loss.

        Args:
            logits: Model scores [batch_size, num_docs]
            labels: Binary relevance labels [batch_size, num_docs]

        Returns:
            Loss scalar
        """
        batch_size, num_docs = logits.shape

        # Create pairs of (relevant, irrelevant) documents
        pos_mask = labels > 0
        neg_mask = labels == 0

        losses = []
        for i in range(batch_size):
            pos_scores = logits[i][pos_mask[i]]
            neg_scores = logits[i][neg_mask[i]]

            if len(pos_scores) == 0 or len(neg_scores) == 0:
                continue

            # Compute all pairwise margins
            pos_expanded = pos_scores.unsqueeze(1)  # [num_pos, 1]
            neg_expanded = neg_scores.unsqueeze(0)  # [1, num_neg]

            # Loss: max(0, margin - (pos_score - neg_score))
            pairwise_loss = F.relu(self.margin - (pos_expanded - neg_expanded))
            losses.append(pairwise_loss.mean())

        if len(losses) == 0:
            return torch.tensor(0.0, device=logits.device)

        return torch.stack(losses).mean()
