import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # Calculate focal weights
        one_hot = F.one_hot(targets, num_classes=inputs.size(1)).float()
        focal_weights = torch.where(one_hot == 1, self.alpha * (1 - inputs.softmax(dim=1)) ** self.gamma, (1 - self.alpha) * inputs.softmax(dim=1) ** self.gamma)

        # Apply the focal weights to the cross-entropy loss
        focal_loss = ce_loss * focal_weights[:, 1]

        # Apply reduction if specified
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss