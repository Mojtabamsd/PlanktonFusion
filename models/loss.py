import torch
import torch.nn as nn
import torch.nn.functional as F
from models.autoencoder import ResNetCustom


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Use built-in PyTorch cross-entropy loss
        loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')

        # Apply reduction if specified
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)

        return loss


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
        focal_weights = torch.where(one_hot == 1, self.alpha * (1 - inputs.softmax(dim=1)) ** self.gamma,
                                    (1 - self.alpha) * (inputs.softmax(dim=1)) ** self.gamma)

        # Apply the focal weights to the cross-entropy loss
        focal_loss = ce_loss * focal_weights[:, 1]

        # Apply reduction if specified
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss


class LogitAdjustmentLoss(nn.Module):
    def __init__(self, weight=None):
        super(LogitAdjustmentLoss, self).__init__()
        self.class_weights = nn.Parameter(weight)

    def forward(self, logits, targets):
        # Apply logit adjustment
        adjusted_logits = logits * self.class_weights

        # Compute cross-entropy loss
        loss = F.cross_entropy(adjusted_logits, targets)

        return loss


class LogitAdjust(nn.Module):

    def __init__(self, cls_num_list, tau=1, weight=None, device='cuda:0'):
        super(LogitAdjust, self).__init__()
        self.device = device
        cls_num_list = torch.FloatTensor(cls_num_list).to(self.device)
        cls_p_list = cls_num_list / cls_num_list.sum()
        m_list = tau * torch.log(cls_p_list)
        self.m_list = m_list.view(1, -1)
        if weight is not None:
            self.weight = weight.to(self.device)
        else:
            self.weight = None

    def forward(self, x, target):
        x_m = x + self.m_list
        return F.cross_entropy(x_m, target, weight=self.weight)


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.5):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, preds, targets):
        errors = targets - preds
        quantile_loss = torch.max((self.quantile - 1) * errors, self.quantile * errors)
        return torch.mean(quantile_loss)


class WeightedMSELoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedMSELoss, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets, label):
        mse_loss = nn.MSELoss()(inputs, targets)
        weighted_mse_loss = mse_loss * self.weight[label]

        # Apply reduction if specified
        if self.reduction == 'mean':
            weighted_mse_loss = torch.mean(weighted_mse_loss)
        elif self.reduction == 'sum':
            weighted_mse_loss = torch.sum(weighted_mse_loss)

        return weighted_mse_loss


class PerceptualReconstructionLoss(nn.Module):
    def __init__(self, config, device, alpha=0.2, beta=0.8):
        super(PerceptualReconstructionLoss, self).__init__()
        self.resnet = ResNetCustom(num_classes=config.sampling.num_class,
                                   latent_dim=config.autoencoder.latent_dim,
                                   gray=config.autoencoder.gray,
                                   pretrained=True)
        self.resnet.to(device)
        self.criterion = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, generated, ground_truth):
        # Extract features from ResNet18
        _, gen_features = self.resnet(generated)
        _, gt_features = self.resnet(ground_truth)

        perceptual_loss = self.criterion(gen_features, gt_features)
        reconstruction_loss = self.criterion(generated, ground_truth)

        return self.alpha * perceptual_loss + self.beta * reconstruction_loss
