import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ive
import numpy as np


def miller_recurrence(nu, x):
    device = x.device
    I_n = torch.ones(1, dtype=torch.float64).to(device)
    I_n1 = torch.zeros(1, dtype=torch.float64).to(device)

    Estimat_n = [nu, nu + 1]
    scale0 = 0
    scale1 = 0
    scale = 0

    for i in range(2 * nu, 0, -1):
        I_n_tem, I_n1_tem = 2 * i / x * I_n + I_n1, I_n
        if torch.isinf(I_n_tem).any():
            I_n1 /= I_n
            scale += torch.log(I_n)
            if i >= (nu + 1):
                scale0 += torch.log(I_n)
                scale1 += torch.log(I_n)
            elif i == nu:
                scale0 += torch.log(I_n)

            I_n = torch.ones(1, dtype=torch.float64).to(device)
            I_n, I_n1 = 2 * i / x * I_n + I_n1, I_n

        else:
            I_n, I_n1 = I_n_tem, I_n1_tem

        if i == nu:
            Estimat_n[0] = I_n1
        elif i == (nu + 1):
            Estimat_n[1] = I_n1

    ive0 = torch.special.i0e(x)

    Estimat_n[0] = torch.log(ive0) + torch.log(Estimat_n[0]) - torch.log(I_n) + scale0 - scale
    Estimat_n[1] = torch.log(ive0) + torch.log(Estimat_n[1]) - torch.log(I_n) + scale1 - scale

    return Estimat_n[0], Estimat_n[1]


class LogRatioC(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, k, p, logc):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """

        nu, nu1 = miller_recurrence((p / 2 - 1).int(), k.double())
        # nu = log(ive(p/2-1, k)) = log(iv(p/2-1, k)) - k
        # nu1 = log(ive(p/2, k)) = log(iv(p/2, k)) - k

        tensor = nu + k - (p / 2 - 1) * torch.log(k + 1e-20) - logc
        # tensor = log(ive(p/2-1, k)) + k - (p/2 - 1) * log(k) - logc
        #        = log(iv(p/2-1, k)) - (p/2 - 1) * log(k) - logc
        #        = log(C(\tilde{\kappa})/C(\kappa))
        ctx.save_for_backward(torch.exp(nu1 - nu))

        # d/dk iv(p/2-1, k) = iv(p/2, k) + (p/2 - 1) / k * iv(p/2-1, k)
        #
        # d/dk tensor = (iv(p/2, k) + (p/2 - 1) / k * iv(p/2-1, k)) / iv(p/2-1, k) - (p/2 - 1) / k
        #             = iv(p/2, k) / iv(p/2-1, k)
        #             = exp(nu1 - nu)
        #

        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """

        grad = ctx.saved_tensors[0]
        # grad clip
        grad[grad > 1.0] = 1.0

        grad *= grad_output

        return grad, None, None


class EstimatorCV():
    def __init__(self, feature_num, class_num, device):
        super(EstimatorCV, self).__init__()

        self.class_num = class_num
        self.feature_num = feature_num
        self.device = device
        self.Ave = F.normalize(torch.randn(class_num, feature_num), dim=1) * 0.9
        self.Amount = torch.zeros(class_num)
        self.kappa = torch.ones(class_num) * self.feature_num * 90 / 19
        tem = torch.from_numpy(ive(self.feature_num / 2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(
            self.kappa.device)
        self.logc = torch.log(tem + 1e-300) + self.kappa - (self.feature_num / 2 - 1) * torch.log(self.kappa + 1e-300)

        self.Ave = self.Ave.to(self.device)
        self.Amount = self.Amount.to(self.device)
        self.kappa = self.kappa.to(self.device)
        self.logc = self.logc.to(self.device)

    def reset(self):
        device = self.Ave.device  # Get the device from the attribute
        self.Ave = F.normalize(torch.randn(self.class_num, self.feature_num, device=device), dim=1) * 0.9
        self.Amount = torch.zeros(self.class_num, device=device)
        self.kappa = torch.ones(self.class_num, device=device) * self.feature_num * 90 / 19
        tem = torch.from_numpy(ive(self.feature_num / 2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(device)
        self.logc = torch.log(tem + 1e-300) + self.kappa - (self.feature_num / 2 - 1) * torch.log(self.kappa + 1e-300)

    def reload_memory(self):
        device = self.Ave.device
        self.Ave = self.Ave.to(device)
        self.Amount = self.Amount.to(device)
        self.kappa = self.kappa.to(device)
        self.logc = self.logc.to(device)

    def update_CV(self, features, labels):
        device = features.device

        self.Ave = self.Ave.to(device)
        self.Amount = self.Amount.to(device)
        self.kappa = self.kappa.to(device)
        self.logc = self.logc.to(device)

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(
            N, 1, A
        ).expand(
            N, C, A
        )
        onehot = torch.zeros(N, C, device=device)

        onehot.scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        sum_weight_AV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_AV = sum_weight_AV.div(
            sum_weight_AV + self.Amount.view(C, 1).expand(C, A)
        ).to(device)
        weight_AV[weight_AV != weight_AV] = 0

        self.Ave = (self.Ave.mul(1 - weight_AV) + ave_CxA.mul(weight_AV)).detach()

        self.Amount += onehot.sum(0)

    def update_kappa(self, kappa_inf=False):
        R = torch.linalg.norm(self.Ave, dim=1)
        self.kappa = self.feature_num * R / (1 - R ** 2)

        self.kappa[self.kappa > 1e5] = 1e5
        self.kappa[self.kappa < 0] = 1e5

        nu, _ = miller_recurrence(torch.tensor(self.feature_num / 2 - 1).int().to(self.kappa.device),
                                  self.kappa.double())

        self.logc = nu + self.kappa - (self.feature_num / 2 - 1) * torch.log(self.kappa + 1e-20)


class ProCoUNLoss(nn.Module):
    def __init__(self, contrast_dim, temperature=1.0, num_classes=1000, device='cuda:0', sampling_option='down_sample'):
        super(ProCoUNLoss, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_num = contrast_dim
        self.device = device
        self.estimator_old = EstimatorCV(self.feature_num, num_classes, self.device)
        self.estimator = EstimatorCV(self.feature_num, num_classes, self.device)
        self.sampling_option = sampling_option

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        if torch.cuda.is_available():
            self.weight = self.weight.to(self.weight.device)

    def reload_memory(self):

        self.estimator_old.reload_memory()
        self.estimator.reload_memory()

    def _hook_before_epoch(self, epoch, epochs):
        # exchange ave and covariances

        self.estimator_old.Ave = self.estimator.Ave
        self.estimator_old.Amount = self.estimator.Amount

        self.estimator_old.kappa = self.estimator.kappa
        self.estimator_old.logc = self.estimator.logc

        self.estimator.reset()

    def forward(self, features, labels=None):
        batch_size = features.size(0)
        N = batch_size
        device = features.device

        if labels is not None:
            total_features = features
            total_labels = labels

            self.estimator_old.update_CV(total_features.detach(), total_labels)
            self.estimator.update_CV(total_features.detach(), total_labels)
            self.estimator_old.update_kappa()

        Ave = self.estimator_old.Ave.detach()
        Ave_norm = F.normalize(Ave, dim=1)
        logc = self.estimator_old.logc.detach()
        kappa = self.estimator_old.kappa.detach()

        uncertainty_metric = 'kappa'  # kappa or cosine
        if uncertainty_metric == 'cosine':
            class_prototypes = Ave[labels]
            cos_sim = F.cosine_similarity(features, class_prototypes, dim=1)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            uncertainty = 1 - cos_sim
        else:
            uncertainty = 1 / (kappa[labels] + 1e-8)

        if self.sampling_option == 'over_sample':
            sampling_probs = uncertainty.clone()
        elif self.sampling_option == 'down_sample':
            inverse_uncertainty = 1 / (uncertainty + 1e-8)
            sampling_probs = inverse_uncertainty.clone()
        else:
            sampling_probs = None

        if sampling_probs is not None:
            sampling_probs[torch.isnan(sampling_probs)] = 0
            sampling_probs[torch.isinf(sampling_probs)] = 0
            sampling_probs = torch.clamp(sampling_probs, min=0)

            total_prob = sampling_probs.sum()
            if total_prob == 0 or not torch.isfinite(total_prob):
                sampling_probs = torch.ones_like(sampling_probs) / sampling_probs.numel()
            else:
                sampling_probs = sampling_probs / total_prob

            sampling_probs = sampling_probs.to(device)
            if self.sampling_option == 'over_sample':
                indices = torch.multinomial(sampling_probs, batch_size, replacement=True)
            elif self.sampling_option == 'down_sample':
                indices = torch.multinomial(sampling_probs, batch_size, replacement=False)
            else:
                indices = torch.arange(batch_size)

            features = features[indices]
            labels = labels[indices]
            uncertainty = uncertainty[indices]

            if uncertainty_metric == 'cosine':
                class_prototypes = class_prototypes[indices]

        if uncertainty_metric == 'cosine':
            cos_sim = F.cosine_similarity(features, class_prototypes, dim=1)
            uncertainty = 1 - cos_sim

        N = features.size(0)

        tem = kappa.reshape(-1, 1) * Ave_norm
        tem = tem.unsqueeze(0) + features[:N].unsqueeze(1) / self.temperature
        kappa_new = torch.linalg.norm(tem, dim=2)

        contrast_logits = LogRatioC.apply(kappa_new, torch.tensor(self.estimator.feature_num), logc)
        uncertainty = uncertainty.unsqueeze(1)  # [N, 1]
        weight = (1 - uncertainty)

        adjusted_logits = contrast_logits * weight

        return adjusted_logits












