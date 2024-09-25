
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import ive
import numpy as np
import torch.distributed as dist


def miller_recurrence(nu, x):
    device = x.device
    I_n = torch.ones(1, dtype=torch.float64).to(device)
    I_n1 = torch.zeros(1, dtype=torch.float64).to(device)

    Estimat_n = [nu, nu+1]
    scale0 = 0 
    scale1 = 0 
    scale = 0

    for i in range(2*nu, 0, -1):
        I_n_tem, I_n1_tem = 2*i/x*I_n + I_n1, I_n
        if torch.isinf(I_n_tem).any():
            I_n1 /= I_n
            scale += torch.log(I_n)
            if i >= (nu+1):
                scale0 += torch.log(I_n)
                scale1 += torch.log(I_n)
            elif i == nu:
                scale0 += torch.log(I_n)

            I_n = torch.ones(1, dtype=torch.float64).cuda()
            I_n, I_n1 = 2*i/x*I_n + I_n1, I_n

        else:
            I_n, I_n1 = I_n_tem, I_n1_tem

        if i == nu:
            Estimat_n[0] = I_n1
        elif i == (nu+1):
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

        nu, nu1 = miller_recurrence((p/2 - 1).int(), k.double())
        # nu = log(ive(p/2-1, k)) = log(iv(p/2-1, k)) - k
        # nu1 = log(ive(p/2, k)) = log(iv(p/2, k)) - k

        tensor = nu + k - (p/2 - 1) * torch.log(k+1e-20) - logc
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
    def __init__(self, feature_num, temperature, class_num, max_modes, device):
        super(EstimatorCV, self).__init__()

        self.class_num = class_num
        self.feature_num = feature_num
        self.temperature = temperature
        self.max_modes = max_modes
        self.device = device

        self.Ave = F.normalize(torch.randn(class_num, max_modes, feature_num), dim=1) * 0.9
        self.Amount = torch.zeros(class_num, max_modes)
        self.kappa = torch.ones(class_num, max_modes) * self.feature_num * 90 / 19
        tem = torch.from_numpy(ive(self.feature_num/2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(self.kappa.device)
        self.logc = torch.log(tem+1e-300) + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-300)

        self.Ave = self.Ave.to(self.device)
        self.Amount = self.Amount.to(self.device)
        self.kappa = self.kappa.to(self.device)
        self.logc = self.logc.to(self.device)

    def assign_mode(self, features, labels):
        device = features.device

        N = features.size(0)
        C = self.class_num
        M = self.max_modes

        # Calculate the distance (or similarity) of each feature to every mode of its class
        distances = torch.zeros(N, M, device=device)
        for i in range(N):
            class_label = labels[i].item()
            for m in range(M):
                # Compute cosine similarity or Euclidean distance to each mode #TODO
                distances[i, m] = F.cosine_similarity(features[i], self.Ave[class_label, m], dim=0)

        # Assign each feature to the mode with the highest similarity
        mode_assignments = torch.argmax(distances, dim=1)
        return mode_assignments

    def soft_assign_mode(self, features, labels):
        device = features.device
        N = features.size(0)  # Number of features
        C = self.class_num  # Number of classes
        M = self.max_modes  # Number of modes per class

        mode_probs = torch.zeros(N, M, device=device)

        # Iterate over each feature and compute similarity to each mode of its class
        for i in range(N):
            class_label = labels[i].item()

            # Compute cosine similarities to all modes of the class
            similarities = F.cosine_similarity(features[i].unsqueeze(0), self.Ave[class_label], dim=1)

            # Use softmax to convert similarities to probabilities
            mode_probs[i] = F.softmax(similarities / self.temperature, dim=0)

        return mode_probs  # Shape: [N, M]

    def reset(self):
        device = self.Ave.device  # Get the device from the attribute
        self.Ave = F.normalize(torch.randn(self.class_num, self.feature_num, device=device), dim=1) * 0.9
        self.Amount = torch.zeros(self.class_num, device=device)
        self.kappa = torch.ones(self.class_num, device=device) * self.feature_num * 90 / 19
        tem = torch.from_numpy(ive(self.feature_num / 2 - 1, self.kappa.cpu().numpy().astype(np.float64))).to(device)
        self.logc = torch.log(tem+1e-300) + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-300)

        # if torch.cuda.is_available():
        #     self.Ave = self.Ave.cuda()
        #     self.Amount = self.Amount.cuda()
        #     self.kappa = self.kappa.cuda()
        #     self.logc = self.logc.cuda()

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
        M = self.max_modes
        A = features.size(1)

        # mode_assignments = self.assign_mode(features, labels)
        soft_assignments = self.soft_assign_mode(features, labels)

        NxCxMxFeatures = features.view(
            N, 1, 1, A
        ).expand(
            N, C, M, A
        )
        onehot = torch.zeros(N, C, M, device=device)

        # # Scatter labels to one-hot encoded class-mode assignments
        # for i in range(N):
        #     onehot[i, labels[i], mode_assignments[i]] = 1

        # Assign each feature to a class
        for i in range(N):
            onehot[i, labels[i]] = soft_assignments[i]

        NxCxMxA_onehot = onehot.view(N, C, M, 1).expand(N, C, M, A)

        features_by_sort = NxCxMxFeatures.mul(NxCxMxA_onehot)

        Amount_CxMxA = NxCxMxA_onehot.sum(0)
        Amount_CxMxA[Amount_CxMxA == 0] = 1

        ave_CxMxA = features_by_sort.sum(0) / Amount_CxMxA

        # Update the number of features per class-mode combination
        sum_weight_CMxA = onehot.sum(0).view(C, M, 1).expand(C, M, A)
        weight_CMxA = sum_weight_CMxA.div(sum_weight_CMxA + self.Amount.view(C, M, 1).expand(C, M, A)).to(device)
        weight_CMxA[weight_CMxA != weight_CMxA] = 0  # Handle any NaNs

        self.Ave = (self.Ave.mul(1 - weight_CMxA) + ave_CxMxA.mul(weight_CMxA)).detach()

        self.Amount += onehot.sum(0)

    def update_kappa(self, kappa_inf=False):
        R = torch.linalg.norm(self.Ave, dim=2)
        self.kappa = self.feature_num * R / ( 1 - R**2)

        self.kappa[self.kappa > 1e5] = 1e5
        self.kappa[self.kappa < 0] = 1e5

        nu, _ = miller_recurrence(torch.tensor(self.feature_num / 2 - 1).int().to(self.kappa.device),
                                  self.kappa.double())

        self.logc = nu + self.kappa - (self.feature_num/2 - 1) * torch.log(self.kappa+1e-20)


class ProCoMLoss(nn.Module):
    def __init__(self, contrast_dim, temperature=1.0, num_classes=1000, max_modes=10, device='cuda:0'):
        super(ProCoMLoss, self).__init__()
        self.temperature = temperature
        self.num_classes = num_classes
        self.feature_num = contrast_dim
        self.device = device
        self.max_modes = max_modes
        self.estimator_old = EstimatorCV(self.feature_num, self.temperature, num_classes, self.max_modes, self.device)
        self.estimator = EstimatorCV(self.feature_num, self.temperature, num_classes, self.max_modes, self.device)

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

    def forward(self, features, labels=None, sup_logits=None, world_size=1):
        batch_size = features.size(0)
        N = batch_size
        device = features.device

        if labels is not None:

            # total_features_list = [torch.zeros_like(features) for _ in range(world_size)]
            # total_labels_list = [torch.zeros_like(labels) for _ in range(world_size)]
            #
            # dist.all_gather(total_features_list, features)
            # dist.all_gather(total_labels_list, labels)

            # total_features = torch.cat(total_features_list, dim=0)
            # total_labels = torch.cat(total_labels_list, dim=0)

            total_features = features
            total_labels = labels

            # Assign each feature to a mode
            # mode_assignments = self.estimator.assign_mode(total_features.detach(), total_labels)
            soft_assignments = self.estimator.soft_assign_mode(total_features.detach(), total_labels)

            self.estimator_old.update_CV(total_features.detach(), total_labels)
            self.estimator.update_CV(total_features.detach(), total_labels)
            self.estimator_old.update_kappa()

        Ave = self.estimator_old.Ave.detach()
        Ave_norm = F.normalize(Ave, dim=2)         # Shape: [class_num, max_modes, feature_num]
        logc = self.estimator_old.logc.detach()    # Shape: [class_num, max_modes]
        kappa = self.estimator_old.kappa.detach()  # Shape: [class_num, max_modes]

        Ave_norm = Ave_norm.view(self.num_classes * self.max_modes, -1)
        logc = logc.view(-1)
        kappa = kappa.view(-1)

        similarities = torch.matmul(features, Ave_norm.T) / self.temperature

        # Compute the norm for each logit
        kappa_new = torch.sqrt(kappa ** 2 + 2 * kappa * similarities + 1)

        # Apply the custom log-ratio function to compute the logits
        contrast_logits = LogRatioC.apply(kappa_new, torch.tensor(self.estimator.feature_num), logc)

        contrast_logits = contrast_logits.view(batch_size, self.num_classes, self.max_modes)

        # Weight the logits by the soft assignment probabilities
        # class_logits = torch.sum(contrast_logits * soft_assignments.unsqueeze(1), dim=2)
        class_logits = torch.max(contrast_logits, dim=2)[0]  # Max Pooling Across modes (Best Fit Mode)
        # class_logits = torch.mean(contrast_logits, dim=2)  # Mean pooling across modes
        # class_logits = torch.logsumexp(contrast_logits, dim=2)  # Log-Sum-Exp pooling across modes

        return class_logits












