import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin):
        """
        If reduce == False, we calculate sample loss, instead of batch loss.
        """
        super(TripletLoss, self).__init__()
        self.triplet_margin_loss = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(self, anchor, positive, negative):
        """
        Triplet loss for model.
        """

        # Normalize embeddings
        # anchor = F.normalize(anchor, p=2, dim=1)
        # positive = F.normalize(positive, p=2, dim=1)
        # negative = F.normalize(negative, p=2, dim=1)

        loss = self.triplet_margin_loss(anchor, positive, negative)
        return loss


class HiDistanceLoss(nn.Module):
    def __init__(self, reduce='mean', sample_reduce='mean'):
        super(HiDistanceLoss, self).__init__()
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, features, binary_cat_labels, labels, margin=10.0, weight=None, split=None):
        device = features.device
        batch_size = features.shape[0]
        margin_tensor = torch.tensor(margin, device=device)
        double_margin = 2 * margin_tensor

        labels = labels.view(-1, 1)

        binary_labels = binary_cat_labels[:, 1].view(-1, 1)
        binary_mask = torch.eq(binary_labels, binary_labels.T).float().to(device)
        multi_mask = torch.eq(labels, labels.T).float().to(device)
        other_mal_mask = binary_mask - multi_mask
        ben_labels = torch.logical_not(binary_labels).float().to(device)
        same_ben_mask = torch.matmul(ben_labels, ben_labels.T)
        same_mal_fam_mask = multi_mask - same_ben_mask

        if self.reduce == 'none':
            tmp = other_mal_mask
            other_mal_mask = same_mal_fam_mask
            same_mal_fam_mask = tmp

        binary_negate_mask = torch.logical_not(binary_mask).float().to(device)
        diag_mask = torch.logical_not(torch.eye(batch_size, device=device)).float()
        binary_mask *= diag_mask
        multi_mask *= diag_mask
        other_mal_mask *= diag_mask
        same_ben_mask *= diag_mask
        same_mal_fam_mask *= diag_mask

        if split is not None:
            split_index = torch.nonzero(split, as_tuple=True)[0]
            binary_negate_mask[:, split_index] = 0
            binary_mask[:, split_index] = 0
            multi_mask[:, split_index] = 0
            other_mal_mask[:, split_index] = 0
            same_ben_mask[:, split_index] = 0
            same_mal_fam_mask[:, split_index] = 0

        x = features
        distance_matrix = (
            x.norm(dim=1, keepdim=True).pow(2) +
            x.norm(dim=1).pow(2).view(1, -1) -
            2 * x.mm(x.T)
        ).clamp(min=1e-10)

        if self.sample_reduce == 'mean' or self.sample_reduce is None:
            if weight is None:
                sum_same_ben = torch.clamp(
                    torch.sum(same_ben_mask * distance_matrix, dim=1) - same_ben_mask.sum(1) * margin_tensor, min=0)
                sum_other_mal = torch.clamp(
                    torch.sum(other_mal_mask * distance_matrix, dim=1) - other_mal_mask.sum(1) * margin_tensor, min=0)
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix, dim=1)
                sum_bin_neg = torch.clamp(
                    binary_negate_mask.sum(1) * double_margin -
                    torch.sum(binary_negate_mask * distance_matrix, dim=1), min=0)
            else:
                weight_matrix = torch.matmul(weight.view(-1, 1), weight.view(1, -1)).to(device)
                sum_same_ben = torch.clamp(
                    torch.sum(same_ben_mask * distance_matrix * weight_matrix, dim=1) -
                    same_ben_mask.sum(1) * margin_tensor, min=0)
                sum_other_mal = torch.clamp(
                    torch.sum(other_mal_mask * distance_matrix * weight_matrix, dim=1) -
                    other_mal_mask.sum(1) * margin_tensor, min=0)
                sum_same_mal_fam = torch.sum(same_mal_fam_mask * distance_matrix * weight_matrix, dim=1)
                weight_prime = 1.0 / weight
                weight_matrix_prime = torch.matmul(weight_prime.view(-1, 1), weight_prime.view(1, -1)).to(device)
                sum_bin_neg = torch.clamp(
                    binary_negate_mask.sum(1) * double_margin -
                    torch.sum(binary_negate_mask * distance_matrix * weight_matrix_prime, dim=1), min=0)
            loss = (
                sum_same_ben / same_ben_mask.sum(1).clamp(min=1) +
                sum_other_mal / other_mal_mask.sum(1).clamp(min=1) +
                sum_same_mal_fam / same_mal_fam_mask.sum(1).clamp(min=1) +
                sum_bin_neg / binary_negate_mask.sum(1).clamp(min=1)
            )
        elif self.sample_reduce == 'max':
            max_same_ben = torch.clamp(
                torch.amax(same_ben_mask * distance_matrix, 1) - margin_tensor, min=0)
            max_other_mal = torch.clamp(
                torch.amax(other_mal_mask * distance_matrix, 1) - margin_tensor, min=0)
            max_same_mal_fam = torch.amax(same_mal_fam_mask * distance_matrix, 1)
            max_bin_neg = torch.clamp(
                double_margin - torch.amin(binary_negate_mask * distance_matrix, 1), min=0)
            loss = max_same_ben + max_other_mal + max_same_mal_fam + max_bin_neg
        else:
            raise Exception(f'sample_reduce = {self.sample_reduce} not implemented yet.')

        if self.reduce == 'mean':
            loss = loss.mean()

        return loss


class HiDistanceXentLoss(nn.Module):
    def __init__(self, reduce='mean', sample_reduce='mean'):
        super(HiDistanceXentLoss, self).__init__()
        # reduce: whether use 'mean' reduction or keep sample loss
        self.reduce = reduce
        self.sample_reduce = sample_reduce

    def forward(self, xent_lambda,
                y_bin_pred, y_bin_batch,
                features, labels=None,
                margin=10.0,
                weight=None,
                split=None):
        """
        Args:
            xent_lambda: scale the binary xent loss
            y_bin_pred: predicted MLP output
            y_bin_batch: binary one-hot encoded y
            features: hidden vector of shape [bsz, feature_dim].
            labels: ground truth of shape [bsz].
            margin: margin for HiDistanceLoss.
            weight: sample weights to adjust the sample loss values
            split: whether it is in test data, so we ignore these entries
        Returns:
            A loss scalar.
        """

        Dist = HiDistanceLoss(reduce=self.reduce, sample_reduce=self.sample_reduce)
        # try not giving any weight to HiDistanceLoss
        supcon_loss = Dist(features, y_bin_batch, labels=labels, margin=margin, weight=None, split=split)


        xent_bin_loss = torch.nn.functional.binary_cross_entropy(y_bin_pred[:, 1], y_bin_batch[:, 1],
                                                                 reduction=self.reduce, weight=weight)


        if self.reduce == 'mean':
            xent_bin_loss = xent_bin_loss.mean()

        loss = supcon_loss + xent_lambda * xent_bin_loss

        # del Dist
        # torch.cuda.empty_cache()

        return loss, supcon_loss, xent_bin_loss


class MixupBinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MixupBinaryCrossEntropyLoss, self).__init__()
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, out, target):
        # Check if the target is float (indicating mixup)
        if target.dtype == torch.float32 or target.dtype == torch.float64:
            # Apply log_softmax to the output logits to get log probabilities
            log_probs = F.log_softmax(out, dim=1)

            # Compute mixup loss as a weighted sum of log-probabilities for both classes
            # target is the probability of class 1 (malware)
            loss = - (target * log_probs[:, 1] + (1 - target) * log_probs[:, 0]).mean()
        else:
            # Integer labels (no mixup), we use the standard CrossEntropyLoss
            loss = self.loss_func(out, target.long())  # Ensure target is long (int64) type

        return loss
