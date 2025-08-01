from torch.utils.data.sampler import Sampler
import torch
import random
import numpy as np
from collections import defaultdict
from collections import Counter
from pytorch_metric_learning.utils import common_functions as c_f


class HalfSampler(Sampler):
    """
    At every iteration, this will first sample half of the batch, and then
    fill the other half of the batch with the same label distribution.
    batch_size must be an even number
    """

    def __init__(self, labels, batch_size):
        super().__init__()
        assert (batch_size % 2 == 0), "batch_size must be an even number"

        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.batch_size = int(batch_size)
        self.index_to_label = labels
        self.labels_to_indices = c_f.get_labels_to_indices(labels)

        self.all_indices = np.arange(len(labels))
        # sample half of the batch_size as self.length_of_single_pass
        self.length_of_single_pass = self.batch_size // 2
        self.batch_num = len(self.all_indices) // self.length_of_single_pass + 1
        self.list_size = len(self.all_indices) * 2

        assert self.list_size >= self.batch_size

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = [0] * self.list_size
        num_iters = self.calculate_num_iters()

        # one pass of training data with size n
        n = len(self.all_indices)
        # self.all_indices may have repeated items after upsampling
        indices = torch.randperm(n).numpy()
        perm_indices = self.all_indices[indices]
        i = 0  # index the idx_list
        k = 0  # index the perm_indices
        for bcnt in range(num_iters):
            if bcnt < num_iters - 1:
                step = self.length_of_single_pass
            else:
                step = len(self.index_to_label) % self.length_of_single_pass
            half_batch_indices = perm_indices[k: k + step]
            k += step
            idx_list[i: i + step] = half_batch_indices
            i += step
            # sample the other half with the same label distribution
            label_counts = Counter(self.index_to_label[half_batch_indices])
            for label, count in label_counts.items():
                t = self.labels_to_indices[label]
                idx_list[i: i + count] = c_f.safe_random_choice(
                    t, size=count
                )
                i += count
        return iter(idx_list)

    def calculate_num_iters(self):
        return self.batch_num


class TripletSampler(Sampler):
    def __init__(self, labels):
        super().__init__()
        self.labels = labels
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

    def __iter__(self):
        indices = []
        for label, indices_list in self.label_to_indices.items():
            for anchor_idx in indices_list:
                positive_idx = random.choice([i for i in indices_list if i != anchor_idx])
                negative_label = random.choice([l for l in self.label_to_indices.keys() if l != label])
                negative_idx = random.choice(self.label_to_indices[negative_label])
                indices.append((anchor_idx, positive_idx, negative_idx))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.labels)


class BalancedTripletSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size

        self.positive_indices = [i for i, (_, label, _) in enumerate(dataset) if label == 1]
        self.negative_indices = [i for i, (_, label, _) in enumerate(dataset) if label == 0]

        random.shuffle(self.positive_indices)
        random.shuffle(self.negative_indices)

        self.num_positive_samples = len(self.positive_indices)
        self.num_negative_samples = len(self.negative_indices)

        self.min_samples = min(self.num_positive_samples, self.num_negative_samples)
        self.num_iteration = self.min_samples // (batch_size // 2)
        self.num_samples = self.num_iteration * batch_size

    def __iter__(self):
        indices = []
        step_size = self.batch_size // 6
        for j in range(self.num_iteration):
            # samples 1/3 as anchor from both classes combined, so 1/6 from each class
            # then sample positive and negative samples for the remaining 2/3 of batch size
            i = j * step_size * 3
            # anchor
            indices += self.positive_indices[i: i + step_size]
            indices += self.negative_indices[i: i + step_size]

            # positive
            indices += self.positive_indices[i + step_size: i + 2 * step_size]
            indices += self.negative_indices[i + step_size: i + 2 * step_size]

            # negative
            indices += self.negative_indices[i + 2 * step_size: i + 3 * step_size]
            indices += self.positive_indices[i + 2 * step_size: i + 3 * step_size]

        return iter(indices)

    def __len__(self):
        return self.num_samples