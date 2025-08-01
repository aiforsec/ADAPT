import os
import torch
from torch.utils.data import Dataset
import numpy as np


class MalwareDataset(Dataset):
    def __init__(self, data_dir, split, cfg):
        """
        Args:
            data_dir (str): Path to the directory containing the dataset files
            split (str): Data split ("train", "val", "test")
        """
        super(MalwareDataset, self).__init__()

        self.data_dir = data_dir
        self.split = split
        self.transform = cfg.TRANSFORM
        self.standardize = cfg.STANDARDIZE

        # Load dataset
        self.data_path = os.path.join(self.data_dir, f"{self.split}.npz")

        # Initialize data attributes
        self.features = None
        self.binary_labels = None
        self.family_labels = None
        self.category_labels = None
        self.timestamps = None

        self.load_data(cfg.DUPLICATES)
        self.is_pytorch = cfg.IS_PYTORCH
        self.cache = cfg.CACHE
        if self.is_pytorch:
            self.process_data_for_pytorch()

    def load_data(self, duplicates):
        with np.load(self.data_path, allow_pickle=True) as data:
            features = data['feature']
            binary_labels = data['binary_label']
            family_labels = data['family_label']
            category_labels = data['category_label']
            timestamps = data['timestamp']

            intra_split_dupes = data['intra_split_dupes']
            cross_split_dupes = data['cross_split_dupes']

        if duplicates == 'keep':
            self.features = features
            self.binary_labels = binary_labels
            self.family_labels = family_labels
            self.category_labels = category_labels
            self.timestamps = timestamps
        elif duplicates == 'remove-intra':
            indices_to_keep = np.where(intra_split_dupes == -1)[0]
            self.features = features[indices_to_keep]
            self.binary_labels = binary_labels[indices_to_keep]
            self.family_labels = family_labels[indices_to_keep]
            self.category_labels = category_labels[indices_to_keep]
            self.timestamps = timestamps[indices_to_keep]
        elif duplicates == 'remove-cross':
            indices_to_keep = [idx for idx, x in enumerate(cross_split_dupes) if x[0] == 'none']
            self.features = features[indices_to_keep]
            self.binary_labels = binary_labels[indices_to_keep]
            self.family_labels = family_labels[indices_to_keep]
            self.category_labels = category_labels[indices_to_keep]
            self.timestamps = timestamps[indices_to_keep]
        else:
            raise ValueError('Unknown treatment of duplicates, must be from [keep, remove-intra, remove-cross]')

    def process_data_for_pytorch(self):
        """Converts data to PyTorch tensors and caches if enabled."""
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.binary_labels = torch.tensor(self.binary_labels, dtype=torch.long)
        self.family_labels = torch.tensor(self.family_labels, dtype=torch.long)

        if self.cache:
            self.features = self.features.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.binary_labels = self.binary_labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            self.family_labels = self.family_labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def add_data(self, new_feature, new_labels, new_family_labels):
        """Adds data to the dataset, updates the scaler (if training)

        Args:
            new_feature (np.ndarray): New feature data to add.
            new_labels (np.ndarray): Corresponding binary labels.
            new_family_labels (np.ndarray): Corresponding family labels.
        """
        # Move data to CPU and NumPy if necessary
        self.features = self.features.cpu().numpy() if isinstance(self.features, torch.Tensor) else self.features
        self.binary_labels = self.binary_labels.cpu().numpy() if isinstance(self.binary_labels, torch.Tensor) \
            else self.binary_labels
        self.family_labels = self.family_labels.cpu().numpy() if isinstance(self.family_labels, torch.Tensor) \
            else self.family_labels

        self.features = np.concatenate((self.features, new_feature))
        self.binary_labels = np.concatenate((self.binary_labels, new_labels))
        self.family_labels = np.concatenate((self.family_labels, new_family_labels))

        if self.is_pytorch:
            self.process_data_for_pytorch()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        binary_label = self.binary_labels[idx]
        family_label = self.family_labels[idx]

        if self.transform:
            feature = self.transform(feature)

        return feature, binary_label, family_label
