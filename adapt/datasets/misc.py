import numpy as np


def augment_data(X, y, mask_ratio, model=None):
    """
    Augment the dataset by masking a fraction of features for each sample and replacing
    them with values sampled from the empirical distribution of that feature for the same class.

    If a model is provided, only augmented samples that have the same prediction as the original
    samples (according to the model's predict function) will be returned.

    Parameters:
    - X: numpy array of shape (num_samples, num_features), the feature matrix.
    - y: numpy array of shape (num_samples,), the labels (0 for benign, 1 for malware).
    - mask_ratio: float, the fraction of features to mask and replace for each sample.
    - model: A model with a 'predict' method (optional).

    Returns:
    - X_augmented: numpy array of the same shape as X, the augmented feature matrix.
    """

    num_samples, num_features = X.shape
    X_augmented = X.copy()

    # Separate the data by class
    class_0_samples = X[y == 0]
    class_1_samples = X[y == 1]

    # Iterate over each sample
    for i in range(num_samples):
        # Determine the number of features to mask
        num_masked_features = int(mask_ratio * num_features)

        # Randomly choose features to mask
        mask_indices = np.random.choice(num_features, size=num_masked_features, replace=False)

        if y[i] == 0:
            # If the sample is benign (class 0), replace masked features by sampling from class 0 samples
            for idx in mask_indices:
                X_augmented[i, idx] = np.random.choice(class_0_samples[:, idx])
        else:
            # If the sample is malware (class 1), replace masked features by sampling from class 1 samples
            for idx in mask_indices:
                X_augmented[i, idx] = np.random.choice(class_1_samples[:, idx])

    # If a model is provided, filter the augmented samples
    if model is not None:
        y_pred_augmented = model.predict(X_augmented)

        # Only keep samples where original and augmented predictions match
        matching_indices = np.where(y == y_pred_augmented)[0]

        # Filter X_augmented to only include these matching samples
        X_augmented = X_augmented[matching_indices]
        y = y[matching_indices]

    return X_augmented, y


def mixup_data(X, y, alpha=0.2):
    """Apply mixup to the dataset, generating a separate lambda for each sample."""
    batch_size = X.shape[0]

    # Generate a separate lambda for each sample in the batch
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=batch_size)  # Lambda for each sample
    else:
        lam = np.ones(batch_size)  # If alpha is 0, no mixup (lam=1 for all)

    # Randomly shuffle the batch
    index = np.random.permutation(batch_size)

    # Reshape lam to be (batch_size, 1) so it broadcasts correctly across features
    lam_x = lam[:, np.newaxis]

    # Mix the inputs
    mixed_X = lam_x * X + (1 - lam_x) * X[index, :]

    # Mix the targets (assuming y is 1D; adjust if y is 2D or one-hot encoded)
    mixed_y = lam * y + (1 - lam) * y[index]

    return mixed_X, mixed_y