import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, f1_score, accuracy_score


def binary_f1_score(precision, recall):
    """Calculates the F1 score."""
    if precision + recall != 0:
        return 2 * precision * recall / float(precision + recall)
    else:
        return 0.0  # Return 0 if both precision and recall are 0


def calculate_metrics(y_true, y_pred):
    """Calculates various metrics given true and predicted labels."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / float(fp + tn) if (fp + tn) != 0 else 0.0  # Handle potential division by zero
    fnr = fn / float(fn + tp) if (fn + tp) != 0 else 0.0
    precision = tp / float(tp + fp) if (tp + fp) != 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) != 0 else 0.0
    f1 = binary_f1_score(precision, recall)
    return 100 * f1, 100 * fpr, 100 * fnr


def calculate_monthwise_metrics(y_true, y_pred, timestamps):
    """Calculates month-wise and average metrics."""
    unique_months = np.unique(timestamps)
    monthly_metrics = {}
    f1_scores = []
    fprs = []
    fnrs = []

    for month in unique_months:
        month_indices = np.where(timestamps == month)[0]
        y_val_month = y_true[month_indices]
        y_pred_month = y_pred[month_indices]

        f1, fpr, fnr = calculate_metrics(y_val_month, y_pred_month)

        monthly_metrics[month] = {'f1': f1, 'fpr': fpr, 'fnr': fnr}
        f1_scores.append(f1)
        fprs.append(fpr)
        fnrs.append(fnr)

    avg_f1 = np.mean(f1_scores)
    avg_fpr = np.mean(fprs)
    avg_fnr = np.mean(fnrs)

    return monthly_metrics, avg_f1, avg_fpr, avg_fnr
