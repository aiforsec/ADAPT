from deepmal.tasks.base_task import BaseTask
from deepmal.models import MODEL_DICT
from deepmal.datasets.dataset import MalwareDataset
from deepmal.tasks.config import get_config
from deepmal.utils import seed_everything
from deepmal.utils.metrics import calculate_metrics
from deepmal.datasets.misc import augment_data

import os
import numpy as np
import gc
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import argparse


class ActiveLearning(BaseTask):
    def __init__(self, cfg):
        """
        Binary malware detection task

        Args:
            cfg: Configuration dictionary for the task.
        """
        super().__init__(cfg)
        seed_everything(cfg.EXPERIMENT.SEED)

        self.train_dataset, self.val_dataset, self.test_dataset = self.load_data()
        self.standard_scaler = None

        self.model, self.params = self.build_model(cfg)
        print(self.params)
        if cfg.EXPERIMENT.VALIDATION_MODE:
            # random seed is used for hyperparam generation
            self.out_dir = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.TASK, cfg.DATASET.NAME + "_" +
                                        cfg.DATASET.DUPLICATES, 'morse', str(cfg.EXPERIMENT.SEED))
        else:
            self.out_dir = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.TASK, cfg.DATASET.NAME + "_" +
                                        cfg.DATASET.DUPLICATES, 'morse')

        self.out_dir = str(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

        self.write_cfg()
        self.write_params()

    def write_cfg(self):
        with open(os.path.join(self.out_dir, 'cfg.txt'), 'w') as f:
            f.write(self.cfg.dump())

    def write_params(self):
        with open(os.path.join(self.out_dir, 'model_params.pkl'), 'wb') as f:
            # print(self.model.params)
            pickle.dump(self.params, f)

    def load_data(self):
        """
        Loads and preprocesses the data for the task.
        """
        # load train/val/test datasets from config
        cfg = self.cfg

        train_dataset = MalwareDataset(cfg.DATASET.DATA_DIR, split='train', cfg=cfg.DATASET)
        val_dataset = MalwareDataset(cfg.DATASET.DATA_DIR, split='val', cfg=cfg.DATASET)
        test_dataset = MalwareDataset(cfg.DATASET.DATA_DIR, split='test', cfg=cfg.DATASET)
        return train_dataset, val_dataset, test_dataset

    def build_model(self, cfg):
        if not cfg.EXPERIMENT.VALIDATION_MODE and cfg.EXPERIMENT.PARAMS is not None:
            params = MODEL_DICT[cfg.EXPERIMENT.MODEL_NAME].load_parameters(cfg.EXPERIMENT.PARAMS)
            model_params = params["model"]
        elif cfg.EXPERIMENT.SEED == 0:
            model_params = MODEL_DICT[cfg.EXPERIMENT.MODEL_NAME].default_parameters()
            pl_params = self.get_default_parameters_pseudo_labeling()
            params = {
                "model": model_params,
                "pseudo-labeling": pl_params
            }
        else:
            model_params = MODEL_DICT[cfg.EXPERIMENT.MODEL_NAME].get_random_parameters_active_learning(cfg.EXPERIMENT.SEED)
            pl_params = self.get_random_parameters_pseudo_labeling(cfg.EXPERIMENT.SEED)
            params = {
                "model": model_params,
                "pseudo-labeling": pl_params
            }

        return MODEL_DICT[cfg.EXPERIMENT.MODEL_NAME](model_params, cfg), params

    def get_augmented_dataset(self, X, y, aug_rate):
        X_aug, y_aug = augment_data(X, y, aug_rate)
        return X_aug, y_aug

    def train(self):
        """
        Trains the given model on the training data.
        """
        if self.cfg.EXPERIMENT.VALIDATION_MODE or self.cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST is False:
            X = self.train_dataset.features
            y = self.train_dataset.binary_labels
            families = self.train_dataset.family_labels
        else:
            X_train = self.train_dataset.features
            y_train = self.train_dataset.binary_labels
            families_train = self.train_dataset.family_labels

            X_val = self.val_dataset.features
            y_val = self.val_dataset.binary_labels
            families_val = self.val_dataset.family_labels

            # Merge training data
            X = np.concatenate((X_train, X_val), axis=0)
            y = np.concatenate((y_train, y_val), axis=0)
            families = np.concatenate((families_train, families_val), axis=0)

        if self.cfg.DATASET.STANDARDIZE:
            self.standard_scaler = StandardScaler()
            X_transformed = self.standard_scaler.fit_transform(X)
        else:
            X_transformed = X

        self.model.fit(X_transformed, y, families=families)

        # return non-scaler transformed version
        return X, y, families

    def write_results_table(self, results_dict, avg_f1, avg_fpr, avg_fnr):
        if self.cfg.EXPERIMENT.VALIDATION_MODE:
            out_file = os.path.join(self.out_dir, 'results.txt')
        else:
            out_file = os.path.join(self.out_dir, 'results_'+str(self.cfg.EXPERIMENT.SEED)+'.txt')
        with open(out_file, 'w') as f:
            # Write header
            f.write("Time\tF1\t\tFPR\t\tFNR\n")

            # Write results for each timestamp
            for timestamp, metrics in results_dict.items():
                f.write(f"{timestamp}\t{metrics['f1']:.4f}\t{metrics['fpr']:.4f}\t{metrics['fnr']:.4f}\n")

            # Write average values
            f.write(f"AVG\t\t{avg_f1:.4f}\t{avg_fpr:.4f}\t{avg_fnr:.4f}\n")

    def evaluate(self):
        """
        Evaluates the trained model on the test data.
        """
        raise NotImplementedError('evaluation done separately for each month for pseudo-labeling')

    def save_model(self, model_path):
        """
        Saves the trained model to the specified path.

        Args:
            model_path (str): The path to save the model.
        """
        raise NotImplementedError

    def load_model(self, model_path):
        """
        Loads a trained model from the specified path.

        Args:
            model_path (str): The path to the saved model.

        Returns:
            Any: The loaded model.
        """
        raise NotImplementedError

    def write_aggregate_test_results(self, all_seeds_results):
        # Initialize storage for monthly metrics
        monthly_aggregates = defaultdict(lambda: defaultdict(list))

        # Collect data for each month and each metric
        for seed, monthly_metrics in all_seeds_results.items():
            for month, metrics in monthly_metrics.items():
                for key in ['f1', 'fpr', 'fnr']:
                    monthly_aggregates[month][key].append(metrics[key])

        # Calculate mean and std for each month
        monthly_stats = {}
        for month, metrics in monthly_aggregates.items():
            monthly_stats[month] = {
                'f1_mean': np.mean(metrics['f1']),
                'f1_std': np.std(metrics['f1']),
                'fpr_mean': np.mean(metrics['fpr']),
                'fpr_std': np.std(metrics['fpr']),
                'fnr_mean': np.mean(metrics['fnr']),
                'fnr_std': np.std(metrics['fnr']),
            }

        # Save to a text file
        with open(os.path.join(self.out_dir, 'results.txt'),  'w') as file:
            file.write("Time\tF1\t\tFPR\t\tFNR\n")
            for month in sorted(monthly_stats.keys()):
                stats = monthly_stats[month]
                f1 = f"{stats['f1_mean']:.4f}±{stats['f1_std']:.4f}"
                fpr = f"{stats['fpr_mean']:.4f}±{stats['fpr_std']:.4f}"
                fnr = f"{stats['fnr_mean']:.4f}±{stats['fnr_std']:.4f}"
                file.write(f"{month}\t{f1}\t{fpr}\t{fnr}\n")

    def get_default_parameters_pseudo_labeling(self):
        params = {
            "threshold": 0.95,
            "weak_augment": 0.1,
            "strong_augment": 0.2,
        }
        return params

    def get_random_parameters_pseudo_labeling(self, seed):
        rs = np.random.RandomState(seed)
        params = {
            "threshold": rs.uniform(0.8, 0.99),
            "weak_augment": rs.uniform(0., 0.1),
            "strong_augment": rs.uniform(0.1, 0.2),
        }
        return params

    def sample_pseudo_labeled_data(self, X):
        prediction_probabilities = self.model.predict_proba(X)[:, 1]  # Probabilities of malware class

        # Calculate thresholds based on individual class pdf
        threshold_benign = 1-self.params["pseudo-labeling"]["threshold"]
        threshold_malware = self.params["pseudo-labeling"]["threshold"]

        # Select samples based on thresholds
        benign_indices = np.where(prediction_probabilities <= threshold_benign)[0]  # High confidence benign samples
        malware_indices = np.where(prediction_probabilities >= threshold_malware)[0]  # Above threshold malware samples

        print(f"Benign Samples: {len(benign_indices)}, Malware samples: {len(malware_indices)}")

        # Combine selected indices and extract corresponding data
        selected_indices = np.concatenate((benign_indices, malware_indices))
        y_pseudo = (prediction_probabilities[selected_indices] >= 0.5).astype(int)  # Assign pseudo-labels

        return selected_indices, y_pseudo


    def active_learning_iterations(self):
        # Train initial model
        train_x, train_y, train_families = self.train()

        weak_augment_rate = self.params["pseudo-labeling"]["weak_augment"]
        strong_augment_rate = self.params["pseudo-labeling"]["strong_augment"]

        if not self.cfg.EXPERIMENT.VALIDATION_MODE:
           self.model.save_model(os.path.join(self.out_dir, 'model_'+str(self.cfg.EXPERIMENT.SEED)+'_train'))

        if self.cfg.EXPERIMENT.VALIDATION_MODE:
            X_val = self.val_dataset.features
            y_val = self.val_dataset.binary_labels
            families = self.val_dataset.family_labels
            timestamps = self.val_dataset.timestamps
        else:
            if self.cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST:
                # validation set has been merged with training, so we only perform active learning on test set
                X_val = self.test_dataset.features
                y_val = self.test_dataset.binary_labels
                families = self.test_dataset.family_labels
                timestamps = self.test_dataset.timestamps
            else:
                # validation set has not  been merged with training
                # perform active learning on both validation and test set
                X1 = self.val_dataset.features
                y1 = self.val_dataset.binary_labels
                fam1 = self.val_dataset.family_labels
                ts1 = self.val_dataset.timestamps

                X2 = self.test_dataset.features
                y2 = self.test_dataset.binary_labels
                fam2 = self.test_dataset.family_labels
                ts2 = self.test_dataset.timestamps

                # Handle empty arrays
                if X1.size == 0:
                    # If validation set is empty, use only test set
                    X_val, y_val, timestamps = X2, y2, ts2
                elif X2.size == 0:
                    # If test set is empty, use only validation set
                    X_val, y_val, timestamps = X1, y1, ts1
                else:
                    # If both are non-empty, concatenate normally
                    X_val = np.concatenate((X1, X2), axis=0)
                    y_val = np.concatenate((y1, y2), axis=0)
                    timestamps = np.concatenate((ts1, ts2), axis=0)

        unique_months = sorted(np.unique(timestamps))

        print(unique_months)

        monthly_metrics = {}
        f1_scores = []
        fprs = []
        fnrs = []

        for month in unique_months:
            print(month)
            # Evaluate the model on current month
            month_indices = np.where(timestamps == month)[0]
            X_val_month_orig = X_val[month_indices]
            y_val_month = y_val[month_indices]

            if self.cfg.DATASET.STANDARDIZE:
                X_val_month = self.standard_scaler.transform(X_val_month_orig)
            else:
                X_val_month = X_val_month_orig

            y_pred_month = self.model.predict(X_val_month)
            f1, fpr, fnr = calculate_metrics(y_val_month, y_pred_month)

            monthly_metrics[month] = {'f1': f1, 'fpr': fpr, 'fnr': fnr}
            print(monthly_metrics[month])
            f1_scores.append(f1)
            fprs.append(fpr)
            fnrs.append(fnr)

            # Select pseudo-labeled samples from current month
            # perform weak augmentation before pseudo label selection
            X_val_month_aug, _ = self.get_augmented_dataset(X_val_month, y_val_month, weak_augment_rate)
            index_pseudo, y_pseudo = self.sample_pseudo_labeled_data(X_val_month_aug)
            X_pseudo = X_val_month_orig[index_pseudo]

            # perform weak augmentation on labeled and strong augmentation on pseudo-labeled data
            train_x_aug, _ = self.get_augmented_dataset(train_x, train_y, weak_augment_rate)
            pseudo_x_aug, _ = self.get_augmented_dataset(X_pseudo, y_pseudo, strong_augment_rate)

            X_combined = np.concatenate((train_x_aug, pseudo_x_aug), axis=0)
            y_combined = np.concatenate((train_y, y_pseudo), axis=0)

            if self.cfg.DATASET.STANDARDIZE:
                self.standard_scaler = StandardScaler()
                X_combined = self.standard_scaler.fit_transform(X_combined)

            # retrain the classifier
            self.model.fit(X_combined, y_combined, families=train_families, cont_learning=True)

            del X_combined, y_combined, X_pseudo, y_pseudo
            gc.collect()

            if not self.cfg.EXPERIMENT.VALIDATION_MODE:
                self.model.save_model(os.path.join(self.out_dir, 'model_' + str(self.cfg.EXPERIMENT.SEED) + '_'+str(month)))

        avg_f1 = np.mean(f1_scores)
        avg_fpr = np.mean(fprs)
        avg_fnr = np.mean(fnrs)

        self.write_results_table(monthly_metrics, avg_f1, avg_fpr, avg_fnr)
        return monthly_metrics, avg_f1, avg_fpr, avg_fnr


    @classmethod
    def run(cls, cfg):
        # only run once during validation mode
        if cfg.EXPERIMENT.VALIDATION_MODE:
            _task = ActiveLearning(cfg)
            _task.active_learning_iterations()
        else:
            # run for different seeds for test mode
            all_seeds_results = {}
            for seed in range(1, cfg.EXPERIMENT.NUM_TEST_RUNS+1):
                cfg.EXPERIMENT.SEED = seed
                _task = ActiveLearning(cfg)

                out_file = os.path.join(_task.out_dir, 'results_' + str(_task.cfg.EXPERIMENT.SEED) + '.txt')
                if os.path.exists(out_file):
                    continue

                monthly_metrics, avg_f1, avg_fpr, avg_fnr  = _task.active_learning_iterations()
                all_seeds_results[seed] = monthly_metrics
                all_seeds_results[seed]['AVG'] = {
                    'f1': avg_f1,
                    'fpr': avg_fpr,
                    'fnr': avg_fnr
                }

                if seed == cfg.EXPERIMENT.NUM_TEST_RUNS:
                    _task.write_aggregate_test_results(all_seeds_results)


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dupes", type=str, default="remove-intra")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--merge_val", action="store_true")
    parser.add_argument("--params", type=str, default=None)
    args = parser.parse_args()

    assert args.dupes in ['keep', 'remove-intra']
    _cfg = get_config(args.dataset, args.model)
    _cfg.DATASET.DUPLICATES = args.dupes
    _cfg.EXPERIMENT.SEED = args.seed
    _cfg.EXPERIMENT.OUT_DIR = 'output-pseudo-label'


    if args.test:
        _cfg.EXPERIMENT.OUT_DIR = 'test-output-pseudo-label'
        _cfg.EXPERIMENT.VALIDATION_MODE = False

        _cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST = False
        if args.merge_val:
            _cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST = True

        if args.params is not None:
            _cfg.EXPERIMENT.PARAMS = args.params
        else:
            # load from saved best hyperparameter
            param_dir = "pseudo-labeling"
            _cfg.EXPERIMENT.PARAMS = f"""params/{param_dir}/{args.dataset}_{args.dupes}/morse/model_params.pkl"""
    ActiveLearning.run(_cfg)


if __name__ == '__main__':
    run()
