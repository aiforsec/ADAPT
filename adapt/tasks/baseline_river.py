from adapt.tasks.base_task import BaseTask
from adapt.datasets.dataset import MalwareDataset
from adapt.tasks.config import get_config
from adapt.utils import seed_everything
from adapt.utils.metrics import calculate_metrics

import os
import numpy as np
import gc
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import argparse

from river import forest
from sklearn.utils import shuffle
from river import ensemble
from river import drift
from river import tree


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
                                        cfg.DATASET.DUPLICATES, cfg.EXPERIMENT.MODEL_NAME, str(cfg.EXPERIMENT.SEED))
        else:
            self.out_dir = os.path.join(cfg.EXPERIMENT.OUT_DIR, cfg.EXPERIMENT.TASK, cfg.DATASET.NAME + "_" +
                                        cfg.DATASET.DUPLICATES, cfg.EXPERIMENT.MODEL_NAME)

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

    def get_random_parameters_arf(self, seed):
        rs = np.random.RandomState(seed)
        params = {
            "n_models": int(np.round(np.power(2, rs.uniform(3, 5)))),
            "max_features": rs.choice(["sqrt", "log2", None]),
            "max_depth": int(np.round(np.power(2, rs.uniform(5, 10)))),
            "lambda_value": 6,
        }
        return params

    def get_random_parameters_bole(self, seed):
        rs = np.random.RandomState(seed)
        params = {
            "grace_period": int(np.round(np.power(2, rs.uniform(5, 8)))),
            "max_depth": int(np.round(np.power(2, rs.uniform(5, 10)))),
            "tau": rs.uniform(0.05, 0.5),
            "n_models": int(np.round(np.power(2, rs.uniform(3, 5)))),
        }
        return params

    def get_random_parameters_pseudo_labeling(self, seed):
        rs = np.random.RandomState(seed)
        params = {
            "threshold": rs.uniform(0.6, 0.99)
        }
        return params

    def load_parameters(self, param_file):
        with open(param_file, 'rb') as file:
            params = pickle.load(file)
        return params

    def build_model(self, cfg):
        # get model params
        if not cfg.EXPERIMENT.VALIDATION_MODE and cfg.EXPERIMENT.PARAMS is not None:
            params = self.load_parameters(cfg.EXPERIMENT.PARAMS)
            model_params = params["model"]
        else:
            pl_params = self.get_random_parameters_pseudo_labeling(cfg.EXPERIMENT.SEED)
            if cfg.EXPERIMENT.MODEL_NAME == 'arf':
                model_params = self.get_random_parameters_arf(cfg.EXPERIMENT.SEED)
            elif cfg.EXPERIMENT.MODEL_NAME == 'bole':
                model_params = self.get_random_parameters_bole(cfg.EXPERIMENT.SEED)
            else:
                raise ValueError('Unknown model!')
            params = {
                "model": model_params,
                "pseudo-labeling": pl_params
            }

        # build model
        if cfg.EXPERIMENT.MODEL_NAME == 'arf':
            model = forest.ARFClassifier(n_models=model_params["n_models"], max_features=model_params["max_features"],
                                         max_depth=model_params["max_depth"], lambda_value=model_params["lambda_value"],
                                         seed=42)
        elif cfg.EXPERIMENT.MODEL_NAME == 'bole':
            model = ensemble.BOLEClassifier(
                model=drift.DriftRetrainingClassifier(
                    model=tree.HoeffdingTreeClassifier(grace_period=model_params["grace_period"],
                                                       max_depth=model_params["max_depth"], tau=model_params["tau"]),
                ),
                n_models=model_params["n_models"],
                seed=42
            )
        else:
            raise ValueError('Unknown model!')

        return model, params

    def get_shuffled_dataset(self, X, y, model=None):
        # Shuffle the dataset
        X_shuffled, y_shuffled = shuffle(X, y, random_state=42)
        return X_shuffled, y_shuffled

    def fit_river_model(self, X, y):
        X, y = self.get_shuffled_dataset(X, y)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        for x_i, y_i in zip(X, y):
            x_dict = {feature_names[i]: x_i[i] for i in range(len(x_i))}
            self.model.learn_one(x_dict, y_i)


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

        self.fit_river_model(X_transformed, y)

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

    def predict_proba_river(self, X):
        predictions = []
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        for x in X:
            x_dict = {feature_names[i]: x[i] for i in range(len(x))}

            # Get probability prediction for each sample
            y_proba_dict = self.model.predict_proba_one(x_dict)

            # Ensure the probabilities include both classes (0 and 1)
            # If no prediction for a class, assume 0 probability for that class.
            y_proba = [y_proba_dict.get(0, 0.0), y_proba_dict.get(1, 0.0)]

            # Append the probability of class 1 (malware) to the list of predictions
            predictions.append(y_proba)

        # Convert list of probabilities to a numpy array
        return np.array(predictions)

    def sample_pseudo_labeled_data(self, X):
        prediction_probabilities = self.predict_proba_river(X)[:, 1]  # Probabilities of malware class

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

    def predict_river_model(self, X):
        predictions = []
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        for x in X:
            x_dict = {feature_names[i]: x[i] for i in range(len(x))}
            # For each sample in the batch, predict the label
            y_pred = self.model.predict_one(x_dict)
            predictions.append(y_pred)
        return predictions


    def active_learning_iterations(self):
        # Train initial model
        train_x, train_y, train_families = self.train()
        #
        # if not self.cfg.EXPERIMENT.VALIDATION_MODE:
        #    self.model.save_model(os.path.join(self.out_dir, 'model_'+str(self.cfg.EXPERIMENT.SEED)+'_train'))

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
            print(f"Processing month: {month}")
            # Evaluate the model on current month
            month_indices = np.where(timestamps == month)[0]
            X_val_month_orig = X_val[month_indices]
            y_val_month = y_val[month_indices]

            if self.cfg.DATASET.STANDARDIZE:
                X_val_month = self.standard_scaler.transform(X_val_month_orig)
            else:
                X_val_month = X_val_month_orig

            y_pred_month = self.predict_river_model(X_val_month)

            f1, fpr, fnr = calculate_metrics(y_val_month, y_pred_month)

            monthly_metrics[month] = {'f1': f1, 'fpr': fpr, 'fnr': fnr}
            print(monthly_metrics[month])
            f1_scores.append(f1)
            fprs.append(fpr)
            fnrs.append(fnr)

            # Select pseudo-labeled samples from current month
            index_pseudo, y_pseudo = self.sample_pseudo_labeled_data(X_val_month)
            X_pseudo = X_val_month_orig[index_pseudo]

            self.fit_river_model(X_pseudo, y_pseudo)

            # if not self.cfg.EXPERIMENT.VALIDATION_MODE:
            #     self.model.save_model(os.path.join(self.out_dir, 'model_' + str(self.cfg.EXPERIMENT.SEED) + '_'+str(month)))

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
    parser.add_argument("--use_gt", action="store_true")
    parser.add_argument("--params", type=str, default=None)
    args = parser.parse_args()

    assert args.dupes in ['keep', 'remove-intra']
    _cfg = get_config(args.dataset, args.model)
    _cfg.DATASET.DUPLICATES = args.dupes
    _cfg.EXPERIMENT.SEED = args.seed
    _cfg.EXPERIMENT.OUT_DIR = 'output-pseudo-label'
    _cfg.EXPERIMENT.USE_GT = False
    if args.use_gt:
        _cfg.EXPERIMENT.USE_GT = True

    if args.test:
        _cfg.EXPERIMENT.OUT_DIR = 'test-output-pseudo-label'
        _cfg.EXPERIMENT.VALIDATION_MODE = False

        _cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST = False
        if args.merge_val:
            _cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST = True

        if args.params is not None:
            _cfg.EXPERIMENT.PARAMS = args.paramsws
        else:
            # load from saved best hyperparameter
            param_dir = "pseudo-labeling"
            _cfg.EXPERIMENT.PARAMS = f"""params/{param_dir}/{args.dataset}_{args.dupes}/{args.model}/model_params.pkl"""
    ActiveLearning.run(_cfg)


if __name__ == '__main__':
    run()
