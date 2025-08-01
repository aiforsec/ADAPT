from adapt.tasks.base_task import BaseTask
from adapt.datasets.dataset import MalwareDataset
from adapt.tasks.config import get_config
from adapt.utils import seed_everything
from adapt.utils.metrics import calculate_metrics
from adapt.models.de_models import get_de_model

import os
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import argparse
from sklearn.utils import shuffle


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

        self.models, self.params = self.build_model(cfg)
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

        self.app_buffer = []  # Buffer to store recent applications' confidences
        self.buffer_labels = []  # Corresponding labels (for buffer maintenance)

    def write_cfg(self):
        with open(os.path.join(self.out_dir, 'cfg.txt'), 'w') as f:
            f.write(self.cfg.dump())

    def write_params(self):
        with open(os.path.join(self.out_dir, 'model_params.pkl'), 'wb') as f:
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

    def get_random_parameters_de(self, seed):
        rs = np.random.RandomState(seed)
        params = {
            "age_threshold_low": rs.uniform(0.0, 0.5),
            "buffer_ratio": rs.uniform(0.1, 0.5),
            "buffer_size": rs.choice([1000, 2000, 3000]),
        }
        return params

    def load_parameters(self, param_file):
        with open(param_file, 'rb') as file:
            params = pickle.load(file)
        return params

    def build_model(self, cfg):
        """
        Builds the ensemble of models for DroidEvolver.
        """
        # Import or define your models here. For the purpose of this example,
        # we'll assume you have a function get_model_by_name(model_name)
        # that returns the appropriate model instance.

        model_names = ['pa1', 'ogd', 'arow', 'rda', 'ada-fobos']
        models = []

        for name in model_names:
            model = get_de_model(name)
            models.append(model)
        if not cfg.EXPERIMENT.VALIDATION_MODE and cfg.EXPERIMENT.PARAMS is not None:
            params = self.load_parameters(cfg.EXPERIMENT.PARAMS)
        else:
            params = self.get_random_parameters_de(cfg.EXPERIMENT.SEED)
        return models, params

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

    def get_shuffled_dataset(self, X, y):
        # Shuffle the dataset
        X_shuffled, y_shuffled = shuffle(X, y, random_state=self.cfg.EXPERIMENT.SEED)
        return X_shuffled, y_shuffled

    def fit_models(self, X, y):
        X, y = self.get_shuffled_dataset(X, y)
        y = np.where(y == 0, -1, y)  # Convert labels to {-1, 1}

        for model in self.models:
            model.fit(X, y)

    def train(self):
        """
        Trains the initial models on the training data.
        """
        if self.cfg.EXPERIMENT.VALIDATION_MODE or not self.cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST:
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

        self.fit_models(X_transformed, y)

        # Return non-scaler transformed version
        return X, y, families

    def compute_p_value(self, confidences, current_confidence, current_prediction, model_index):
        # Extract confidences and predictions from the buffer for the specific model
        if len(self.app_buffer) == 0:
            return 1.0  # No history, default p-value

        buffer_confidences = np.array([entry['confidences'][model_index] for entry in self.app_buffer])
        buffer_predictions = np.array([entry['predictions'][model_index] for entry in self.app_buffer])

        # Separate positive and negative samples
        if current_prediction == 1:
            positive_confs = buffer_confidences[buffer_predictions == 1]
            if len(positive_confs) == 0:
                return 1.0
            larger = np.sum(positive_confs <= current_confidence)
            p_value = larger / len(positive_confs)
        else:
            negative_confs = buffer_confidences[buffer_predictions != 1]
            if len(negative_confs) == 0:
                return 1.0
            larger = np.sum(negative_confs >= current_confidence)
            p_value = larger / len(negative_confs)

        return p_value


    def get_de_predictions(self, X):
        num_samples = len(X)
        y_preds = []
        for idx in range(num_samples):
            x_i = X[idx]
            x_i = x_i.reshape(1, -1)

            model_predictions = []
            for model in self.models:
                pred = model.predict(x_i)[0]
                model_predictions.append(pred)

            # Generate pseudo-label
            votes = sum(model_predictions)
            pseudo_label = 1 if votes > 0 else 0

            y_preds.append(pseudo_label)
        return np.array(y_preds)

    def init_app_buffer(self, X, y):
        buffer_size = self.params["buffer_size"]
        buffer_ratio = self.params["buffer_ratio"]

        num_malware = int(buffer_size * buffer_ratio)
        num_benign = buffer_size - num_malware

        # Get indices of malware and benign samples in training data
        malware_indices = np.where(y == 1)[0]
        benign_indices = np.where(y == 0)[0]

        num_malware = min(num_malware, len(malware_indices))
        num_benign = min(num_benign, len(benign_indices))

        # Randomly select samples to fill the buffer
        selected_malware_indices = np.random.choice(malware_indices, num_malware, replace=False)
        selected_benign_indices = np.random.choice(benign_indices, num_benign, replace=False)

        for _idx in selected_benign_indices:
            model_confidences = []
            model_predictions = []
            for model in self.models:
                x_i = X[_idx]
                x_i = x_i.reshape(1, -1)
                conf = model.decision_function(x_i)[0]
                pred = model.predict(x_i)[0]
                if pred == 0: pred = -1
                model_confidences.append(conf)
                model_predictions.append(pred)
            # Append to buffer
            self.app_buffer.append({
                'feature': X[_idx].reshape(1, -1),
                'confidences': model_confidences,
                'predictions': model_predictions
            })
            self.buffer_labels.append(-1)

        for _idx in selected_malware_indices:
            model_confidences = []
            model_predictions = []
            for model in self.models:
                x_i = X[_idx]
                x_i = x_i.reshape(1, -1)
                conf = model.decision_function(x_i)[0]
                pred = model.predict(x_i)[0]
                if pred == 0: pred = -1
                model_confidences.append(conf)
                model_predictions.append(pred)
            # Append to buffer
            self.app_buffer.append({
                'feature': X[_idx].reshape(1, -1),
                'confidences': model_confidences,
                'predictions': model_predictions
            })
            self.buffer_labels.append(1)
        self.buffer_labels = np.array(self.buffer_labels)

    def active_learning_iterations(self):
        # Train initial models
        X_train, y_train, families_train = self.train()

        age_threshold_low = self.params["age_threshold_low"]

        self.init_app_buffer(X_train, y_train)
        pos_index = np.where(self.buffer_labels == 1)[0]
        neg_index = np.where(self.buffer_labels == -1)[0]

        if self.cfg.EXPERIMENT.VALIDATION_MODE:
            X_val = self.val_dataset.features
            y_val = self.val_dataset.binary_labels
            timestamps = self.val_dataset.timestamps
        else:
            if self.cfg.EXPERIMENT.MERGE_VAL_DATA_FOR_TEST:
                # validation set has been merged with training, so we only perform active learning on test set
                X_val = self.test_dataset.features
                y_val = self.test_dataset.binary_labels
                timestamps = self.test_dataset.timestamps
            else:
                # validation set has not been merged with training
                # perform active learning on both validation and test set
                X1 = self.val_dataset.features
                y1 = self.val_dataset.binary_labels
                ts1 = self.val_dataset.timestamps

                X2 = self.test_dataset.features
                y2 = self.test_dataset.binary_labels
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

        monthly_metrics = {}
        f1_scores = []
        fprs = []
        fnrs = []

        for month in unique_months:
            print(f"Processing month: {month}")
            # Evaluate model on current month
            month_indices = np.where(timestamps == month)[0]
            X_month_orig = X_val[month_indices]
            y_month = y_val[month_indices]
            num_samples_month = len(y_month)

            if self.cfg.DATASET.STANDARDIZE:
                X_month = self.standard_scaler.transform(X_month_orig)
            else:
                X_month = X_month_orig

            X_month, y_month = shuffle(X_month, y_month, random_state=42)

            y_pred_month = self.get_de_predictions(X_month)

            f1, fpr, fnr = calculate_metrics(y_month, y_pred_month)

            monthly_metrics[month] = {'f1': f1, 'fpr': fpr, 'fnr': fnr}
            print(monthly_metrics[month])
            f1_scores.append(f1)
            fprs.append(fpr)
            fnrs.append(fnr)

            # Initialize lists to collect true labels and predictions
            y_trues = []
            y_preds = []
            pl_labels = []  # Pseudo labels
            gt_labels = []  # Ground truth labels for instances where models are updated

            # Initialize label cost
            label_cost = 0

            for idx in range(num_samples_month):
                x_i = X_month[idx]
                y_i = y_month[idx]
                x_i = x_i.reshape(1, -1)

                # Get predictions and confidences from each model
                model_predictions = []
                model_confidences = []
                for model in self.models:
                    conf = model.decision_function(x_i)[0]
                    pred = model.predict(x_i)[0]
                    if pred == 0: pred = -1
                    model_confidences.append(conf)
                    model_predictions.append(pred)

                if sum(model_predictions) > 0:
                    replacement_idx = random.choice(pos_index)
                else:
                    replacement_idx = random.choice(neg_index)
                self.app_buffer[replacement_idx] = {
                    'feature': x_i,
                    'confidences': model_confidences,
                    'predictions': model_predictions
                }

                # Compute p-values
                p_values = []
                for m_idx, (conf, pred) in enumerate(zip(model_confidences, model_predictions)):
                    p_value = self.compute_p_value(model_confidences, conf, pred, m_idx)
                    p_values.append(p_value)

                # Determine aged and young models
                aged_models = []
                young_models = []
                for m_idx, p_value in enumerate(p_values):
                    if p_value <= age_threshold_low:
                        aged_models.append(m_idx)
                    else:
                        young_models.append(m_idx)

                # Generate pseudo-label
                young_votes = sum([model_predictions[m_idx] for m_idx in young_models])
                aged_votes = sum([model_predictions[m_idx] for m_idx in aged_models])

                # print(young_models, aged_models, model_predictions,young_votes, aged_votes)

                if len(young_models) == 0 or young_votes == 0:
                    # Use aged models
                    pseudo_label = 1 if aged_votes > 0 else 0
                    if len(young_models) == 0:
                        label_cost += 1  # Since no young models, this is considered a labeling cost
                else:
                    # Use young models
                    pseudo_label = 1 if young_votes > 0 else 0

                y_preds.append(pseudo_label)
                y_trues.append(y_i)

                # Update aged models if necessary
                if len(aged_models) > 0 and len(young_models) >=1:
                    all_young_preds = [model_predictions[m_idx] for m_idx in young_models]
                    x_i_temp = x_i
                    y_i_temp = np.array([pseudo_label if pseudo_label == 1 else -1])
                    for m_idx in aged_models:
                        model = self.models[m_idx]
                        model.partial_fit(x_i_temp, y_i_temp)

                    # Update confidences in buffer for aged models
                    for buffer_entry in self.app_buffer:
                        x_buffer = buffer_entry['feature']
                        for m_idx in aged_models:
                            conf = self.models[m_idx].decision_function(x_buffer)[0]
                            buffer_entry['confidences'][m_idx] = conf

                    label_cost += 1
                    pl_labels.append(pseudo_label)
                    gt_labels.append(y_i)


            # print(f"Labelling rate for month {month}: {label_cost} / {num_samples_month} = {label_cost / num_samples_month}")

            # Evaluate pseudo labels accuracy if ground truth labels are available
            # if len(pl_labels) > 0:
            #     pl_accuracy = np.mean(np.array(pl_labels) == np.array(gt_labels))
            #     print(f"Pseudo-label accuracy for month {month}: {pl_accuracy:.4f}")

        avg_f1 = np.mean(f1_scores)
        avg_fpr = np.mean(fprs)
        avg_fnr = np.mean(fnrs)

        self.write_results_table(monthly_metrics, avg_f1, avg_fpr, avg_fnr)
        return monthly_metrics, avg_f1, avg_fpr, avg_fnr

    def write_results_table(self, results_dict, avg_f1, avg_fpr, avg_fnr):
        if self.cfg.EXPERIMENT.VALIDATION_MODE:
            out_file = os.path.join(self.out_dir, 'results.txt')
        else:
            out_file = os.path.join(self.out_dir, 'results_' + str(self.cfg.EXPERIMENT.SEED) + '.txt')
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
        raise NotImplementedError('Evaluation is done during active learning iterations.')

    @classmethod
    def run(cls, cfg):
        # only run once during validation mode
        if cfg.EXPERIMENT.VALIDATION_MODE:
            _task = ActiveLearning(cfg)
            _task.active_learning_iterations()
        else:
            # run for different seeds for test mode
            all_seeds_results = {}
            for seed in range(1, cfg.EXPERIMENT.NUM_TEST_RUNS + 1):
                cfg.EXPERIMENT.SEED = seed
                _task = ActiveLearning(cfg)
                monthly_metrics, avg_f1, avg_fpr, avg_fnr = _task.active_learning_iterations()
                all_seeds_results[seed] = monthly_metrics
                all_seeds_results[seed]['AVG'] = {
                    'f1': avg_f1,
                    'fpr': avg_fpr,
                    'fnr': avg_fnr
                }

                if seed == cfg.EXPERIMENT.NUM_TEST_RUNS:
                    _task.write_aggregate_test_results(all_seeds_results)

    def write_aggregate_test_results(self, all_seeds_results):
        # Initialize storage for monthly metrics
        monthly_aggregates = defaultdict(lambda: defaultdict(list))

        # Collect data for each month and each metric
        for seed, monthly_metrics in all_seeds_results.items():
            for month, metrics in monthly_metrics.items():
                if month == 'AVG':
                    continue
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
        with open(os.path.join(self.out_dir, 'results.txt'), 'w') as file:
            file.write("Time\tF1\t\tFPR\t\tFNR\n")
            for month in sorted(monthly_stats.keys()):
                stats = monthly_stats[month]
                f1 = f"{stats['f1_mean']:.4f}±{stats['f1_std']:.4f}"
                fpr = f"{stats['fpr_mean']:.4f}±{stats['fpr_std']:.4f}"
                fnr = f"{stats['fnr_mean']:.4f}±{stats['fnr_std']:.4f}"
                file.write(f"{month}\t{f1}\t{fpr}\t{fnr}\n")

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--dupes", type=str, default="remove-intra")
    parser.add_argument("--model", type=str, default='de')
    parser.add_argument("--seed", type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--merge_val", action="store_true")
    parser.add_argument("--params", type=str, default=None)
    args = parser.parse_args()

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
            _cfg.EXPERIMENT.PARAMS = f"""params/{param_dir}/{args.dataset}_{args.dupes}/{args.model}/model_params.pkl"""
    ActiveLearning.run(_cfg)


if __name__ == '__main__':
    run()