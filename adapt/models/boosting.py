from .basemodel import BaseModel

import numpy as np
import xgboost as xgb


class XGBoost(BaseModel):
    def __init__(self, params, cfg):
        super().__init__(params, cfg)

        self.params["verbosity"] = 1

        if cfg.MODEL.XGBOOST.USE_GPU:
            self.params["tree_method"] = "gpu_hist"

        if cfg.MODEL.XGBOOST.OBJECTIVE == "classification":
            self.params["objective"] = "multi:softprob"
            self.params["num_class"] = cfg.EXPERIMENT.NUM_CLASS
            self.params["eval_metric"] = "mlogloss"
        elif cfg.MODEL.XGBOOST.OBJECTIVE == "binary":
            self.params["objective"] = "binary:logistic"
            self.params["eval_metric"] = "auc"

    def _get_params_for_train(self, y):
        _params = {}
        for k, v in self.params.items():
            if k not in ["balance", "num_boost_round"]:
                _params[k] = v

        if self.params["balance"]:
            scale_pos_weight = np.sum(y == 0) / np.sum(y == 1)
        else:
            scale_pos_weight = 1

        _params["scale_pos_weight"] = scale_pos_weight
        return _params

    def fit(self, X, y, **kwargs):
        train = xgb.DMatrix(X, label=y)

        num_boost_round = self.params["num_boost_round"]

        self.model = xgb.train(
            self._get_params_for_train(y),
            train,
            num_boost_round=num_boost_round
        )

    def partial_fit(self, X, y, **kwargs):
        # Create the DMatrix from the new data
        new_data = xgb.DMatrix(X, label=y)

        num_boost_round = self.params["num_boost_round"]

        # Continue training from the current model
        self.model = xgb.train(
            self._get_params_for_train(y),
            new_data,
            num_boost_round=num_boost_round,
            xgb_model=self.model  # Continue from the current model
        )

    def save_model(self, path):
        self.model.save_model(path+'.json')

    def predict_proba(self, X):
        X = xgb.DMatrix(X)
        probabilities = self.model.predict(X)

        if self.cfg.MODEL.XGBOOST.OBJECTIVE == "binary":
            probabilities = probabilities.reshape(-1, 1)
            probabilities = np.concatenate((1 - probabilities, probabilities), 1)

        return probabilities

    def predict(self, X):
        prediction_probabilities = self.predict_proba(X)
        predictions = np.argmax(prediction_probabilities, axis=1)

        return predictions

    def sample_active_learning(self, X, num_samples):
        prediction_probabilities = self.predict_proba(X)
        # Get the probabilities of the predicted class
        max_probs = np.max(prediction_probabilities, axis=1)

        # Get indices of samples with the least probability of the predicted class
        least_confident_indices = np.argsort(max_probs)[:num_samples]

        return least_confident_indices

    def sample_randomly(self, X, num_samples):
        random_indices = np.random.choice(len(X), num_samples, replace=False)
        return random_indices

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "max_depth": int(np.round(np.power(2, rs.uniform(3, 7)))),
            "alpha": np.power(10, rs.uniform(-8, 0)),
            "lambda": np.power(10, rs.uniform(-8, 0)),
            "eta": 3.0 * np.power(10, rs.uniform(-2, -1)),
            "balance": rs.choice([True, False]),
            "num_boost_round": rs.choice([100, 150, 200, 300, 400]),
        }
        return params

    # @classmethod
    # def get_random_parameters_active_learning(cls, seed):
    #     rs = np.random.RandomState(seed)
    #     params = {
    #         "max_depth": int(np.round(np.power(2, rs.uniform(2, 4)))),
    #         # "max_depth": int(np.round(np.power(2, rs.uniform(2, 3)))),
    #         "alpha": np.power(10, rs.uniform(-8, 0)),
    #         "lambda": np.power(10, rs.uniform(-8, 0)),
    #         "eta": 3.0 * np.power(10, rs.uniform(-2, -1)),
    #         "balance": rs.choice([True, False]),
    #         "num_boost_round": rs.choice([100, 150, 200, 300, 400]),
    #     }
    #     return params

    @classmethod
    def get_random_parameters_active_learning(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "max_depth": int(np.round(np.power(2, rs.uniform(3, 7)))),
            "alpha": np.power(10, rs.uniform(-8, 0)),
            "lambda": np.power(10, rs.uniform(-8, 0)),
            "eta": 3.0 * np.power(10, rs.uniform(-2, -1)),
            "balance": rs.choice([True, False]),
            "num_boost_round": rs.choice([100, 150, 200, 300, 400]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "max_depth": 5,
            "alpha": 1e-4,
            "lambda": 1e-4,
            "eta": 0.08,
            "balance": False,
            "num_boost_round": 100
        }
        return params
