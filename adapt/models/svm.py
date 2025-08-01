from .basemodel import BaseModel

import numpy as np
import pickle
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


class SVM(BaseModel):
    """
    Support Vector Machine model using scikit-learn.
    """
    def __init__(self, params, cfg):
        super().__init__(params, cfg)
        self.model = CalibratedClassifierCV(LinearSVC(C=params["C"], class_weight=params["class_weight"],
                                                      max_iter=1000))

    def fit(self, X, y, **kwargs):
        if not np.issubdtype(y.dtype, np.integer):
            # Convert probabilistic labels to 0 or 1 by rounding
            y = np.round(y).astype(int)
        self.model.fit(X, y)

    def save_model(self, path):
        with open(path+'.pkl', 'wb') as file:
            pickle.dump(self.model, file)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def sample_active_learning(self, X, num_samples):
        prediction_probabilities = self.predict_proba(X)
        # Get the probabilities of the predicted class
        max_probs = np.max(prediction_probabilities, axis=1)

        # Get indices of samples with the least probability of the predicted class
        least_confident_indices = np.argsort(max_probs)[:num_samples]

        return least_confident_indices

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "C": np.power(10.0, rs.uniform(-4, 3)),
            "class_weight": rs.choice([None, "balanced"]),
        }
        return params

    @classmethod
    def get_random_parameters_active_learning(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "C": np.power(10.0, rs.uniform(-4, 3)),
            "class_weight": rs.choice([None, "balanced"]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {"C": 1.0, "class_weight": None}
        return params
