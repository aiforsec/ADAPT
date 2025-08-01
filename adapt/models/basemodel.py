from abc import ABC, abstractmethod
import pickle


class BaseModel(ABC):
    """
    Base model class providing a common interface for training and evaluation.
    """
    def __init__(self, params, cfg):
        self.params = params
        self.cfg = cfg
        self.model = None

    @abstractmethod
    def fit(self, X, y, **kwargs):
        raise NotImplementedError("Subclasses must implement the 'fit' method.")

    @abstractmethod
    def predict(self, X):
        """
        Make predictions on the given data.

        Args:
            X (array-like): Data features for prediction.

        Returns:
            array-like: Predicted labels.
        """
        raise NotImplementedError("Subclasses must implement the 'predict' method.")

    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities for the given data.

        Args:
            X (array-like): Data features for prediction.

        Returns:
            array-like: Predicted class probabilities.
        """
        raise NotImplementedError("Subclasses must implement the 'predict_proba' method if applicable.")

    @staticmethod
    def load_parameters(param_file):
        with open(param_file, 'rb') as file:
            params = pickle.load(file)
        return params

    @classmethod
    def get_random_parameters(cls, seed):
        """
        returns a random set of hyperparameters, which can be replicated using the provided seed
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    def get_random_parameters_active_learning(cls, seed):
        """
        returns a random set of hyperparameters, which can be replicated using the provided seed
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    @classmethod
    def default_parameters(cls):
        """
        returns the default set of hyperparameters
        """
        raise NotImplementedError("This method has to be implemented by the sub class")

    def save_model(self, path):
        """Saves the current state of the model.

        Saves the model using pickle. Override this method if model should be saved in a different format.

        :param path: file path to save the model (without extension)
        """
        pass

    def save_predictions(self, y_true, filename_extension=""):
        pass
