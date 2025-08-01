from abc import ABC, abstractmethod


class BaseTask(ABC):
    """
    Abstract base class for different malware analysis tasks
    """

    def __init__(self, cfg):
        """
        Initializes the task.

        Args:
            cfg: Configuration dictionary for the task.
        """
        self.cfg = cfg

    @abstractmethod
    def load_data(self):
        """
        Loads and preprocesses the data for the task
        """
        raise NotImplementedError

    @abstractmethod
    def build_model(self, cfg):
        """
        Builds the model for the task
        model: model name
        seed: if 0, load default hyperparameter, else a random combination with given seed
        args: additional args (such as number of feature/class)
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """
        Trains the given model on the training data.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        """
        Evaluates the trained model on the test data.
        """
        raise NotImplementedError

    @abstractmethod
    def save_model(self, model_path):
        """
        Saves the trained model to the specified path.

        Args:
            model_path (str): The path to save the model.
        """
        raise NotImplementedError

    @abstractmethod
    def load_model(self, model_path: str):
        """
        Loads a trained model from the specified path.

        Args:
            model_path (str): The path to the saved model.

        Returns:
            Any: The loaded model.
        """
        raise NotImplementedError

    @classmethod
    def run(cls, cfg):
        raise NotImplementedError
