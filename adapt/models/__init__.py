from .svm import SVM
from .random_forest import RandomForest
from .boosting import XGBoost
from .mlp import MLP
from .hcc import HCC

MODEL_DICT = {
    'svm': SVM,
    'random-forest': RandomForest,
    'xgboost': XGBoost,
    'mlp': MLP,
    'hcc': HCC,
}
