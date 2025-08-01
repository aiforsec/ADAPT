from yacs.config import CfgNode as CN


def get_dataset_config(dataset_name):
  cfg = CN()
  cfg.NAME = dataset_name
  cfg.NUM_WORKERS = 2
  cfg.DATA_DIR = 'data/' + dataset_name
  cfg.TRAIN_FILE = 'train.npz'
  cfg.VAL_FILE = 'val.npz'
  cfg.TEST_FILE = 'test.npz'
  cfg.DUPLICATES = 'remove-cross'
  cfg.CACHE = True
  cfg.TRANSFORM = None
  cfg.IS_PYTORCH = False  # whether it will be used to train a PyTorch model or not (i.e., skelarn/XGBoost)
  cfg.VAL_BATCH_SIZE = 64  # for neural nets

  if dataset_name == 'androzoo-drebin':
    cfg.SOURCE = 'android'
    cfg.FEATURE = 'drebin'
    cfg.NUM_FEATURE = 16978
    cfg.STANDARDIZE = False
  elif dataset_name == 'androzoo-apigraph':
    cfg.SOURCE = 'android'
    cfg.FEATURE = 'apigraph'
    cfg.NUM_FEATURE = 1159
    cfg.STANDARDIZE = False
  elif dataset_name == 'bodmas':
    cfg.SOURCE = 'windows'
    cfg.FEATURE = 'ember'
    cfg.NUM_FEATURE = 2381
    cfg.STANDARDIZE = True
  elif dataset_name in ['bodmas-8', 'ember-2018', 'bodmas-family', 'packing']:
    cfg.SOURCE = 'windows'
    cfg.FEATURE = 'ember'
    cfg.NUM_FEATURE = 2381
    cfg.STANDARDIZE = True
  elif dataset_name == 'apigraph-family':
    cfg.SOURCE = 'android'
    cfg.FEATURE = 'apigraph'
    cfg.NUM_FEATURE = 1159
    cfg.STANDARDIZE = False
    cfg.NUM_CLASS = 4
  elif dataset_name == 'pdf':
    cfg.SOURCE = 'pdf'
    cfg.FEATURE = 'pdf'
    cfg.NUM_FEATURE = 950
    cfg.STANDARDIZE = True
  else:
    raise ValueError('unknown dataset {}'.format(dataset_name))

  return cfg


def get_model_config(model_name):
  cfg = CN()
  cfg.MODEL_NAME = model_name
  if model_name == 'xgboost':
    cfg.XGBOOST = CN()
    cfg.XGBOOST.USE_GPU = False
    cfg.XGBOOST.OBJECTIVE = 'binary'   # binary or classification (multi-class)
  elif model_name == 'mlp':
    cfg.MLP = CN()
    cfg.MLP.USE_GPU = True
  elif model_name == 'hcc' or model_name == 'cmlp':
    cfg.HCC = CN()
    cfg.HCC.USE_GPU = True
  elif model_name == 'scc':
    cfg.SCC = CN()
    cfg.SCC.USE_GPU = True

  return cfg


def get_experiment_cfg():
  cfg = CN()
  cfg.SEED = 1
  cfg.OUT_DIR = 'output'
  cfg.VALIDATION_MODE = True   # True during hyperparameter optimization, False during final evaluation
  cfg.MERGE_VAL_DATA_FOR_TEST = True
  cfg.NUM_HPO_RUNS = 1  # Number of runs for each hyperparameter (for different random initialization)
  cfg.NUM_TEST_RUNS = 5   # run with 5 random seeds for final hyperparameters
  cfg.TASK = 'malware-detection'
  cfg.PARAMS = None  # file containing model hyperparams, can be passed to instantiate parameter from a file
  cfg.NUM_CLASS = 2  # more task-specific params will be added later
  cfg.ACTIVE_LEARNING_SAMPLES = 50
  cfg.PSEUDO_LABELING_SOURCE_FREE = False
  return cfg


def get_default_config():
  cfg = CN()
  cfg.EXPERIMENT = get_experiment_cfg()

  cfg.EXPERIMENT.MODEL_NAME = 'xgboost'
  cfg.DATASET = get_dataset_config('androzoo-drebin')
  cfg.MODEL = get_model_config(cfg.EXPERIMENT.MODEL_NAME)
  return cfg


def get_config(dataset, model):
  cfg = CN()
  cfg.EXPERIMENT = get_experiment_cfg()
  cfg.EXPERIMENT.MODEL_NAME = model
  cfg.DATASET = get_dataset_config(dataset)
  cfg.MODEL = get_model_config(model)
  return cfg


if __name__ == '__main__':
  cf = get_dataset_config('androzoo-apigraph')
  print(cf)
