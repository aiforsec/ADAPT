# ADAPT: A Pseudo-labeling Approach to Combat Concept Drift in Malware Detection

Implementation of the ADAPT framework.
Accepted at the
*Proceedings of the 28th International Symposium on Research in Attacks, Intrusions and Defenses (RAID 2025)*

---

## Installation

Install dependencies using the provided requirements file:

```
pip install -r requirements.txt
```

---

## Usage

### Hyperparameter Tuning

To run hyperparameter search:

```
python -m adapt.tasks.pseudo_labeling --dataset <DATASET> --model <MODEL> --seed <SEED>
```

* <MODEL>: One of 'svm', 'random-forest', 'xgboost', 'mlp'
* <DATASET>: Name of the dataset
* <SEED>: Random seed for hyperparameter search

For batch hyperparameter search across multiple datasets/models using SLURM, use:

```
python -m adapt.tasks.launcher_pl
```

This script generates a list of commands you can submit as SLURM jobs.

---

### Testing

The `params` directory contains the best hyperparameters found for each dataset and model.
To run evaluation with saved hyperparameters, simply add the `--test` flag:

```
python -m adapt.tasks.pseudo_labeling --dataset <DATASET> --model <MODEL> --seed <SEED> --test
```

---

## Notes

* For any questions or issues, please contact the authors or open an issue.
