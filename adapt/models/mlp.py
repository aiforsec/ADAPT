from .basemodel import BaseModel
from adapt.utils.losses import MixupBinaryCrossEntropyLoss

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


class MLP(BaseModel):
    def __init__(self, params, cfg):
        super().__init__(params, cfg)

        mlp_dims = [cfg.DATASET.NUM_FEATURE] + params["mlp_layers"] + [cfg.EXPERIMENT.NUM_CLASS]
        self.device = self.get_device()
        self.model = self.get_mlp_pytorch(mlp_dims, params["dropout"]).to(self.device)

    @staticmethod
    def get_mlp_pytorch(mlp_dims, dropout):
        mlp_modules = []
        m_stacks = len(mlp_dims) - 1
        for i in range(m_stacks - 1):
            mlp_modules.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]))
            mlp_modules.append(nn.ReLU())
            if dropout > 0:
                mlp_modules.append(nn.Dropout(p=dropout))

        # mlp output
        mlp_modules.append(nn.Linear(mlp_dims[-2], mlp_dims[-1]))
        return nn.Sequential(*mlp_modules)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    def get_device(self):
        if self.cfg.MODEL.MLP.USE_GPU is True and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return torch.device(device)

    def fit(self, X, y, **kwargs):
        if 'cont_learning' in kwargs and kwargs['cont_learning'] is True and "cont_learning_epochs" in self.params:
            _epochs = int(self.params["cont_learning_epochs"] * self.params["epochs"])
        else:
            _epochs = self.params["epochs"]
        _lr = self.params["learning_rate"]
        if self.params["optimizer"] == "Adam":
            optimizer = optim.Adam(self.model.parameters(), lr=_lr)
        else:
            raise ValueError("MLP optimizer {} not supported".format(self.params["optimizer"]))

        X = torch.tensor(X).float()
        y = torch.tensor(y)
        loss_func = MixupBinaryCrossEntropyLoss()

        train_dataset = TensorDataset(X, y)

        if self.params["balance"]:
            # Calculate class weights (inverse frequency of each class)
            rounded_y = torch.round(y).to(torch.int64)
            class_counts = torch.bincount(rounded_y)
            class_weights = 1. / class_counts
            sample_weights = class_weights[rounded_y]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.params["batch_size"],
                sampler=sampler,
                num_workers=self.cfg.DATASET.NUM_WORKERS,
            )
        else:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.params["batch_size"],
                shuffle=True,
                num_workers=self.cfg.DATASET.NUM_WORKERS,
            )

        loss_history = []

        for epoch in range(_epochs):
            epoch_loss = 0.0
            num_batches = 0

            for i, (batch_X, batch_y) in enumerate(train_loader):
                out = self.model(batch_X.to(self.device))
                loss = loss_func(out, batch_y.to(self.device))

                epoch_loss += loss.item()
                num_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_loss = epoch_loss / num_batches
            loss_history.append(average_loss)
            # print('Epoch: {}, Average Loss: {}'.format(epoch, epoch_loss))

    def partial_fit(self, X, y, **kwargs):
        self.fit(X, y, **kwargs)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path+'.pth')

    def predict(self, X):
        prediction_probabilities = self.predict_proba(X)
        predictions = np.argmax(prediction_probabilities, axis=1)
        return predictions

    def predict_proba(self, X):
        self.model.eval()

        X = torch.tensor(X).float()
        test_dataset = TensorDataset(X)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg.DATASET.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.cfg.DATASET.NUM_WORKERS,
        )
        predictions = []
        with torch.no_grad():
            for batch_X in test_loader:
                preds = self.model(batch_X[0].to(self.device))
                preds = F.softmax(preds, dim=1)
                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

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

        layer_structures = [
            [100, 100],
            [512, 256, 128],
            [512, 384, 256, 128],
            [512, 384, 256, 128, 64]
        ]

        params = {
            "mlp_layers": layer_structures[rs.choice(len(layer_structures))],
            "learning_rate": np.power(10, rs.uniform(-5, -3)),
            "dropout": rs.uniform(0., 0.5),
            "batch_size": int(np.power(2, rs.choice([5, 6, 7, 8, 9, 10]))),
            "epochs": rs.choice([25, 30, 35, 40, 50, 60, 80, 100, 150]),
            "optimizer": rs.choice(["Adam"]),
            "balance": rs.choice([True, False]),
        }
        return params

    @classmethod
    def get_random_parameters_active_learning(cls, seed):
        rs = np.random.RandomState(seed)

        layer_structures = [
            [100, 100],
            [512, 256, 128],
            [512, 384, 256, 128],
            [512, 384, 256, 128, 64]
        ]

        params = {
            "mlp_layers": layer_structures[rs.choice(len(layer_structures))],
            "learning_rate": np.power(10, rs.uniform(-5, -3)),
            "dropout": rs.uniform(0., 0.5),
            "batch_size": int(np.power(2, rs.choice([5, 6, 7, 8, 9, 10]))),
            "epochs": rs.choice([25, 30, 35, 40, 50, 60, 80, 100, 150]),
            "optimizer": rs.choice(["Adam"]),
            "balance": rs.choice([True, False]),
            "cont_learning_epochs": rs.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        }
        return params

    @classmethod
    def default_parameters(cls):
        params = {
            "mlp_layers": [100, 100],
            "learning_rate": 0.0001,
            "dropout": 0.2,
            "batch_size": 64,
            "epochs": 20,
            "optimizer": "Adam",
            "Balance": True
        }
        return params
