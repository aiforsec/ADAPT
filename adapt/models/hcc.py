from adapt.models.basemodel import BaseModel
from adapt.utils.losses import HiDistanceXentLoss
from adapt.datasets.samplers import HalfSampler
from adapt.utils.utils import AverageMeter
from adapt.models.nn_architectures import SimpleEncClassifier

from collections import Counter
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class HCC(BaseModel):
    def __init__(self, params, cfg):
        super().__init__(params, cfg)
        encoder_dims = [cfg.DATASET.NUM_FEATURE] + params["encoder_layers"]
        mlp_dims = [params["encoder_layers"][-1]] + params["mlp_layers"] + [cfg.EXPERIMENT.NUM_CLASS]
        self.device = self.get_device()
        self.model = SimpleEncClassifier(encoder_dims, mlp_dims, params["dropout"]).to(self.device)

    def get_model_size(self):
        model_size = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        return model_size

    def get_device(self):
        if self.cfg.MODEL.HCC.USE_GPU is True and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        return torch.device(device)

    def remove_singleton_families(self, X, y, y_fam):
        counted_y_train = Counter(y_fam)
        singleton_families = [family for family, count in counted_y_train.items() if count == 1]

        X_updated = np.array([X[i] for i, family in enumerate(y_fam) if family not in singleton_families])
        y_updated = np.array([y[i] for i, family in enumerate(y_fam) if family not in singleton_families])
        y_fam_updated = np.array([y_fam[i] for i, family in enumerate(y_fam) if family not in singleton_families])

        return X_updated, y_updated, y_fam_updated


    def fit(self, X, y, **kwargs):
        if 'cont_learning' in kwargs and kwargs['cont_learning'] is True:
            _epochs = int(self.params["cont_learning_epochs"])
            _lr = self.params["learning_rate"] * self.params["cont_learning_lr"]
        else:
            _lr = self.params["learning_rate"]
            _epochs = self.params["epochs"]

        if ('cont_learning' in kwargs and kwargs['cont_learning'] is True) or self.params["optimizer"] == "Adam":
            # only use Adam during active learning
            optimizer = optim.Adam(self.model.parameters(), lr=_lr)
        elif self.params["optimizer"] == "SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=_lr)
        else:
            raise ValueError("Contrastive MLP optimizer {} not supported".format(self.params["optimizer"]))

        y_fam = kwargs['families']
        # if 'cont_learning' not in kwargs or ['cont_learning'] is False:
        #     X, y, y_fam = self.remove_singleton_families(X, y, y_fam)

        X = torch.tensor(X).float()
        y = torch.tensor(y)

        loss_func = HiDistanceXentLoss().to(self.device)

        if self.params["sampler"] == "half":
            sampler = HalfSampler(y_fam, self.params['batch_size'])
        else:
            raise ValueError("Contrastive MLP sampler {} not supported".format(self.params["sampler"]))

        y_fam = torch.tensor(y_fam)
        train_dataset = TensorDataset(X, y, y_fam)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.params["batch_size"],
            sampler=sampler,
            num_workers = self.cfg.DATASET.NUM_WORKERS,
        )

        # might add more scheduler in the future
        scheduler_step_size = self.params["scheduler_step"]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size,
                                                    gamma=self.params["scheduler_gamma"])

        loss_history = []

        for epoch in range(_epochs):
            epoch_loss = AverageMeter()
            epoch_supcon_loss = AverageMeter()
            epoch_xent_loss = AverageMeter()

            for i, (batch_X, batch_y, batch_y_fam) in enumerate(train_loader):
                features, y_pred = self.model(batch_X.to(self.device))

                batch_y = torch.nn.functional.one_hot(batch_y, num_classes=2).float().to(self.device)
                batch_y_fam = batch_y_fam.to(self.device)

                loss, supcon_loss, xent_loss = loss_func(self.params["xent_lambda"],
                                                         y_pred,
                                                         batch_y,
                                                         features,
                                                         labels=batch_y_fam,
                                                         margin=self.params["margin"]
                                                         )

                # print(loss, supcon_loss, xent_loss)

                epoch_loss.update(loss.item(), batch_y.shape[0])
                epoch_supcon_loss.update(supcon_loss.item(), batch_y.shape[0])
                epoch_xent_loss.update(xent_loss.item(), batch_y.shape[0])

                optimizer.zero_grad()
                loss.backward()
                # Clip gradients to prevent very large update due to supcon_loss
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()

            loss_history.append(epoch_loss.avg)
            # print('Epoch: {}, Average Loss: {}, Supcon Loss: {}, Xent Loss: {}'.
            #       format(epoch, epoch_loss.avg, epoch_supcon_loss.avg, epoch_xent_loss.avg))
            scheduler.step()

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
                _, preds = self.model(batch_X[0].to(self.device))
                predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions)

    def select_samples(self, X_train, y_train_binary, X_test, y_test_pred, num_samples):
        X_train_tensor = torch.from_numpy(X_train).float().cuda()
        z_train = self.model.encode(X_train_tensor)
        z_train = torch.nn.functional.normalize(z_train)
        z_train = z_train.cpu().detach().numpy()

        X_test_tensor = torch.from_numpy(X_test).float().cuda()
        z_test = self.model.encode(X_test_tensor)
        z_test = torch.nn.functional.normalize(z_test)
        z_test = z_test.cpu().detach().numpy()

        sample_indices = []
        sample_scores = []

        # build the KDTree
        tree = KDTree(z_train)
        # query all z_test up to a margin
        all_neighbors = tree.query(z_test, k=z_train.shape[0], workers=8)
        all_distances, all_indices = all_neighbors

        bsize = self.params["batch_size"]

        # nn_loss = np.zeros([sample_num])
        sample_num = z_test.shape[0]
        for i in range(sample_num):
            test_sample = X_test_tensor[i:i + 1]  # on GPU
            # bsize-1 nearest neighbors of the test sample i
            batch_indices = all_indices[i][:bsize - 1]
            # x_batch
            x_train_batch = X_train_tensor[batch_indices]  # on GPU
            x_batch = torch.cat((test_sample, x_train_batch), 0)
            # y_batch
            y_train_batch = y_train_binary[batch_indices]
            y_batch_np = np.hstack((y_test_pred[i], y_train_batch))
            y_batch = torch.from_numpy(y_batch_np).cuda()
            # y_bin_batch
            y_bin_batch = torch.nn.functional.one_hot(y_batch, num_classes=2).float().to(self.device)
            # y_bin_batch = torch.from_numpy(to_categorical(y_batch_np, num_classes=2)).float().cuda()
            # we don't need split_tensor. all samples are training samples
            # split_tensor = torch.zeros(x_batch.shape[0]).int().cuda()
            # split_tensor[test_offset:] = 1


            # in the loss function, y_bin_batch is the categorical version
            # call the loss function once for every test sample

            features, y_pred = self.model(x_batch)
            HiDistanceXent = HiDistanceXentLoss().cuda()
            loss, _, _ = HiDistanceXent(self.params["xent_lambda"],
                                        y_pred, y_bin_batch,
                                        features, labels=y_batch,
                                        margin=self.params["margin"])
            loss = loss.to('cpu').detach().item()

            sample_scores.append(loss)

        sorted_sample_scores = list(sorted(list(enumerate(sample_scores)), key=lambda item: item[1], reverse=True))
        sample_cnt = 0
        for idx, score in sorted_sample_scores:
            sample_indices.append(idx)
            sample_cnt += 1
            if sample_cnt == num_samples:
                break
        return sample_indices, sample_scores

    def sample_active_learning(self, X, num_samples, **kwargs):
        # X_train, y_train_binary,  X_test, y_test_pred, total_count
        X_train = kwargs["X_train"]
        y_train_binary = kwargs["y_train_binary"]
        y_test_pred = kwargs["y_test_pred"]
        sample_indices, _ = self.select_samples(X_train, y_train_binary, X, y_test_pred, num_samples)
        return np.array(sample_indices)

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)

        encoder_layer_structures = [
            [512, 256, 128],
            [512, 384, 256, 128],
        ]

        mlp_layer_structures = [
            [100],
            [100, 100],
        ]

        params = {
            "encoder_layers": encoder_layer_structures[rs.choice(len(encoder_layer_structures))],
            "mlp_layers": mlp_layer_structures[rs.choice(len(mlp_layer_structures))],
            "learning_rate": rs.choice([0.001, 0.003, 0.005, 0.007]),
            "dropout": rs.uniform(0., 0.25),
            "batch_size": int(np.power(2, rs.choice([10]))),
            "epochs": rs.choice([100, 150, 200, 250]),
            "xent_lambda": rs.choice([100]),
            "margin": rs.choice([10]),
            "optimizer": rs.choice(["Adam", "SGD"]),
            "sampler": rs.choice(["half"]),
            "scheduler_step": rs.choice([10]),
            "scheduler_gamma": rs.choice([0.5, 0.95]),
        }
        return params

    @classmethod
    def get_random_parameters_active_learning(cls, seed):
        rs = np.random.RandomState(seed)

        encoder_layer_structures = [
            [512, 256, 128],
            [512, 384, 256, 128],
        ]

        mlp_layer_structures = [
            [100],
            [100, 100],
        ]

        params = {
            "encoder_layers": encoder_layer_structures[rs.choice(len(encoder_layer_structures))],
            "mlp_layers": mlp_layer_structures[rs.choice(len(mlp_layer_structures))],
            "learning_rate": rs.choice([0.001, 0.003, 0.005, 0.007]),
            "dropout": rs.uniform(0., 0.25),
            "batch_size": int(np.power(2, rs.choice([10]))),
            "epochs": rs.choice([100, 150, 200, 250]),
            "xent_lambda": rs.choice([100]),
            "margin": rs.choice([10]),
            "optimizer": rs.choice(["Adam", "SGD"]),
            "sampler": rs.choice(["half"]),
            "scheduler_step": rs.choice([10]),
            "scheduler_gamma": rs.choice([0.5, 0.95]),
            "cont_learning_lr": rs.choice([0.01, 0.05]),
            "cont_learning_epochs": rs.choice([50, 100]),
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

# if __name__ == '__main__':
#     encoder_dims = [1000] + [512, 384, 256, 128]
#     mlp_dims = [128] + [100, 100] + [2]
#     model = SimpleEncClassifier(encoder_dims, mlp_dims, 0.2)
#     x = torch.rand(4, 1000)
#     _, y = model(x)
#     print(y.shape)
