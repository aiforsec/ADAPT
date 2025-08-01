import torch.nn as nn


class SimpleEncClassifier(nn.Module):
    def __init__(self, enc_dims, mlp_dims, dropout):
        super().__init__()
        self.enc_dims = enc_dims
        self.mlp_dims = mlp_dims
        self.encoder_model = None
        self.mlp_model = None
        self.encoded = None
        self.mlp_out = None
        self.encoder_modules = []
        self.mlp_modules = []

        # encoder
        n_stacks = len(self.enc_dims) - 1
        # internal layers in encoder
        for i in range(n_stacks - 1):
            self.encoder_modules.append(nn.Linear(self.enc_dims[i], self.enc_dims[i + 1]))
            self.encoder_modules.append(nn.ReLU())
        # encoded features layer. no activation.
        self.encoder_modules.append(nn.Linear(self.enc_dims[-2], self.enc_dims[-1]))

        # encoder model
        self.encoder_model = nn.Sequential(*self.encoder_modules)

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*self.mlp_modules)

    def update_mlp_head(self, dropout=0.2):
        self.mlp_out = None
        self.mlp_modules = []

        # MLP
        m_stacks = len(self.mlp_dims) - 1
        for i in range(m_stacks - 1):
            self.mlp_modules.append(nn.Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
            self.mlp_modules.append(nn.ReLU())
            if dropout > 0:
                self.mlp_modules.append(nn.Dropout(p=dropout))
        # mlp output
        self.mlp_modules.append(nn.Linear(self.mlp_dims[-2], self.mlp_dims[-1]))
        self.mlp_modules.append(nn.Softmax(dim=1))
        self.mlp_model = nn.Sequential(*self.mlp_modules)

    def forward(self, x):
        encoded = self.encoder_model(x)
        out = self.mlp_model(encoded)
        return encoded, out

    def predict_proba(self, x):
        _, mlp_out = self.forward(x)
        return mlp_out

    def predict(self, x):
        encoded = self.encoder_model(x)
        out = self.mlp_model(encoded)
        preds = out.max(1)[1]
        return preds

    def encode(self, x):
        encoded = self.encoder_model(x)
        return encoded
