import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class PAClassifier(BaseEstimator, ClassifierMixin):
    """
    Passive Aggressive classifier.

    Modes: 'standard', 'pa1', 'pa2'
    """

    def __init__(self, C=1.0, mode='pa1'):
        self.C = C  # Only used in pa1 and pa2 modes
        self.mode = mode.lower()  # 'standard', 'pa1', 'pa2'
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0.0

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Compute margin
            m = y_i * (np.dot(self.w, x_i) + self.b)

            # Compute loss
            loss = max(0, 1 - m)

            # Compute eta (tau)
            x_norm_squared = np.dot(x_i, x_i)
            if x_norm_squared == 0:
                tau = 0
            else:
                if self.mode == 'standard':
                    tau = loss / x_norm_squared
                elif self.mode == 'pa1':
                    tau = min(self.C, loss / x_norm_squared)
                elif self.mode == 'pa2':
                    tau = loss / (x_norm_squared + 1 / (2 * self.C))
                else:
                    raise ValueError("Invalid mode for PAClassifier: {}".format(self.mode))

            # Update weights and bias
            self.w += tau * y_i * x_i
            self.b += tau * y_i

        return self

    def partial_fit(self, X, y, classes=None):
        if self.w is None:
            n_features = X.shape[1]
            self.w = np.zeros(n_features)
            self.b = 0.0

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        n_samples = X.shape[0]
        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Compute margin
            m = y_i * (np.dot(self.w, x_i) + self.b)

            # Compute loss
            loss = max(0, 1 - m)

            # Compute eta (tau)
            x_norm_squared = np.dot(x_i, x_i)
            if x_norm_squared == 0:
                tau = 0
            else:
                if self.mode == 'standard':
                    tau = loss / x_norm_squared
                elif self.mode == 'pa1':
                    tau = min(self.C, loss / x_norm_squared)
                elif self.mode == 'pa2':
                    tau = loss / (x_norm_squared + 1 / (2 * self.C))
                else:
                    raise ValueError("Invalid mode for PAClassifier: {}".format(self.mode))

            # Update weights and bias
            self.w += tau * y_i * x_i
            self.b += tau * y_i

        return self

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

class OGDClassifier(BaseEstimator, ClassifierMixin):
    """
    Online Gradient Descent classifier.
    """

    def __init__(self, eta0=1.0, power_t=0.5):
        self.eta0 = eta0
        self.power_t = power_t
        self.w = None
        self.b = None
        self.iteration = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        self.iteration = 0

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        for i in range(n_samples):
            self.iteration += 1
            x_i = X[i]
            y_i = y[i]

            # Compute learning rate
            eta = self.eta0 / (self.iteration ** self.power_t)

            # Compute margin
            m = y_i * (np.dot(self.w, x_i) + self.b)

            # Compute gradient
            if m >= 1:
                grad_w = 0
                grad_b = 0
            else:
                grad_w = - y_i * x_i
                grad_b = - y_i

            # Update weights and bias
            self.w -= eta * grad_w
            self.b -= eta * grad_b

        return self

    def partial_fit(self, X, y, classes=None):
        if self.w is None:
            n_features = X.shape[1]
            self.w = np.zeros(n_features)
            self.b = 0.0
            self.iteration = 0

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        n_samples = X.shape[0]

        for i in range(n_samples):
            self.iteration += 1
            x_i = X[i]
            y_i = y[i]

            # Compute learning rate
            eta = self.eta0 / (self.iteration ** self.power_t)

            # Compute margin
            m = y_i * (np.dot(self.w, x_i) + self.b)

            # Compute gradient
            if m >= 1:
                grad_w = 0
                grad_b = 0
            else:
                grad_w = - y_i * x_i
                grad_b = - y_i

            # Update weights and bias
            self.w -= eta * grad_w
            self.b -= eta * grad_b

        return self

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

class AROWClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Regularization of Weight Vectors (AROW) classifier.
    """

    def __init__(self, r=1.0):
        self.r = r  # Regularization parameter r_
        self.W = None  # Weight vector w
        self.b = None  # Bias term
        self.Sigma = None  # Covariance matrix Sigma

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize weight vector W and covariance matrix Sigma
        self.W = np.zeros(n_features)
        self.b = 0.0
        self.Sigma = np.ones(n_features)  # Equivalent to Vector<real_t> of ones in C++

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Compute margin m = y_i * (w^T x_i + b)
            m = y_i * (np.dot(self.W, x_i) + self.b)

            # Compute loss l = max(0,1 - m)
            l = max(0, 1 - m)

            # Calculate variance v = x_i^T diag(Sigma) x_i
            v = np.sum(self.Sigma * x_i ** 2)

            # beta_t = 1 / (v + r)
            beta_t = 1.0 / (v + self.r)

            # alpha_t = l * beta_t
            alpha_t = l * beta_t

            # Update weight vector W
            # W = W + alpha_t * y_i * Sigma * x_i
            self.W += alpha_t * y_i * self.Sigma * x_i

            # Update bias term b
            self.b += alpha_t * y_i

            # Update covariance matrix Sigma
            # Sigma = Sigma - beta_t * (Sigma * x_i)^2
            sigma_x = self.Sigma * x_i
            self.Sigma -= beta_t * (sigma_x ** 2)

            # Ensure that Sigma remains positive
            self.Sigma = np.maximum(self.Sigma, 1e-10)

        return self

    def partial_fit(self, X, y, classes=None):
        """
        Allows incremental updates to the model.
        """
        if self.W is None:
            n_features = X.shape[1]
            self.W = np.zeros(n_features)
            self.b = 0.0
            self.Sigma = np.ones(n_features)

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        n_samples = X.shape[0]
        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Compute margin m = y_i * (w^T x_i + b)
            m = y_i * (np.dot(self.W, x_i) + self.b)

            # Compute loss l = max(0,1 - m)
            l = max(0, 1 - m)

            # Calculate variance v = x_i^T diag(Sigma) x_i
            v = np.sum(self.Sigma * x_i ** 2)

            # beta_t = 1 / (v + self.r)
            beta_t = 1.0 / (v + self.r)

            # alpha_t = l * beta_t
            alpha_t = l * beta_t

            # Update weight vector W
            self.W += alpha_t * y_i * self.Sigma * x_i

            # Update bias term b
            self.b += alpha_t * y_i

            # Update covariance matrix Sigma
            sigma_x = self.Sigma * x_i
            self.Sigma -= beta_t * (sigma_x ** 2)

            # Ensure that Sigma remains positive
            self.Sigma = np.maximum(self.Sigma, 1e-10)

        return self

    def decision_function(self, X):
        return np.dot(X, self.W) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

class RDAClassifier(BaseEstimator, ClassifierMixin):
    """
    Regularized Dual Averaging (RDA) classifier.
    Implements L2 regularization (RDA-L2).
    """

    def __init__(self, lambda_param=1.0):
        """
        Parameters:
        - lambda_param: Regularization parameter (sigma in SOL code)
        """
        self.lambda_param = lambda_param
        self.w = None  # Weight vector
        self.u = None  # Accumulated gradient
        self.t = 0     # Time step

    def fit(self, X, y):
        """
        Fit the model to the data.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.u = np.zeros(n_features)
        self.t = 0

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        for i in range(n_samples):
            self.t += 1
            x_i = X[i]
            y_i = y[i]

            # Compute loss gradient
            m = y_i * np.dot(self.w, x_i)
            if m >= 1:
                grad = 0
            else:
                grad = - y_i * x_i

            # Accumulate gradient
            self.u += grad

            # Compute learning rate eta_t
            eta_t = 1.0 / (self.t * self.lambda_param)

            # Update weight vector
            self.w = - eta_t * self.u

        return self

    def partial_fit(self, X, y, classes=None):
        if self.w is None:
            n_features = X.shape[1]
            self.w = np.zeros(n_features)
            self.u = np.zeros(n_features)
            self.t = 0

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        n_samples = X.shape[0]
        for i in range(n_samples):
            self.t += 1
            x_i = X[i]
            y_i = y[i]

            # Compute loss gradient
            m = y_i * np.dot(self.w, x_i)
            if m >= 1:
                grad = 0
            else:
                grad = - y_i * x_i

            # Accumulate gradient
            self.u += grad

            # Compute learning rate eta_t
            eta_t = 1.0 / (self.t * self.lambda_param)

            # Update weight vector
            self.w = - eta_t * self.u

        return self

    def decision_function(self, X):
        return np.dot(X, self.w)

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

class AdaFOBOSClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Subgradient FOBOS classifier.
    """

    def __init__(self, eta=1.0, delta=1e-5):
        self.eta = eta
        self.delta = delta
        self.w = None
        self.b = None
        self.H = None  # Accumulated H per feature
        self.H_b = None  # For bias term

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.H = np.full(n_features, self.delta)
        self.H_b = self.delta

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Compute margin
            m = y_i * (np.dot(self.w, x_i) + self.b)

            # Compute gradient
            if m >= 1:
                grad_w = np.zeros_like(self.w)
                grad_b = 0
            else:
                grad_w = - y_i * x_i
                grad_b = - y_i

            # Update H
            self.H = np.sqrt((self.H - self.delta) ** 2 + grad_w ** 2) + self.delta
            self.H_b = np.sqrt((self.H_b - self.delta) ** 2 + grad_b ** 2) + self.delta

            # Update weights
            self.w -= (self.eta * grad_w) / self.H

            # Update bias
            self.b -= (self.eta * grad_b) / self.H_b

        return self

    def partial_fit(self, X, y, classes=None):
        if self.w is None:
            n_features = X.shape[1]
            self.w = np.zeros(n_features)
            self.b = 0.0
            self.H = np.full(n_features, self.delta)
            self.H_b = self.delta

        # Convert y from {0,1} to {-1,1}
        y = np.where(y == 0, -1, y)

        n_samples = X.shape[0]

        for i in range(n_samples):
            x_i = X[i]
            y_i = y[i]

            # Compute margin
            m = y_i * (np.dot(self.w, x_i) + self.b)

            # Compute gradient
            if m >= 1:
                grad_w = np.zeros_like(self.w)
                grad_b = 0
            else:
                grad_w = - y_i * x_i
                grad_b = - y_i

            # Update H
            self.H = np.sqrt((self.H - self.delta) ** 2 + grad_w ** 2) + self.delta
            self.H_b = np.sqrt((self.H_b - self.delta) ** 2 + grad_b ** 2) + self.delta

            # Update weights
            self.w -= (self.eta * grad_w) / self.H

            # Update bias
            self.b -= (self.eta * grad_b) / self.H_b

        return self

    def decision_function(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

def get_de_model(model_name):
    """
    Returns a model object given the model name.

    Parameters:
    - model_name: str, one of "pa", "pa1", "pa2", "ogd", "arow", "rda", "ada-fobos"

    Returns:
    - model: An instance with fit/predict methods
    """
    model_name = model_name.lower()
    if model_name == 'pa' or model_name == 'standard':
        # Passive-Aggressive classifier (standard)
        return PAClassifier(mode='standard')
    elif model_name == 'pa1':
        # PA-I classifier
        return PAClassifier(mode='pa1')
    elif model_name == 'pa2':
        # PA-II classifier
        return PAClassifier(mode='pa2')
    elif model_name == 'ogd':
        # OGD classifier
        return OGDClassifier()
    elif model_name == 'arow':
        # AROWClassifier implementation
        return AROWClassifier()
    elif model_name == 'rda':
        # RDAClassifier implementation
        return RDAClassifier()
    elif model_name == 'ada-fobos':
        # AdaFOBOSClassifier implementation
        return AdaFOBOSClassifier()
    else:
        raise ValueError(f"Unknown model name: {model_name}")
