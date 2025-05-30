import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import matplotlib.pyplot as plt
from modules import timeit, xy_plot, actual_vs_pred, xyz_plot
import pandas as pd

class NumpyLinReg:
    """
    An implementation of linear regression (LR) as a single neuron
    """
    def __init__(self, in_features, bias=False, lr=0.1):
        """
        initiate all the model parameters
        """
        self.weights = np.random.randn(in_features) # ~ (f)
        self.bias = np.random.randn(1) if bias else np.zeros(1) # ~ (1)
        self.calc_bias = bias
        self.lr = lr

    def forward(self, x):
        """
        function for inference
        """
        # x ~ (n, f)
        return x @ self.weights + self.bias
    
    def update_weights(self, x, y):
        """
        single training iteration
        """
        weight_sum = np.dot(self.weights, x)
        for i in range(len(self.weights)):
            derivative = - x[i] * (y - weight_sum - self.bias) # SE = 1/2 * (y - y')^2
            self.weights[i] -= self.lr*derivative.item()

    def update_bias(self, x, y):
        derivative = - (y - x @ self.weights - self.bias) # SE = 1/2 * (y - y')^2
        self.bias -= self.lr*derivative.item()
    
    @timeit
    def train(self, x, y, epochs=100):
        """
        training loop
        """
        xy = np.random.permutation(np.hstack((x, y))) # shuffle
        x, y = xy[:, :-1], xy[:, -1]

        if self.calc_bias:
            for _ in range(epochs):
                for x_, y_ in zip(x, y):
                    self.update_weights(x_, y_)
                    self.update_bias(x_, y_)
        else:
            for _ in range(epochs):
                for x_, y_ in zip(x, y):
                    self.update_weights(x_, y_)

        return self
    
    def validate(self, x, y):
        """
        validation of the model by calculating MSE and R2
        """
        y_pred = self.forward(x)
        y = y.reshape(-1)
        return np.mean((y - y_pred)**2), r2_score(y, y_pred)

class NumpyLinRegCloseForm:
    """
    An implementation of linear regresiion based its the closed form
    """
    def __init__(self, bias=False):
        """
        initiate parameters
        """
        self.bias = bias
        self.weights = None

    @timeit
    def fit(self, x, y):
        """
        fit the paremeters to the data, here we do not require training (as it is only required for gradient based approaches   )
        """
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        try:
            self.weights = np.linalg.inv(x.T @ x) @ x.T @ y
        except:
            warnings.warn("matrix inverse failed, using pseudo inverse instead", Warning)
            self.weights = np.linalg.pinv(x.T @ x) @ x.T @ y
    
    def forward(self, x):
        """
        inference
        """
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        return x @ self.weights
    
    def validate(self, x, y):
        """
        validation og the model based on MSE and R2
        """
        y_pred = self.forward(x)
        return np.mean((y - y_pred)**2), r2_score(y, y_pred)


class TorchLinReg(nn.Module):
    """
    An implementation of linear regression (LR) as a single neuron with Pytorch
    """
    def __init__(self, in_features, bias=False, lr=0.1, device=None):
        """
        inherit from nn.Module and initiate parameters
        """
        super().__init__()
        self.l = nn.Linear(in_features, 1, bias=bias)
        self.lr = lr
        self.device = device
        self.bias = bias
        if device:
            self.to(device)
    
    def forward(self, x):
        """
        forward function, as required by Pytorch nn.Module
        """
        if self.device:
            x = x.to(self.device)
        return self.l(x)

    @timeit
    def train(self, x, y, epochs=100):
        """
        training loop
        """
        if self.device:
            x, y = x.to(self.device), y.to(self.device)
        # data = TensorDataset(x, y)
        # data_loader = DataLoader(data, batch_size=1, shuffle=True)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        for _ in range(epochs):
            # for x, y in data_loader:
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def validate(self, x, y):
        """
        validation loop - MSE and R2
        """
        y_pred = self.forward(x)
        y, y_pred = y.detach().numpy(), y_pred.detach().numpy()
        return np.mean((y - y_pred)**2), r2_score(y, y_pred)


if __name__ == "__main__":

    # REVERSE COMMENTS HERE TO SWITCH BETWEEN LR AND MLR
    x = np.arange(0, 1000, 1, dtype=np.float32).reshape(-1, 1)/1000
    x = np.hstack((x, x*1.5))
    y = np.sum(x, axis=1).reshape(-1, 1) + 0.1
    # y = x*2 + 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

    print(x.shape, y.shape)

    model = NumpyLinReg(x.shape[1], bias=True, lr=1e-1)
    model.train(x_train, y_train)
    nlt_metrics = model.validate(x_test, y_test)
    nlt_metrics_train = model.validate(x_train, y_train)
    nlt_weights = np.hstack((model.weights, model.bias))

    # xy_plot(x_test, y_test, model.forward(x_test), nlt_metrics, 'NumpyLinReg.png', 'numpy LR backprop')
    actual_vs_pred(y_test, model.forward(x_test), nlt_metrics, 'NumpyLinReg_ap.png', 'numpy LR backprop')
    xyz_plot(x_test, y_test, model.forward(x_test), nlt_metrics, 'NumpyLinReg_3d.png', 'numpy LR backprop')

    closed = NumpyLinRegCloseForm(bias=True)
    closed.fit(x_train, y_train)
    nltc_metrics = closed.validate(x_test, y_test)
    nltc_metrics_train = closed.validate(x_train, y_train)
    clr_weights = closed.weights.reshape(-1) if closed.bias else np.concat((closed.weights.reshape(-1), np.zeros(1)))
    
    # xy_plot(x_test, y_test, closed.forward(x_test), nltc_metrics, 'NumpyLinRegCloseForm.png', 'numpy LR closed form')
    actual_vs_pred(y_test, closed.forward(x_test), nltc_metrics, 'NumpyLinRegCloseForm_ap.png', 'numpy LR closed form')
    xyz_plot(x_test, y_test, closed.forward(x_test), nltc_metrics, 'NumpyLinRegCloseForm_3d.png', 'numpy LR closed form')
    
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

    model = TorchLinReg(x.shape[1], bias=True, lr=1e-1, device=None)
    model.train(x_train, y_train)
    tlr_metrics = model.validate(x_test, y_test)
    tlr_metrics_train = model.validate(x_train, y_train)
    tlt_weights = np.hstack((model.l.weight.detach().numpy().reshape(-1), model.l.bias.detach().numpy())) if model.bias else np.concat((model.l.weight.detach().numpy().reshape(-1), np.zeros(1)))
    
    # xy_plot(x_test, y_test, model.forward(x_test).detach().numpy(), tlr_metrics, 'TorchLinReg.png', 'torch LR backprop')
    actual_vs_pred(y_test, model.forward(x_test).detach().numpy(), tlr_metrics, 'TorchLinReg_ap.png', 'torch LR backprop')
    xyz_plot(x_test, y_test, model.forward(x_test).detach().numpy(), tlr_metrics, 'TorchLinReg_3d.png', 'torch LR backprop')

    data = np.vstack((nlt_metrics, nltc_metrics, tlr_metrics))
    data = np.hstack((data, np.vstack((nlt_metrics_train, nltc_metrics_train, tlr_metrics_train))))
    data = np.hstack((data, np.vstack((nlt_weights, clr_weights, tlt_weights)))).round(3)
    metrics = pd.DataFrame(columns=['MSE test', 'R2 test', 'MSE train', 'R2 train']+[f'weight_{i}' for i in range(x.shape[1])]+['bias'], index=['NumpyLinReg', 'NumpyLinRegCloseForm', 'TorchLinReg'], data=data)
    print(metrics.to_string())