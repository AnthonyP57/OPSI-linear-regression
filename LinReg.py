import numpy as np
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import warnings
import matplotlib.pyplot as plt
from modules import timeit
import pandas as pd

warnings.filterwarnings("ignore")

class NumpyLinReg:
    def __init__(self, in_features, bias=False, lr=0.1):
        self.weights = np.random.random(in_features) # ~ (f)
        self.bias = np.random.random(1) # ~ (1)
        self.calc_bias = bias
        self.lr = lr

    def forward(self, x):
        # x ~ (n, f)
        return self.weights @ x.T + self.bias
    
    def update_weights(self, x, y):
        for i in range(len(self.weights)):
            derivative = - x[i] * (y - self.weights[i] * x[i] - self.bias) # SE = 1/2 * (y - y')^2
            self.weights[i] -= self.lr*derivative

    def update_bias(self, x, y):
        derivative = - (y - self.weights @ x.T - self.bias) # SE = 1/2 * (y - y')^2
        self.bias -= self.lr*derivative
    
    @timeit
    def train(self, x, y, epochs=100):
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
        y_pred = self.forward(x)
        y = y.reshape(-1)
        return np.mean((y - y_pred)**2), r2_score(y, y_pred)

class NumpyLinRegCloseForm:
    def __init__(self, bias=False):
        self.bias = bias
        self.weights = None

    @timeit
    def fit(self, x, y):
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        try:
            self.weights = np.linalg.inv(x.T @ x) @ x.T @ y
        except Exception as e:
            print(e)
            self.weights = np.linalg.pinv(x.T @ x) @ x.T @ y
    
    def forward(self, x):
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0], 1))))
        return x @ self.weights
    
    def validate(self, x, y):
        y_pred = self.forward(x)
        y = y.reshape(-1)
        return np.mean((y - y_pred)**2), r2_score(y, y_pred)


class TorchLinReg(nn.Module):
    def __init__(self, in_features, bias=False, lr=0.1, device=None):
        super().__init__()
        self.l = nn.Linear(in_features, 1, bias=bias)
        self.lr = lr
        self.device = device
        if device:
            self.to(device)
    
    def forward(self, x):
        if self.device:
            x = x.to(self.device)
        return self.l(x)

    @timeit
    def train(self, x, y, epochs=100):
        if self.device:
            x, y = x.to(self.device), y.to(self.device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        for _ in range(epochs):
            y_pred = self.forward(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def validate(self, x, y):
        y_pred = self.forward(x)
        y, y_pred = y.detach().numpy(), y_pred.detach().numpy()
        return np.mean((y - y_pred)**2), r2_score(y, y_pred)


if __name__ == "__main__":

    # REVERSE COMMENTS HERE TO SWITCH BETWEEN LR AND MLR (no visualization)
    x = np.arange(0, 1000, 1, dtype=np.float32).reshape(-1, 1)/1000
    # x = np.hstack((x, x*1.5))
    # y = np.sum(x, axis=1).reshape(-1, 1) + 0.1
    y = x*2 + 1

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

    print(x.shape, y.shape)

    model = NumpyLinReg(x.shape[1], bias=True, lr=1e-5 if x.shape[1] == 2 else 1e-2)
    model.train(x_train, y_train)
    nlt_metrics = model.validate(x_test, y_test)
    nlt_weights = np.hstack((model.weights, model.bias))

    plt.plot((min(x_test), max(x_test)), (min(y_test), max(y_test)), linestyle='--', color='red', linewidth=1, label='True Values')
    plt.scatter(x_test, model.forward(x_test), s=10, linewidths=0.5, label=f'Predicted Values (R2={nlt_metrics[1]:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Numpy Linear Regression Backpropagation')
    plt.legend()
    plt.savefig("NumpyLinReg.png")
    plt.close()


    closed = NumpyLinRegCloseForm(bias=True)
    closed.fit(x_train, y_train)
    nltc_metrics = closed.validate(x_test, y_test)
    clr_weights = closed.weights.reshape(-1)
    
    plt.plot((min(x_test), max(x_test)), (min(y_test), max(y_test)), linestyle='--', color='red', linewidth=1, label='True Values')
    plt.scatter(x_test, model.forward(x_test), s=10, linewidths=0.5, label=f'Predicted Values (R2={nltc_metrics[1]:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Numpy Linear Regression Closed Form')
    plt.legend()
    plt.savefig("NumpyLinRegCloseForm.png")
    plt.close()
    
    x_train, y_train = torch.from_numpy(x_train), torch.from_numpy(y_train)
    x_test, y_test = torch.from_numpy(x_test), torch.from_numpy(y_test)

    model = TorchLinReg(x.shape[1], bias=True, lr=1e-1, device=None)
    model.train(x_train, y_train)
    tlr_metrics = model.validate(x_test, y_test)
    tlt_weights = np.hstack((model.l.weight.detach().numpy().reshape(-1), model.l.bias.detach().numpy()))
    
    plt.plot((min(x_test), max(x_test)), (min(y_test), max(y_test)), linestyle='--', color='red', linewidth=1, label='True Values')
    plt.scatter(x_test, model.forward(x_test).detach().numpy(), s=10, linewidths=0.5, label=f'Predicted Values (R2={tlr_metrics[1]:.2f})')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Torch Linear Regression Backpropagation')
    plt.legend()
    plt.savefig("TorchLinReg.png")
    plt.close()

    data = np.vstack((nlt_metrics, nltc_metrics, tlr_metrics))
    data = np.hstack((data, np.vstack((nlt_weights, clr_weights, tlt_weights)))).round(3)
    metrics = pd.DataFrame(columns=['MSE', 'R2']+[f'weight_{i}' for i in range(x.shape[1])]+['bias'], index=['NumpyLinReg', 'NumpyLinRegCloseForm', 'TorchLinReg'], data=data)
    print(metrics.to_string())


