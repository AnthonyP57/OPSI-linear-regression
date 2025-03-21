import numpy as np
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import warnings
import matplotlib.pyplot as plt
from modules import timeit
import time

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
    def __init__(self, in_features, bias=False, lr=0.1, device='cuda:0'):
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
        return self
    
    def validate(self, x, y):
        y_pred = self.forward(x)
        y, y_pred = y.detach().numpy(), y_pred.detach().numpy()
        return np.mean((y - y_pred)**2), r2_score(y, y_pred)


if __name__ == "__main__":

    x = np.arange(0, 1000, 1, dtype=np.float32).reshape(-1, 1)
    # x = np.hstack((x, x*1.5))
    x = MinMaxScaler().fit_transform(x)
    # y = np.sum(x, axis=1).reshape(-1, 1) + 1
    y = x*2 + 1

    print(x.shape, y.shape)

    def NumpyLinReg_(x, y):
        model = NumpyLinReg(x.shape[1], bias=True, lr=1e-2)
        start = time.time()
        model.train(x, y)
        print(time.time() - start)
        print(model.validate(x, y))
        # print(model.weights, model.bias)
        # plt.scatter(x, model.forward(x))
        # plt.scatter(x, y)
        # plt.savefig("NumpyLinReg.png")
        # plt.close()

    NumpyLinReg_(x, y)

    def closed_form(x, y):
        closed = NumpyLinRegCloseForm(bias=True)
        start = time.time()
        closed.fit(x, y)
        print(time.time() - start)
        print(closed.validate(x, y))
        # print(closed.weights)
        # plt.scatter(x, closed.forward(x))
        # plt.scatter(x, y)
        # plt.savefig("NumpyLinRegCloseForm.png")
        # plt.close()
    
    closed_form(x, y)

    x, y = torch.from_numpy(x), torch.from_numpy(y)

    def TorchLinReg_(x, y):
        model = TorchLinReg(x.shape[1], bias=True, lr=1e-1, device=None)
        start = time.time()
        model.train(x, y)
        print(time.time() - start)
        print(model.validate(x, y))
        # print(model.l.weight, model.l.bias)
        # plt.scatter(x, model.forward(x).detach().numpy())
        # plt.scatter(x, y)
        # plt.savefig("TorchLinReg.png")
        # plt.close()

    TorchLinReg_(x, y)

