import numpy as np
import matplotlib.pyplot as plt

class ExponentialData:
    def __init__(self, a, b, size, minmax=(0,1)):
        self.a = a
        self.b = b
        self.size = size
        self.minmax = minmax
        self.x, self.y = self.generate_data()

    def generate_data(self):
        x = np.linspace(*self.minmax, self.size)
        y = self.a * np.exp(x * self.b)
        return x, y
    
    def __add__(self, noise):
        if hasattr(noise, 'value'):
            self.y += noise.value
            return self
        raise Exception('invalid value')
    
    def linearize(self):
        return np.log(self.y), self.x

class Noise:
    def __init__(self, mu, sigma, size):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.value = np.random.normal(mu, sigma, size)

class InverseData:
    def __init__(self, a, b, size, minmax=(0,1)):
        self.a = a
        self.b = b
        self.size = size
        self.minmax = minmax
        self.x, self.y = self.generate_data()

    def generate_data(self):
        x = np.linspace(*self.minmax, self.size)
        y = self.a / (x + self.b)
        return x, y
    
    def __add__(self, noise):
        if hasattr(noise, 'value'):
            self.y += noise.value
            return self
        raise Exception('invalid value')
    
    def linearize(self):
        return 1/self.y, self.x

class LogData:
    def __init__(self, a, b, size, minmax=(1,2)):
        self.a = a
        self.b = b
        self.size = size
        self.minmax = minmax
        self.x, self.y = self.generate_data()

    def generate_data(self):
        x = np.linspace(*self.minmax, self.size)
        y = self.a * np.log(x) + self.b
        return x,y

    def __add__(self, noise):
        if hasattr(noise, 'value'):
            self.y += noise.value
            return self
        raise Exception('invalid value')
    
    def linearize(self):
        return self.y, np.log(self.x)

if __name__ == "__main__":
    a=1
    b=2
    size=16

    mu, sigma = 0, 0.01

    exp = ExponentialData(a,b,size)
    inv = InverseData(a,b,size)
    log = LogData(a,b,size)
    noise = Noise(mu, sigma, size)

    exp = exp + noise
    inv = inv + noise
    log = log + noise

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(log.x, log.y, label='y = a * ln(x) + b')
    plt.scatter(inv.x, inv.y, label='y = a / (x + b)')
    plt.scatter(exp.x, exp.y, label='y = a * exp(x * b)')
    plt.xlabel(r"x")
    plt.ylabel(r"y")
    plt.title(r"y = f(x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data.png')
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(*log.linearize(), label='y = a * ln(x) + b')
    plt.scatter(*inv.linearize(), label='1/y = x/a + b/a')
    plt.scatter(*exp.linearize(), label='ln(y) = ln(a) + b * x')
    plt.xlabel(r"x'")
    plt.ylabel(r"y'")
    plt.title(r"y' = f(x')")
    plt.legend()
    plt.tight_layout()
    plt.savefig('./data_linear.png')
    plt.close()