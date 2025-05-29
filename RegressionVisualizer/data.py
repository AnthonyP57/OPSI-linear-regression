import numpy as np


class ExponentialData:
    def __init__(self, a, b, size, minmax=(0, 1)):
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


class InverseData:
    def __init__(self, a, b, size, minmax=(0, 1)):
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
        return 1 / self.y, self.x


class LogData:
    def __init__(self, a, b, size, minmax=(1, 2)):
        self.a = a
        self.b = b
        self.size = size
        self.minmax = minmax
        self.x, self.y = self.generate_data()

    def generate_data(self):
        x = np.linspace(*self.minmax, self.size)
        y = self.a * np.log(x) + self.b
        return x, y

    def __add__(self, noise):
        if hasattr(noise, 'value'):
            self.y += noise.value
            return self
        raise Exception('invalid value')

    def linearize(self):
        return self.y, np.log(self.x)