import numpy as np


class ExponentialData:
    """
    Class representing exponential data
    """
    def __init__(self, a, b, size, minmax=(0, 1)):
        self.a = a
        self.b = b
        self.size = size
        self.minmax = minmax
        self.x, self.y = self.generate_data()

    def generate_data(self):
        """
        Generates exponential data
        """
        x = np.linspace(*self.minmax, self.size)
        y = self.a * np.exp(x * self.b)
        return x, y

    def __add__(self, noise):
        """
        Adds noise to exponential data
        """
        if hasattr(noise, 'value'):
            self.y += noise.value
            return self
        raise Exception('invalid value')

    def linearize(self):
        """
        Linearizes exponential data
        """
        return np.log(self.y), self.x


class InverseData:
    """
    Class representing inverse data
    """
    def __init__(self, a, b, size, minmax=(0, 1)):
        self.a = a
        self.b = b
        self.size = size
        self.minmax = minmax
        self.x, self.y = self.generate_data()

    def generate_data(self):
        """
        Generates inverse data
        """
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
    """
    Class representing logarithmic data
    """
    def __init__(self, a, b, size, minmax=(1, 2)):
        self.a = a
        self.b = b
        self.size = size
        self.minmax = minmax
        self.x, self.y = self.generate_data()

    def generate_data(self):
        """
        Generates logarithmic data
        """
        x = np.linspace(*self.minmax, self.size)
        y = self.a * np.log(x) + self.b
        return x, y

    def __add__(self, noise):
        """
        Adds noise to logarithmic data
        """
        if hasattr(noise, 'value'):
            self.y += noise.value
            return self
        raise Exception('invalid value')

    def linearize(self):
        """
        Linearizes logarithmic data
        """
        return self.y, np.log(self.x)