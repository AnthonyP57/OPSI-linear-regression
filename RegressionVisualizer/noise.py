import numpy as np

class Noise:
    """
    A class representing Gaussian noise.
    """
    def __init__(self, mu, sigma, size):
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.value = np.random.normal(mu, sigma, size)

class UniformNoise:
    """
    A class representing uniform noise.
    """
    def __init__(self, mu, max_deviation, size):
        self.mu = mu
        self.max_deviation = max_deviation
        self.size = size
        self.value = (np.random.random(size)*2 - 1) * max_deviation + mu