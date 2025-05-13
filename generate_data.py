import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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

class UniformNoise:
    def __init__(self, mu, max_deviation, size):
        self.mu = mu
        self.max_deviation = max_deviation
        self.size = size
        self.value = (np.random.random(size)*2 - 1) * max_deviation + mu        

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

def apply_regression(x, y):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_, model.score(x, y), model

if __name__ == "__main__":
    a=1
    b=2
    size=16

    noise_levels = [0.01, 0.05, 0.1]
    results_noise = {level: {} for level in noise_levels}

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

    fig_linear, axs_linear = plt.subplots(len(noise_levels), 3, figsize=(18, 12))
    fig_original, axs_original = plt.subplots(len(noise_levels), 3, figsize=(18, 12))
    fig_residuals, axs_residuals = plt.subplots(len(noise_levels), 3, figsize=(18, 12))
    residuals_min = 1e10
    residuals_max = -1e10

    for i, sigma in enumerate(noise_levels):
        # noise = Noise(mu=0, sigma=sigma, size=size)
        noise = UniformNoise(mu=0, max_deviation=sigma, size=size)

        for j, (label, DataClass) in enumerate([
            ('Exponential', ExponentialData),
            ('Inverse', InverseData),
            ('Logarithmic', LogData)
        ]):
            data = DataClass(a, b, size) + noise
            y_prime, x_prime = data.linearize()
            coef, intercept, r2, model = apply_regression(x_prime, y_prime)
            y_pred = model.predict(x_prime.reshape(-1, 1))
            residuals = y_prime - y_pred

            results_noise[sigma][label] = (coef, intercept, r2)

            # Wykres linearyzowany
            axs_linear[i, j].scatter(x_prime, y_prime, label="Dane liniaryzowane")
            axs_linear[i, j].plot(x_prime, y_pred, color="red", label=f"$R^2$={r2:.3f}")
            axs_linear[i, j].set_title(f"{label} (max deviation={sigma}) - lin.")
            axs_linear[i, j].legend()
            axs_linear[i, j].grid(True)

            # Wykres oryginalny
            axs_original[i, j].scatter(data.x, data.y, label="Dane oryginalne")
            axs_original[i, j].set_title(f"{label} (max deviation={sigma}) - orig.")
            axs_original[i, j].grid(True)

            # Wykres reszt
            axs_residuals[i, j].bar(range(len(residuals)), residuals)
            axs_residuals[i, j].set_title(f"{label} (max deviation={sigma}) - reszty")
            axs_residuals[i, j].grid(True)

            if residuals_min > min(residuals):
                residuals_min = min(residuals)
            if residuals_max < max(residuals):
                residuals_max = max(residuals)

    residuals_max *= 1.1
    residuals_min *= 0.9

    for plot in [axs_linear, axs_original, axs_residuals]:
        for ax in plot.flatten():
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                    ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(16)

    fig_linear.tight_layout()
    fig_linear.savefig('linearized_all_sigmas_uniform.png')
    plt.close(fig_linear)

    fig_original.tight_layout()
    fig_original.savefig('original_all_sigmas_uniform.png')
    plt.close(fig_original)

    for i in range(len(noise_levels)):
        axs_residuals[i, 0].set_ylim(residuals_min, residuals_max)
        axs_residuals[i, 1].set_ylim(residuals_min, residuals_max)
        axs_residuals[i, 2].set_ylim(residuals_min, residuals_max)
    

    fig_residuals.tight_layout()
    fig_residuals.savefig('residuals_all_sigmas_uniform.png')
    plt.close(fig_residuals)
