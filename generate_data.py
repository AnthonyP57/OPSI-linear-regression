import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white
from statsmodels.stats.stattools import durbin_watson
import pprint
from RegressionVisualizer.noise import UniformNoise, Noise
from RegressionVisualizer.data import ExponentialData, InverseData, LogData
from RegressionVisualizer.regression import apply_regression
import pandas as pd


"""
In this file we generate figures showing the effect of noise on the data.
The data is then linearized and the linear regression is applied to the linearized data.
We can then compare the coefficients, intercepts, and R^2 values.
"""

if __name__ == "__main__":
    a = 1
    b = 2
    size = 16
    noise_levels = [0.01, 0.05, 0.1]
    repetitions = 10

    results_noise = {level: {} for level in noise_levels}
    results_tests = {level: {label: {'MSE': [], 'White_p': [], 'Durbin_Watson': []}
                             for label in ['Exponential', 'Inverse', 'Logarithmic']}
                     for level in noise_levels}

    mu, sigma = 0, 0.01

    exp = ExponentialData(a, b, size) + Noise(mu, sigma, size)
    inv = InverseData(a, b, size) + Noise(mu, sigma, size)
    log = LogData(a, b, size) + Noise(mu, sigma, size)

    # Wykres danych nieliniowych
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

    # Wykres danych liniaryzowanych
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

    # Przygotowanie wykresów
    fig_linear, axs_linear = plt.subplots(len(noise_levels), 3, figsize=(18, 12))
    fig_original, axs_original = plt.subplots(len(noise_levels), 3, figsize=(18, 12))
    fig_residuals, axs_residuals = plt.subplots(len(noise_levels), 3, figsize=(18, 12))
    residuals_min = 1e10
    residuals_max = -1e10

    for _ in range(repetitions):
        for i, sigma in enumerate(noise_levels):
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

                # results_noise[sigma][label] = (coef, intercept, r2)

                # Obliczenia diagnostyczne
                mse = mean_squared_error(y_prime, y_pred)
                exog = sm.add_constant(np.column_stack((x_prime, x_prime ** 2)))
                _, white_p, _, _ = het_white(residuals, exog)
                dw_stat = durbin_watson(residuals)

                results_tests[sigma][label]['MSE'].append(mse)
                results_tests[sigma][label]['White_p'].append(white_p)
                results_tests[sigma][label]['Durbin_Watson'].append(dw_stat)

                # Wykresy

                # Wykres linearyzowany
                axs_linear[i, j].scatter(x_prime, y_prime, label="Dane liniaryzowane")
                axs_linear[i, j].plot(x_prime, y_pred, color="red", label=f"$R^2$={r2:.3f}")
                axs_linear[i, j].set_title(
                    f"{label} (σ={sigma})\na={a}, b={b}\ncoef={coef:.2f}, intercept={intercept:.2f}, MSE={mse:.3f}",
                    fontsize=10
                )
                # axs_linear[i, j].legend(fontsize=10)
                axs_linear[i, j].grid(True)

                # Wykres oryginalny
                axs_original[i, j].scatter(data.x, data.y, label="Dane oryginalne")
                axs_original[i, j].set_title(
                    f"{label} (σ={sigma})\na={a}, b={b}",
                    fontsize=10
                )
                # axs_original[i, j].legend(fontsize=10)
                axs_original[i, j].grid(True)

                # Wykres reszt
                axs_residuals[i, j].bar(range(len(residuals)), residuals)
                axs_residuals[i, j].set_title(
                    f"{label} (σ={sigma})\nDW={dw_stat:.2f}, White p={white_p:.3f}",
                    fontsize=10
                )
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

        averaged_results_mse = {}
        averaged_results_white_p = {}
        averaged_results_dw = {}
        for sigma in noise_levels:
            averaged_results_mse[sigma] = {}
            averaged_results_white_p[sigma] = {}
            averaged_results_dw[sigma] = {}
            for label in ['Exponential', 'Inverse', 'Logarithmic']:
                averaged_results_mse[sigma][label] = np.mean(results_tests[sigma][label]['MSE'])
                averaged_results_white_p[sigma][label] = np.mean(results_tests[sigma][label]['White_p'])
                averaged_results_dw[sigma][label] = np.mean(results_tests[sigma][label]['Durbin_Watson'])

        print("Średnie wyniki diagnostyczne (MSE, White p-value, Durbin-Watson):")
        print('------------------------------')
        print('MSE')
        df = pd.DataFrame(averaged_results_mse)
        print(df)
        print('------------------------------')
        print('White p-value')
        df = pd.DataFrame(averaged_results_white_p)
        print(df)
        print('------------------------------')
        print('Durbin-Watson')
        df = pd.DataFrame(averaged_results_dw)
        print(df)
