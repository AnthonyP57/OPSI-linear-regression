from generate_data import LogData, InverseData, ExponentialData, Noise, apply_regression, UniformNoise
import numpy as np
from sklearn.metrics import r2_score
import pandas as pd


def get_stats(n_measurements, a, b, size, mu, sigma, out_path=None, noise_class=Noise): # not most efficient code but works
    """
    This function generates data using LogData, InverseData, and ExponentialData classes, adds noise using the specified noise class,
    and calculates the R2 values for each type of data. The results are then stored in a pandas DataFrame and returned.
    If out_path is specified, the DataFrame is also saved to a CSV file at the specified location.
    """

    in_data_x = [[] for _ in range(3)]
    in_data_y = [[] for _ in range(3)]
    out_data_y = [[] for _ in range(3)]

    for _ in range(n_measurements):
        for i, t in enumerate([LogData, InverseData, ExponentialData]):
            data = t(a, b, size)
            noise = noise_class(mu, sigma, size)

            data += noise

            y_prime, x_prime = data.linearize()
            coef, intercept, r2, model = apply_regression(x_prime, y_prime)
            y_pred = model.predict(x_prime.reshape(-1, 1))

            in_data_x[i].append(x_prime)
            in_data_y[i].append(y_prime)
            out_data_y[i].append(y_pred)

    log_r2 = [r2_score(label, pred) for label, pred in zip(in_data_y[0], out_data_y[0])]
    inv_r2 = [r2_score(label, pred) for label, pred in zip(in_data_y[1], out_data_y[1])]
    exp_r2 = [r2_score(label, pred) for label, pred in zip(in_data_y[2], out_data_y[2])]
    r2s = [log_r2, inv_r2, exp_r2]

    metrics = pd.DataFrame(columns=['mean R2', 'std'], index=['log', 'inv', 'exp'])

    for i in range(3):
        metrics.iloc[i,:] = [np.mean(r2s[i]), np.std(r2s[i])]

    if out_path is not None:
        metrics.to_csv(out_path)
    return metrics

if __name__ == '__main__':
    n_measurements = 100
    a=1
    b=2
    size=32
    mu, sigma = 0, 0.1

    tab = get_stats(n_measurements, a, b, size, mu, sigma, noise_class=Noise, out_path='./gauss_0_1.csv')