import numpy as np
from sklearn.linear_model import LinearRegression

def apply_regression(x, y):
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_, model.score(x, y), model