from sklearn.linear_model import LinearRegression

def apply_regression(x, y):
    """
    This function reshapes the input data to a column vector, fits a linear regression model using scikit-learn's LinearRegression,
    and returns the model's coefficients, intercept, and R-squared value.
    """
    x = x.reshape(-1, 1)
    model = LinearRegression()
    model.fit(x, y)
    return model.coef_[0], model.intercept_, model.score(x, y), model