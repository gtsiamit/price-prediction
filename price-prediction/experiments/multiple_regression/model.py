from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def fit_lr_model(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    """
    Fits a Linear Regression model to the given training data.

    Args:
        x (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Target values of shape (n_samples,).

    Returns:
        LinearRegression: A trained scikit-learn LinearRegression model.
    """

    lr = LinearRegression()
    return lr.fit(x, y)
