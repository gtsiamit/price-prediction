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


def create_poly_features(x: np.ndarray, degree: int) -> np.ndarray:
    """
    Transforms input features into polynomial features of the specified degree.

    Args:
        x (np.ndarray): Input feature array of shape (n_samples, n_features).
        degree (int): The degree of the polynomial features to generate.

    Returns:
        np.ndarray: Transformed feature array with polynomial features.
    """

    pf = PolynomialFeatures(degree=degree)
    return pf.fit_transform(x)
