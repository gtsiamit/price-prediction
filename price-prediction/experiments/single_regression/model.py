from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def build_lr_model() -> LinearRegression:
    """
    Creates and returns a Linear Regression model.

    Returns:
        LinearRegression: A scikit-learn LinearRegression model instance.
    """
    return LinearRegression()


def build_poly_model(degree: int) -> PolynomialFeatures:
    """
    Creates and returns a PolynomialFeatures transformer with the specified degree.

    Args:
        degree (int): The degree of the polynomial features to generate.

    Returns:
        PolynomialFeatures: A scikit-learn PolynomialFeatures instance.
    """

    return PolynomialFeatures(degree=degree)
