import numpy as np
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer


def standard_scaling(x: np.ndarray, return_scaler: bool = False) -> np.ndarray:
    """
    Applies standard scaling to the input data. Optionally, returns
    the fitted StandardScaler object.

    Args:
        x (np.ndarray): Input data to be scaled.
        return_scaler (bool, optional): If True, returns the fitted StandardScaler
                                        object along with the scaled data. Defaults to False.

    Returns:
        StandardScaler (optional): Fitted StandardScaler object, only returned if `return_scaler=True`.
        np.ndarray: Scaled data with mean 0 and standard deviation 1.
    """

    sc = StandardScaler()
    if return_scaler:
        return sc, sc.fit_transform(x)
    else:
        return sc.fit_transform(x)


def bins_discretize(x: np.ndarray) -> np.ndarray:
    """
    Discretizes continuous data into bins using quantile-based binning.

    Args:
        x (np.ndarray): Input array of continuous values.

    Returns:
        np.ndarray: Discretized integer bin labels.
    """

    bd = KBinsDiscretizer(encode="ordinal", strategy="quantile", random_state=2025)
    return bd.fit_transform(x).astype(int).flatten()
