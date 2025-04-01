import argparse
import os
from pathlib import Path
import numpy as np
from model import *
from utils import load_df
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle

FILEDIR = Path(__file__).parent


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


def train_exp():
    """
    Handles the experimentation process.
    Data load, transformations, models fit, evaluation, results store.
    """

    # load data
    data = load_df(DATASET_PATH)

    # set x (sqft_living) and y (price)
    x = data["sqft_living"].to_numpy().reshape(-1, 1)
    y = data["price"].to_numpy()

    # scale data
    x_scaled = standard_scaling(x)
    y_sc, y_scaled = standard_scaling(y.reshape(-1, 1), return_scaler=True)

    # create polynomial features and apply scaling
    # 2nd degree
    x_poly_2deg = create_poly_features(x, degree=2)
    x_poly_2deg = standard_scaling(x_poly_2deg)
    # 3rd degree
    x_poly_3deg = create_poly_features(x, degree=3)
    x_poly_3deg = standard_scaling(x_poly_3deg)

    # define empty lists for storing actual and predicted
    # values during the StratifiedKFold
    x_actual_folds, y_actual = [], []
    y_pred_lr, y_pred_poly2, y_pred_poly3 = [], [], []

    # separate the data in bins according to price in
    # order to perform stratified train test split
    y_bins = bins_discretize(x=y.reshape(-1, 1))

    # stratified KFold based in the binned y
    skf = StratifiedKFold(n_splits=5, random_state=2025, shuffle=True)
    for num_fold, (train_index, test_index) in enumerate(skf.split(x, y_bins)):
        print(f"----- Fold {num_fold+1} -----")

        # x data for linear regression
        x_train_lr, x_test_lr = x_scaled[train_index], x_scaled[test_index]
        # x data for 2nd degree polynomial regression
        x_train_poly2, x_test_poly2 = x_poly_2deg[train_index], x_poly_2deg[test_index]
        # x data for linear regression
        x_train_poly3, x_test_poly3 = x_poly_3deg[train_index], x_poly_3deg[test_index]
        # y data
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]

        # store actual y test data and x data per fold
        y_actual.append(y_test)
        x_actual_folds.append(x[test_index].flatten())

        # fit linear regression model and make predictions
        print("Linear Regression")
        lr = fit_lr_model(x=x_train_lr, y=y_train)
        pred_lr = lr.predict(x_test_lr)
        y_pred_lr.append(pred_lr)

        # fit 2nd degree polynomial regression model
        print("2nd degree Polynomial Regression")
        poly2_reg = fit_lr_model(x=x_train_poly2, y=y_train)
        pred_poly2 = poly2_reg.predict(x_test_poly2)
        y_pred_poly2.append(pred_poly2)

        # fit 3rd degree polynomial regression model
        print("3rd degree Polynomial Regression")
        poly3_reg = fit_lr_model(x=x_train_poly3, y=y_train)
        pred_poly3 = poly3_reg.predict(x_test_poly3)
        y_pred_poly3.append(pred_poly3)

    # concatenate y test data and y predicted from all folds
    y_actual_all = np.concatenate(y_actual)
    y_pred_lr_all = np.concatenate(y_pred_lr)
    y_pred_poly2_all = np.concatenate(y_pred_poly2)
    y_pred_poly3_all = np.concatenate(y_pred_poly3)
    x_actual_folds_all = np.concatenate(x_actual_folds)

    # bring the y values to the original scale
    y_actual_all = y_sc.inverse_transform(y_actual_all)
    y_pred_lr_all = y_sc.inverse_transform(y_pred_lr_all)
    y_pred_poly2_all = y_sc.inverse_transform(y_pred_poly2_all)
    y_pred_poly3_all = y_sc.inverse_transform(y_pred_poly3_all)

    # calculate metrics in order to evaluate the models
    print("Calculating evaluation metrics")
    lr_mse, lr_mae, lr_r2 = (
        mean_squared_error(y_true=y_actual_all, y_pred=y_pred_lr_all),
        mean_absolute_error(y_true=y_actual_all, y_pred=y_pred_lr_all),
        r2_score(y_true=y_actual_all, y_pred=y_pred_lr_all),
    )
    poly2_mse, poly2_mae, poly2_r2 = (
        mean_squared_error(y_true=y_actual_all, y_pred=y_pred_poly2_all),
        mean_absolute_error(y_true=y_actual_all, y_pred=y_pred_poly2_all),
        r2_score(y_true=y_actual_all, y_pred=y_pred_poly2_all),
    )
    poly3_mse, poly3_mae, poly3_r2 = (
        mean_squared_error(y_true=y_actual_all, y_pred=y_pred_poly3_all),
        mean_absolute_error(y_true=y_actual_all, y_pred=y_pred_poly3_all),
        r2_score(y_true=y_actual_all, y_pred=y_pred_poly3_all),
    )

    # format metrics in str in order to be stored in txt
    metrics_lr = [f"MSE: {lr_mse}\n", f"MAE: {lr_mae}\n", f"R2 Score: {lr_r2}\n"]
    metrics_poly2 = [
        f"MSE: {poly2_mse}\n",
        f"MAE: {poly2_mae}\n",
        f"R2 Score: {poly2_r2}\n",
    ]
    metrics_poly3 = [
        f"MSE: {poly3_mse}\n",
        f"MAE: {poly3_mae}\n",
        f"R2 Score: {poly3_r2}\n",
    ]

    # store results, y of each fold and y scaler locally
    print("Storing results")
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("y_actual_all.csv"), X=y_actual_all, delimiter=","
    )
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("y_pred_lr_all.csv"), X=y_pred_lr_all, delimiter=","
    )
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("y_pred_poly2_all.csv"),
        X=y_pred_poly2_all,
        delimiter=",",
    )
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("y_pred_poly3_all.csv"),
        X=y_pred_poly3_all,
        delimiter=",",
    )
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("x_actual_folds_all.csv"),
        X=x_actual_folds_all,
        delimiter=",",
    )

    with open(OUTPUT_PATH.joinpath("sc.pkl"), "wb") as f:
        pickle.dump(y_sc, f)

    with open(OUTPUT_PATH.joinpath("lr_results.txt"), "w") as f:
        f.writelines(metrics_lr)
    with open(OUTPUT_PATH.joinpath("poly2_results.txt"), "w") as f:
        f.writelines(metrics_poly2)
    with open(OUTPUT_PATH.joinpath("poly3_results.txt"), "w") as f:
        f.writelines(metrics_poly3)

    print("Execution Finished")


def main():

    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to dataset"
    )
    args = parser.parse_args()

    # set dataset path
    global DATASET_PATH
    DATASET_PATH = args.dataset_path
    DATASET_PATH = Path(DATASET_PATH)

    # set and create (if not exists) the output folder
    global OUTPUT_PATH
    OUTPUT_PATH = FILEDIR.joinpath("output")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # run train experiments
    train_exp()


if __name__ == "__main__":
    main()
