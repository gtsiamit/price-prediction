import argparse
import os
from pathlib import Path
import numpy as np
from model import fit_lr_model
from utils import load_df
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
from features import standard_scaling, bins_discretize

FILEDIR = Path(__file__).parent


def train_exp():
    """
    Handles the experimentation process, for multiple linear regression.
    Data load, transformations, model fit, evaluation, results store.
    """

    # load data
    data = load_df(DATASET_PATH)

    # transform feature columns
    # calculate years from year built
    data["years"] = data["yr_built"].apply(lambda x: 2025 - x)
    # boolean feature renovated from year renovated
    data["renovated"] = data["yr_renovated"].apply(lambda x: 0 if x == 0 else 1)

    # set x and y (price)
    x_columns = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "sqft_basement",
        "years",
        "lat",
        "long",
        "waterfront",
        "view",
        "condition",
        "grade",
        "renovated",
    ]
    x = data[x_columns].copy()
    y = data["price"].to_numpy()

    # scale continuus numerical features
    x_scaled = x[
        [
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "sqft_basement",
            "years",
            "lat",
            "long",
        ]
    ]
    x_non_scaled = x[
        ["waterfront", "view", "condition", "grade", "renovated"]
    ].to_numpy()
    x_scaled = standard_scaling(x_scaled.to_numpy())
    # concat scaled and non-scaled data
    x = np.concatenate([x_scaled, x_non_scaled], axis=1)

    # scale y data
    y_sc, y_scaled = standard_scaling(y.reshape(-1, 1), return_scaler=True)

    # define empty lists for storing actual and predicted
    # values during the StratifiedKFold
    x_actual_folds, y_actual = [], []
    y_pred = []

    # separate the data in bins according to price in
    # order to perform stratified train test split
    y_bins = bins_discretize(x=y.reshape(-1, 1))

    # stratified KFold based in the binned y
    skf = StratifiedKFold(n_splits=5, random_state=2025, shuffle=True)
    for num_fold, (train_index, test_index) in enumerate(skf.split(x, y_bins)):
        print(f"----- Fold {num_fold+1} -----")

        # x data for multiple linear regression
        x_train, x_test = x[train_index], x[test_index]
        # y data
        y_train, y_test = y_scaled[train_index], y_scaled[test_index]

        # store actual y test data and x data per fold
        y_actual.append(y_test)
        x_actual_folds.append(x[test_index].flatten())

        # fit multiple linear regression model and make predictions
        print("Linear Regression")
        mlr = fit_lr_model(x=x_train, y=y_train)
        pred_lr = mlr.predict(x_test)
        y_pred.append(pred_lr)

    # concatenate y test data and y predicted from all folds
    y_actual_all = np.concatenate(y_actual)
    y_pred_lr_all = np.concatenate(y_pred)
    x_actual_folds_all = np.concatenate(x_actual_folds)

    # bring the y values to the original scale
    y_actual_all = y_sc.inverse_transform(y_actual_all)
    y_pred_lr_all = y_sc.inverse_transform(y_pred_lr_all)

    # calculate metrics in order to evaluate the model
    print("Calculating evaluation metrics")
    mlr_mse, mlr_mae, mlr_r2 = (
        mean_squared_error(y_true=y_actual_all, y_pred=y_pred_lr_all),
        mean_absolute_error(y_true=y_actual_all, y_pred=y_pred_lr_all),
        r2_score(y_true=y_actual_all, y_pred=y_pred_lr_all),
    )

    # format metrics in str in order to be stored in txt
    metrics_mlr = [f"MSE: {mlr_mse}\n", f"MAE: {mlr_mae}\n", f"R2 Score: {mlr_r2}\n"]

    # store results, y of each fold and y scaler locally
    print("Storing results")
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("y_actual_all.csv"), X=y_actual_all, delimiter=","
    )
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("y_pred_lr_all.csv"), X=y_pred_lr_all, delimiter=","
    )
    np.savetxt(
        fname=OUTPUT_PATH.joinpath("x_actual_folds_all.csv"),
        X=x_actual_folds_all,
        delimiter=",",
    )

    with open(OUTPUT_PATH.joinpath("sc.pkl"), "wb") as f:
        pickle.dump(y_sc, f)

    with open(OUTPUT_PATH.joinpath("mlr_results.txt"), "w") as f:
        f.writelines(metrics_mlr)

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
    OUTPUT_PATH = FILEDIR.joinpath("output_mlr")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # run train experiments
    train_exp()


if __name__ == "__main__":
    main()
