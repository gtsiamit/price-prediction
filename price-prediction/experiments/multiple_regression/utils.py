import pandas as pd
from pathlib import Path
from typing import Union


def load_df(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Loads a CSV file into a pandas DataFrame.

    Args:
        filepath (Union[str, Path]): The path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the CSV data.
    """

    return pd.read_csv(filepath, sep=",")
