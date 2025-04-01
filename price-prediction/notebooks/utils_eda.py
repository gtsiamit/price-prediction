import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple


def plot_boxplot(
    x: Union[pd.Series, np.ndarray, list],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Optional[tuple] = (12, 4),
    xticks: Optional[Union[list, np.ndarray]] = None,
    generate_xticks: Optional[bool] = False,
    num_xticks: Optional[int] = 20,
) -> None:
    """
    Plots a horizontal boxplot with optional custom x-ticks and grid.

    Args:
        x (Union[pd.Series, np.ndarray, list]): The data to be plotted in the boxplot.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        figsize (Optional[tuple], optional): The size of the figure (width, height). Default is (12, 4).
        xticks (Optional[Union[list, np.ndarray]], optional): Custom x-tick positions. If None, defaults to auto-generated ticks or custom-generated xticks. Default is None.
        generate_xticks (Optional[bool], optional): If True, generates evenly spaced x-ticks based on the data range. Default is False.
        num_xticks (Optional[int], optional): The number of custom x-ticks to be generated if `generate_xticks` is True. Default is 20.
    """
    plt.figure(figsize=figsize)
    # create a horizontal boxplot with patch coloring
    plt.boxplot(x, vert=False, patch_artist=True, widths=0.75)

    # set plot title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # generate evenly spaced x-ticks if requested
    if generate_xticks:
        step_xticks = (max(x) - min(x)) // num_xticks
        xticks = np.arange(0, max(x) + step_xticks, step_xticks)

    # show x-ticks in plot if available
    if isinstance(xticks, (np.ndarray, list)):
        plt.xticks(ticks=xticks, rotation=45)

    # remove y-axis labels since it's a single boxplot
    plt.yticks([])

    # display grid for better readability and show the boxplot
    plt.grid(True)
    plt.show()


def plot_histogram(
    x: Union[pd.Series, np.ndarray, list],
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Optional[tuple] = (10, 6),
    bins: Optional[Union[list, np.ndarray]] = None,
    generate_bins: Optional[bool] = False,
    num_bins: Optional[int] = 20,
) -> None:
    """
    Plots a histogram with optional custom bins, labels, and grid.

    Args:
        x (Union[pd.Series, np.ndarray, list]): The data to be plotted in the histogram.
        title (str): The title of the plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        figsize (Optional[tuple], optional): The size of the figure (width, height). Default is (10, 6).
        bins (Optional[Union[list, np.ndarray]], optional): Custom bin edges. Default is None.
        generate_bins (Optional[bool], optional): If True, generates evenly spaced bins based on the data range. Default is False.
        num_bins (Optional[int], optional): The number of bins to generate if `generate_bins` is True. Default is 20.
    """
    plt.figure(figsize=figsize)

    # generate evenly spaced integer bins if requested
    if generate_bins:
        step_bins = (max(x) - min(x)) // num_bins
        bins = np.arange(min(x), max(x) + step_bins, step_bins)

    # plot histogram with specified or generated bins
    if isinstance(bins, (np.ndarray, list)):
        plt.hist(x, bins=bins, color="cornflowerblue", edgecolor="red", alpha=1)
        plt.xticks(ticks=bins, rotation=45)
    # plot histogram without specified bins
    else:
        plt.hist(x, color="cornflowerblue", edgecolor="red", alpha=1)

    # set labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # display grid for better readability and show the histogram
    plt.grid(True)
    plt.show()


def plot_correlaction_matrix(
    x: pd.DataFrame,
    feature_names: List[str],
    title: str,
    figsize: tuple = (12, 12),
    text_fontsize: int = 8,
) -> None:
    """
    Plots a correlation matrix with color mapping and displays correlation values inside each cell.

    Args:
        x (pd.DataFrame): The dataframe whose correlation matrix will be plotted.
        feature_names (List[str]): List of feature names for labeling the axes.
        title (str): The title of the correlation matrix plot.
        figsize (tuple, optional): The size of the figure (width, height). Default is (12, 12).
        text_fontsize (int, optional): Font size of the correlation values inside the heatmap. Default is 8.
    """

    # Compute correlation matrix
    corr_matrix = x.corr()

    plt.figure(figsize=figsize)
    # display the correlation matrix as a heatmap
    plt.imshow(corr_matrix, cmap="PuBuGn")

    plt.colorbar()
    plt.title(title)

    # set x and y axis ticks with feature names
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
    plt.yticks(range(len(feature_names)), feature_names)

    # display correlation values inside the heatmap
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            plt.text(
                j,
                i,
                f"{corr_matrix.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=text_fontsize,
            )

    plt.show()


def scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    figsize: Tuple[int, int] = (6, 6),
    marker_size: int = 4,
) -> None:
    """
    Plots a scatter plot of the given x and y data points.

    Args:
        x (np.ndarray): Array-like object containing x-axis data points.
        y (np.ndarray): Array-like object containing y-axis data points.
        title (str): Title of the scatter plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        figsize (Tuple[int, int], optional): Figure size in inches (width, height). Defaults to (6, 6).
        marker_size (int, optional): Size of the scatter plot markers. Defaults to 4.


    """
    # set figure size
    plt.figure(figsize=figsize)

    # create scatter plot with specified marker size
    plt.scatter(x=x, y=y, s=marker_size)

    # set plot title and axis labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # display the plot
    plt.show()
