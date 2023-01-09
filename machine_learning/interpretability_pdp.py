from typing import Tuple, Union

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt


def calculate_ice(
        model: TabularPredictor,
        X: np.ndarray,
        s: int,
        example_input: pd.DataFrame,
):
    """
    Takes the input data and expands the dimensions from (num_instances, num_features) to (num_instances,
    num_instances, num_features). For the current instance i and the selected feature index s, the
    following equation is ensured: X_ice[i, :, s] == X[i, s].

    Parameters:
        model: Classifier which can call a predict method.
        X (np.array with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.

    Returns:
        X_ice (np.array with shape (num_instances, num_instances, num_features)): Changed input data w.r.t. x_s.
        y_ice (np.array with shape (num_instances, num_instances)): Predicted data.
    """

    num_instances, num_features = X.shape
    x_s = X[:, s]
    X_ice = X.repeat(num_instances).reshape((num_instances, num_features, -1)).transpose((2, 0, 1))
    for i in range(num_instances):
        X_ice[i, :, s] = x_s[i]

    # convert np array back to pd dataframe
    X_ice_flat = X_ice.reshape(-1, num_features)
    X_ice_df = pd.DataFrame(X_ice_flat, columns=example_input.columns).astype(dict(example_input.dtypes))
    y_ice_df = model.predict(X_ice_df)
    # y_ice[i] = model.predict(X_ice.reshape(-1, 3)).reshape((num_instances, num_instances))
    return X_ice, y_ice_df.values.reshape(num_instances, num_instances)


def prepare_ice(
        model: TabularPredictor,
        X: np.ndarray,
        s: int,
        example_input: pd.DataFrame,
        centered=False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses `calculate_ice` to retrieve plot data.

    Parameters:
        model: Classifier which can call a predict method.
        X (np.array with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used or not.

    Returns:
        all_x (list or 1D np.ndarray): List of lists of the x values.
        all_y (list or 1D np.ndarray): List of lists of the y values.
            Each entry in `all_x` and `all_y` represents one line in the plot.
    """
    num_instances, num_features = X.shape
    all_x, all_y = calculate_ice(model, X, s, example_input)
    x_s = X[:, s]
    order = np.argsort(x_s)

    all_x = x_s[order].repeat(num_instances).reshape((num_instances, num_instances))
    all_y = all_y[order].T
    all_x = all_x.T

    if centered:
        all_y -= np.reshape(all_y[:, 0], (-1, 1))

    return all_x, all_y


def plot_ice(
        model: TabularPredictor,
        X: np.ndarray,
        s: int,
        example_input: pd.DataFrame,
        centered=False
):
    """
    Creates a plot object and fills it with the content of `prepare_ice`.
    Note: `show` method is not called.

    Parameters:
        model: Classifier which can call a predict method.
        X (pd.DataFrame): data
        s (int): Index of the feature x_s.
        centered (bool): Whether c-ICE should be used or not.

    """
    all_x, all_y = prepare_ice(model, X, s, example_input, centered=centered)
    for i, data_tpl in enumerate(zip(all_x, all_y)):
        x_values, y_values = data_tpl
        if i == 0:
            plt.plot(x_values, y_values, alpha=0.2, color='grey', label='Individual Conditional Expectation')
        else:
            plt.plot(x_values, y_values, alpha=0.2, color='grey')


def prepare_pdp(
        model: TabularPredictor,
        X: np.ndarray,
        s: int,
        example_input: pd.DataFrame,
):
    """
    Uses `calculate_ice` to retrieve plot data for PDP.

    Parameters:
        model: Classifier which can call a predict method.
        X (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.

    Returns:
        x (list or 1D np.ndarray): x values of the PDP line.
        y (list or 1D np.ndarray): y values of the PDP line.
    """
    X_ice, y_ice = calculate_ice(model, X, s, example_input)
    x_s = X[:, s]
    order = np.argsort(x_s)
    x_s = x_s[order]

    y = np.mean(y_ice[order].T, axis=0)
    return x_s, y


def plot_pdp(
        model: TabularPredictor,
        X: np.ndarray,
        s: int,
        example_input: pd.DataFrame,
):
    """
    Creates a plot object and fills it with the content of `prepare_pdp`.
    Note: `show` method is not called.

    Parameters:
        model: Classifier which can call a predict method.
        data (np.ndarray with shape (num_instances, num_features)): Input data.
        s (int): Index of the feature x_s.

    Returns:
        plt (matplotlib.pyplot or utils.styled_plot.plt)
    """
    x, y = prepare_pdp(model, X, s, example_input)
    plt.plot(x, y, color='red', label='Partial Dependence')


def get_index_and_name(
        data: pd.DataFrame,
        idx_or_name: Union[int, str],
) -> Tuple[int, str]:
    if isinstance(idx_or_name, int):
        idx = idx_or_name
        name = data.columns[idx_or_name]
    elif isinstance(idx_or_name, str):
        idx = data.columns.get_loc(idx_or_name)
        name = idx_or_name
    else:
        raise ValueError('Unknown datatype for index/name')
    return idx, name

