from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from machine_learning.ml_utils import load_split, remove_query, merge_tables
from machine_learning.ml_who import QUERY_WHO, LABEL_WHO, model_path, who_data_best_fpath

image_path = Path(__file__).parent / 'images' / 'fi'
image_path.mkdir(exist_ok=True, parents=True)

def plot_feature_importance(
        fi_df,
        name: str,
):
    labels = list(fi_df.index)
    x = np.arange(len(labels)) + 0.5
    importance = fi_df.loc[:, 'importance'].values
    y_err = fi_df.loc[:, 'stddev'].values
    p_value = fi_df.loc[:, 'p_value'].values

    clist = [(0, "red"), (0.125, "red"), (0.25, "orange"), (0.5, "green"),
             (0.7, "green"), (0.75, "blue"), (1, "blue")]
    rvb = mcolors.LinearSegmentedColormap.from_list("", clist)
    color = rvb(x / len(labels))

    plt.clf()
    f = plt.figure()
    plt.bar(x, height=importance, label=labels, color=color, yerr=y_err)
    plt.legend(fontsize='xx-small')
    plt.xticks([])
    plt.xlabel('Feature')
    plt.ylabel('Performance-drop after shuffle')
    plt.title('Feature Importance')
    plt.show()
    f.savefig(image_path / f"{name}_importance.pdf")

    plt.clf()
    f = plt.figure()
    plt.bar(x, height=p_value, label=labels, color=color)
    plt.legend(fontsize='xx-small')
    plt.xticks([])
    plt.xlabel('Feature')
    plt.ylabel('p-Value')
    plt.title('p-Value for Null Hypothesis: importance == 0')
    plt.show()
    f.savefig(image_path / f"{name}_p_value.pdf")


def calc_feature_importance(
        model: TabularPredictor,
        data: pd.DataFrame,
        subsample_size: int = 5000,
        num_shuffle_sets: int = 10,
) -> pd.DataFrame:
    fi_df = model.feature_importance(data=data, subsample_size=subsample_size, num_shuffle_sets=num_shuffle_sets)
    return fi_df


if __name__ == '__main__':
    best_who_query = pd.read_csv(who_data_best_fpath, sep=';')
    train_data, test_data, _, _ = load_split('who_grouped')
    train_data = merge_tables(train_data, best_who_query, 0, 0, 5)
    test_data = merge_tables(test_data, best_who_query, 0, 0, 5)

    train_data, test_data = remove_query(train_data, test_data, QUERY_WHO)
    train_data, test_data = remove_query(train_data, test_data, 'Adult Mortality')

    model_name = 'who_grouped_best_no_mort_0'
    model_dst = model_path / model_name
    model = TabularPredictor.load(str(model_dst))

    fi_df = calc_feature_importance(model, test_data)
    plot_feature_importance(fi_df, model_name)
