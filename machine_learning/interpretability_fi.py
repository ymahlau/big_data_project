from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

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
    f = plt.figure(figsize=(4.5, 4.5), dpi=1000, frameon=False, layout='tight')
    plt.bar(x, height=importance, label=labels, color=color, yerr=y_err)
    plt.legend(fontsize='x-small')
    plt.xticks([])
    plt.xlabel('Feature')
    plt.ylabel('Performance-drop after shuffle')
    plt.title('Feature Importance')
    plt.show()
    f.savefig(image_path / f"{name}_importance.png", pad_inches=0.0)

    plt.clf()
    f = plt.figure(figsize=(4.5, 4.5), dpi=1000, frameon=False, layout='tight')
    plt.bar(x, height=p_value, label=labels, color=color)
    plt.legend(fontsize='x-small')
    plt.xticks([])
    plt.xlabel('Feature')
    plt.ylabel('p-Value')
    plt.title('p-Value for Null Hypothesis: importance == 0')
    plt.show()
    f.savefig(image_path / f"{name}_p_value.png", pad_inches=0.0)


def calc_feature_importance(
        model: TabularPredictor,
        data: pd.DataFrame,
        subsample_size: int = 10000,
        num_shuffle_sets: int = 20,
) -> pd.DataFrame:
    fi_df = model.feature_importance(data=data, subsample_size=subsample_size, num_shuffle_sets=num_shuffle_sets)
    return fi_df


