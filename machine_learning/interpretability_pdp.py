import time
from pathlib import Path
from typing import Union, List, Optional

import plotly
from autogluon.tabular import TabularPredictor
from interpret import show
from interpret.blackbox import PartialDependence
from interpret.blackbox.partialdependence import PDPExplanation
from interpret.utils import gen_global_selector
from interpret.visual.plot import plot_line, plot_bar
from sklearn.base import BaseEstimator, RegressorMixin

from machine_learning.ml_utils import load_split, remove_query
from machine_learning.ml_who import QUERY_WHO, LABEL_WHO

import pandas as pd

model_path = Path(__file__).parent / 'models'
image_path = Path(__file__).parent / 'images' / 'pdp'
image_path.mkdir(exist_ok=True, parents=True)

class SKLearnWrapper(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            model: TabularPredictor,
            example_input: pd.DataFrame,
    ):
        self.model = model
        self.estimator_type = "regressor"
        self.columns = example_input.columns
        self.dtype_dict = dict(example_input.dtypes)

    def __sklearn_is_fitted__(self):
        return True

    def predict(self, X):
        x_df = pd.DataFrame(X, columns=self.columns).astype(self.dtype_dict)
        y_pred = self.model.predict(x_df)
        return y_pred.values

    def fit(self):
        pass


def explain_single(
        pdp: PartialDependence,
        idx: int,
        name: str,
        num_ice_samples: int,
) -> PDPExplanation:
    feature_type = pdp.feature_types[idx]
    explanation = PartialDependence._gen_pdp(
        pdp.data,
        pdp.predict_fn,
        idx,
        feature_type,
        num_points=pdp.num_points,
        std_coef=pdp.std_coef,
        num_ice_samples=num_ice_samples,
    )
    feature_dict = {
        "feature_values": explanation["values"],
        "scores": explanation["scores"],
        "upper_bounds": explanation["upper_bounds"],
        "lower_bounds": explanation["lower_bounds"],
    }
    internal_obj = {
        "overall": None,
        "specific": [explanation],
        "mli": [
            {"explanation_type": "pdp", "value": {"feature_list": [feature_dict]}},
            {"explanation_type": "density", "value": {"density": explanation["density"]}},
        ],
    }

    selector = gen_global_selector(
        pdp.data, pdp.feature_names, pdp.feature_types, None
    )

    return PDPExplanation(
        "global",
        internal_obj,
        feature_names=pdp.feature_names,
        feature_types=pdp.feature_types,
        name=name,
        selector=selector,
    )


def visualize_single(
        explanation: PDPExplanation,
        idx: int
) -> plotly.graph_objs.Figure:
    data_dict = explanation.data(0)
    feature_type = explanation.feature_types[idx]
    feature_name = explanation.feature_names[idx]
    if feature_type == "continuous":
        figure = plot_line(data_dict, title=feature_name)
    elif feature_type == "categorical":
        figure = plot_bar(data_dict, title=feature_name)
    else:
        raise Exception(f"Feature type {feature_type} is not supported.")

    figure["layout"]["yaxis1"].update(title="Average Response")
    return figure


def calc_pdp(
        model: TabularPredictor,
        data: pd.DataFrame,
        idx_or_name: Union[int, str],
        num_points: int = 10,
        num_ice_samples: int = 10,
):
    if isinstance(idx_or_name, int):
        idx = idx_or_name
        name = data.columns[idx_or_name]
    elif isinstance(idx_or_name, str):
        idx = data.columns.get_loc(idx_or_name)
        name = idx_or_name
    else:
        raise ValueError('Unknown datatype for index/name')

    data = data[data.notnull().all(axis=1)]  # pdp does not work with nan values
    wrapper = SKLearnWrapper(model, data[:1])
    pdp = PartialDependence(predict_fn=wrapper.predict, data=data, num_points=num_points)
    explanation = explain_single(pdp, idx, name, num_ice_samples=num_ice_samples)
    plotly_fig = visualize_single(explanation, idx)
    plotly_fig.write_image(image_path / f'{name}_{num_points}_{num_ice_samples}.pdf')

    # plotly_fig = pdp_single.visualize(0)
    # pdp_global = pdp.explain_global()
    # show(pdp_global)
    # time.sleep(99999)

def explain_who():
    train_data, test_data, _, _ = load_split('who')
    train_data, test_data = remove_query(train_data, test_data, QUERY_WHO)
    train_data, test_data = train_data.drop(columns=[LABEL_WHO]), test_data.drop(columns=[LABEL_WHO])

    model_dst = model_path / 'who_medium_0'
    model = TabularPredictor.load(str(model_dst))

    calc_pdp(model, test_data, 'Adult Mortality', num_points=25, num_ice_samples=100)

if __name__ == '__main__':
    explain_who()
