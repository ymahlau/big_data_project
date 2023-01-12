import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from matplotlib import pyplot as plt
from tqdm import tqdm

from machine_learning.interpretability_fi import calc_feature_importance, plot_feature_importance
from machine_learning.interpretability_pdp import plot_ice, plot_pdp, prepare_ice, prepare_pdp
from machine_learning.ml_utils import prepare_datasets, generate_indices, compute_splits, save_split, load_split, \
    remove_query, merge_tables

model_path = Path(__file__).parent / 'models'
model_path.mkdir(exist_ok=True, parents=True)

who_data_fpath = Path(__file__).parent.parent / 'data' / 'real_data' / 'Life_Expectancy_Data.csv'

image_path = Path(__file__).parent / 'images'
image_path.mkdir(exist_ok=True, parents=True)

QUERY_WHO = 'Country'
LABEL_WHO = 'Life expectancy '

def load_who_data() -> pd.DataFrame:
    df_who = pd.read_csv(who_data_fpath, sep=';')
    df_who = df_who[df_who[LABEL_WHO].notna()]  # target cannot be NaN
    return df_who

def load_who_grouped() -> pd.DataFrame:
    df_who = pd.read_csv(who_data_fpath, sep=';')
    df_who_grouped = df_who.groupby(QUERY_WHO).mean(numeric_only=False).reset_index()
    df_who_grouped = df_who_grouped[df_who_grouped[LABEL_WHO].notna()]
    return df_who_grouped

def save_who_split():
    # save who splits
    df = load_who_data()
    idx_train, idx_test = generate_indices(len(df))
    train_df, test_df = compute_splits(df, idx_train, idx_test)
    save_split('who', train_df, test_df, LABEL_WHO, QUERY_WHO)
    # train_data_n, test_data_n, label_n, query_n = load_split('who')


def load_who_query() -> pd.DataFrame:
    df_who = load_who_data()
    query_with_duplicates = df_who.iloc[:, [0, 3]]
    return query_with_duplicates


def load_train_test_data(
        grouped: bool,
        num_added_cols: int,
        ignore_first: int = 0,
):
    who_with_new_cols = pd.read_csv(Path(__file__).parent.parent / 'data' / 'results' / 'who_5.csv', index_col=0)
    if not grouped:
        train_data, test_data, _, _ = load_split('who')
    else:
        train_data, test_data, _, _ = load_split('who_grouped')

    for i in range(-5 + ignore_first, -5 + num_added_cols):
        train_data = merge_tables(train_data, who_with_new_cols, 0, 0, i)
        test_data = merge_tables(test_data, who_with_new_cols, 0, 0, i)

    train_data, test_data = remove_query(train_data, test_data, QUERY_WHO)
    train_data, test_data = remove_query(train_data, test_data, 'Adult Mortality')

    return train_data, test_data


def train_models():
    time_limit = 1200  # units are seconds
    presets = 'best_quality'

    for grouped in [True, False]:
        for num_added_cols in range(6):
            train_data, test_data = load_train_test_data(grouped, num_added_cols)
            group_str = 'grouped' if grouped else 'no_group'
            save_dst = model_path / f'who_{group_str}_{num_added_cols}'
            predictor = TabularPredictor(label=LABEL_WHO, path=save_dst, problem_type='regression')
            predictor.fit(train_data=train_data, time_limit=time_limit, presets=presets)
            return


def train_models_actual():
    time_limit = 1200  # units are seconds
    presets = 'best_quality'

    for grouped in [True, False]:
        for num_added_cols in [3, 4]:
            train_data, test_data = load_train_test_data(grouped, num_added_cols, ignore_first=2)
            group_str = 'grouped' if grouped else 'no_group'
            save_dst = model_path / f'who_actual_{group_str}_{num_added_cols-2}'
            predictor = TabularPredictor(label=LABEL_WHO, path=save_dst, problem_type='regression')
            predictor.fit(train_data=train_data, time_limit=time_limit, presets=presets)
            return


def model_test_performance():
    result_dict = defaultdict(list)

    for grouped in [True, False]:
        for num_added_cols in range(6):
            train_data, test_data = load_train_test_data(grouped, num_added_cols)
            group_str = 'grouped' if grouped else 'no_group'
            save_dst = model_path / f'who_{group_str}_{num_added_cols}'
            predictor = TabularPredictor.load(str(save_dst))

            performance = predictor.evaluate(test_data)
            result_dict['index'].append(f'who_{group_str}_{num_added_cols}')
            for k, v in performance.items():
                result_dict[k].append(v)
    print(result_dict)
    result_df = pd.DataFrame(data=result_dict)
    result_df.to_csv(Path(__file__).parent / 'results' / 'performance.csv')
    # leaderboard = predictor.leaderboard(test_data)
    # print(leaderboard)

def model_test_performance_actual():
    result_dict = defaultdict(list)

    for grouped in [True, False]:
        for num_added_cols in [0, 3, 4]:
            train_data, test_data = load_train_test_data(grouped, num_added_cols, ignore_first=2)
            group_str = 'grouped' if grouped else 'no_group'
            if num_added_cols == 0:
                save_dst = model_path / f'who_{group_str}_0'
            else:
                save_dst = model_path / f'who_actual_{group_str}_{num_added_cols-2}'
            predictor = TabularPredictor.load(str(save_dst))

            performance = predictor.evaluate(test_data)
            if num_added_cols == 0:
                result_dict['index'].append(f'who_actual_{group_str}_0')
            else:
                result_dict['index'].append(f'who_actual_{group_str}_{num_added_cols-2}')
            for k, v in performance.items():
                result_dict[k].append(v)
    print(result_dict)
    result_df = pd.DataFrame(data=result_dict)
    result_df.to_csv(Path(__file__).parent / 'results' / 'performance_actual.csv')

def load_performance_df():
    result_df = pd.read_csv(Path(__file__).parent / 'results' / 'performance.csv')
    return result_df

def load_performance_df_actual():
    result_df = pd.read_csv(Path(__file__).parent / 'results' / 'performance_actual.csv')
    return result_df

def load_model(
        grouped: bool,
        num_added_cols: int,
) -> TabularPredictor:
    group_str = 'grouped' if grouped else 'no_group'
    save_dst = model_path / f'who_{group_str}_{num_added_cols}'
    predictor = TabularPredictor.load(str(save_dst))
    return predictor

def calc_fi_all():
    for grouped in [True, False]:
        for num_added_cols in range(6):
            train_data, test_data = load_train_test_data(grouped, num_added_cols)
            model = load_model(grouped, num_added_cols)
            group_str = 'grouped' if grouped else 'no_group'
            model_name = f'who_{group_str}_{num_added_cols}'

            fi_df = calc_feature_importance(model, test_data)
            fi_df.to_csv(Path(__file__).parent / 'results' / f'fi_{model_name}.csv')

def calc_fi_actual():
    for grouped in [True, False]:
        for num_added_cols in [3, 4]:
            train_data, test_data = load_train_test_data(grouped, num_added_cols, ignore_first=2)
            group_str = 'grouped' if grouped else 'no_group'
            model_name = f'who_actual_{group_str}_{num_added_cols-2}'
            save_dst = model_path / model_name
            model = TabularPredictor.load(str(save_dst))

            fi_df = calc_feature_importance(model, test_data)
            fi_df.to_csv(Path(__file__).parent / 'results' / f'fi_{model_name}.csv')


def plot_fi_all():
    for grouped in [True, False]:
        for num_added_cols in range(6):
            group_str = 'grouped' if grouped else 'no_group'
            model_name = f'who_{group_str}_{num_added_cols}'

            fi_df = pd.read_csv(Path(__file__).parent / 'results' / f'fi_{model_name}.csv', index_col=0)
            plot_feature_importance(fi_df, model_name)


def plot_fi_actual():
    for grouped in [True, False]:
        for num_added_cols in [3, 4]:
            group_str = 'grouped' if grouped else 'no_group'
            model_name = f'who_actual_{group_str}_{num_added_cols-2}'

            fi_df = pd.read_csv(Path(__file__).parent / 'results' / f'fi_{model_name}.csv', index_col=0)
            plot_feature_importance(fi_df, model_name)


def plot_performance():
    performance_df = pd.read_csv(Path(__file__).parent / 'results' / f'performance.csv', index_col=1)
    performance_df = performance_df.drop(columns=performance_df.columns[0])

    for m, metric in enumerate(performance_df.columns):
        for grouped in [True, False]:
            group_str = 'grouped' if grouped else 'no_group'

            x = list(range(6))
            y_list = []
            labels = [f'{i}' for i in range(6)]

            for num_added_cols in range(6):
                model_name = f'who_{group_str}_{num_added_cols}'
                value = - performance_df[metric][model_name]
                if m == 3 or m == 4:
                    value *= -1
                y_list.append(value)

            plt.clf()
            f = plt.figure(figsize=(4.5, 4.5), dpi=1000, frameon=False, layout='tight')
            plt.bar(x, height=y_list, label=labels, color='lightblue')
            plt.xlabel('Number of columns added')
            plt.ylabel(metric)
            plt.title('Performance')
            plt.show()
            f.savefig(image_path / 'performance' / f"who_{group_str}_{metric}.png", pad_inches=0.0)


def plot_performance_actual():
    performance_df = pd.read_csv(Path(__file__).parent / 'results' / f'performance_actual.csv', index_col=1)
    performance_df = performance_df.drop(columns=performance_df.columns[0])

    for m, metric in enumerate(performance_df.columns):
        for grouped in [True, False]:
            group_str = 'grouped' if grouped else 'no_group'

            x = list(range(3))
            y_list = []
            labels = [f'{i}' for i in range(3)]

            for num_added_cols in [0, 1, 2]:
                model_name = f'who_actual_{group_str}_{num_added_cols}'
                value = - performance_df[metric][model_name]
                if m == 3 or m == 4:
                    value *= -1
                y_list.append(value)

            plt.clf()
            f = plt.figure(figsize=(4.5, 4.5), dpi=1000, frameon=False, layout='tight')
            plt.xticks([0, 1, 2])
            plt.bar(x, height=y_list, label=labels, color='lightblue')
            plt.xlabel('Number of columns added')
            plt.ylabel(metric)
            plt.title('Performance')
            f.savefig(image_path / 'performance' / f"who_actual_{group_str}_{metric}.png", pad_inches=0.0)


def calc_pdp_ice_life_exp():
    for grouped in [True, False]:
        for num_added_cols in tqdm(range(1, 6)):
            train_data, test_data = load_train_test_data(grouped, num_added_cols)
            model = load_model(grouped, num_added_cols)
            group_str = 'grouped' if grouped else 'no_group'
            model_name = f'who_{group_str}_{num_added_cols}'

            all_x, all_y = prepare_ice(model, test_data.values, test_data.columns.get_loc('Life expectancy at birth'),
                                       train_data, centered=False)
            x, y = prepare_pdp(model, test_data.values, test_data.columns.get_loc('Life expectancy at birth'),
                               train_data)
            result_dict = {'x': x, 'y': y, 'all_x': all_x, 'all_y': all_y}
            with open(Path(__file__).parent / 'results' / f'pdp_{model_name}.pickle', 'wb') as f:
                pickle.dump(result_dict, f)

def plot_pdp_life_exp():
    for grouped in [True, False]:  # [True, False]:
        for num_added_cols in range(1, 6):  # range(1, 6):
            for use_ice in [True, False]:
                group_str = 'grouped' if grouped else 'no_group'
                ice_str = 'ice' if use_ice else 'no_ice'
                model_name = f'who_{group_str}_{num_added_cols}'

                with open(Path(__file__).parent / 'results' / f'pdp_{model_name}.pickle', 'rb') as f:
                    result_dict = pickle.load(f)
                x, y, all_x, all_y = result_dict['x'], result_dict['y'], result_dict['all_x'], result_dict['all_y']

                f = plt.figure(figsize=(4.5, 4.5), dpi=1000, frameon=False, layout='tight')
                if use_ice:
                    for i, data_tpl in enumerate(zip(all_x, all_y)):
                        x_values, y_values = data_tpl
                        if i == 0:
                            plt.plot(x_values, y_values, alpha=0.2, color='grey',
                                     label='Individual Conditional Expectation')
                        else:
                            plt.plot(x_values, y_values, alpha=0.2, color='grey')
                plt.plot(x, y, color='red', label='Partial Dependence')

                plt.xlabel('Life Expectancy at Birth (feature)')
                plt.ylabel('Life Expectancy (Model prediction)')
                plt.title("Partial Dependence Plot")
                if use_ice:
                    plt.legend()

                f.savefig(image_path / 'pdp' / f"who_le_{group_str}_{num_added_cols}_{ice_str}.png", pad_inches=0.0)


if __name__ == '__main__':
    # train_models()
    # model_test_performance()

    # train_models_actual()
    # model_test_performance_actual()
    # plot_performance_actual()

    # x = load_performance_df_actual()
    # print(x)

    # calc_fi_all()
    # plot_fi_all()

    # calc_fi_actual()
    plot_fi_actual()

    # plot_performance()

    # calc_pdp_ice_life_exp()
    # plot_pdp_life_exp()

    print('Done')
