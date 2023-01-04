from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from machine_learning.ml_utils import prepare_datasets, generate_indices, compute_splits, save_split, load_split, \
    remove_query, merge_tables

model_path = Path(__file__).parent / 'models'
model_path.mkdir(exist_ok=True, parents=True)
who_data_fpath = Path(__file__).parent.parent / 'data' / 'real_data' / 'Life_Expectancy_Data.csv'
who_data_best_fpath = Path(__file__).parent.parent / 'data' / 'results' / 'who_best_query.csv'
who_data_best_merged_fpath = Path(__file__).parent.parent / 'data' / 'merged' / 'who_best_query.csv'

QUERY_WHO = 'Country'
LABEL_WHO = 'Life expectancy '

def load_who_data() -> pd.DataFrame:
    df_who = pd.read_csv(who_data_fpath, sep=';')
    df_who = df_who[df_who[LABEL_WHO].notna()]  # target cannot be NaN
    return df_who

def load_who_grouped() ->pd.DataFrame:
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

if __name__ == '__main__':
    time_limit = 1200  # units are seconds
    retrain = True
    counter = 0
    save_dst = model_path / f'who_grouped_best_no_mort_{counter}'
    query = load_who_query()
    presets = 'best_quality'

    # who_data = load_who_data()
    # idx_train, idx_test = generate_indices(len(who_data))
    # data_train, data_test = compute_splits(who_data, idx_train, idx_test)
    # save_split('who', data_train, data_test, LABEL_WHO, QUERY_WHO)
    # train_data, test_data, _, _ = load_split('who')

    # who_grouped = load_who_grouped()
    # idx_train, idx_test = generate_indices(len(who_grouped))
    # data_train, data_test = compute_splits(who_grouped, idx_train, idx_test)
    # save_split('who_grouped', data_train, data_test, LABEL_WHO, QUERY_WHO)
    # train_data, test_data, _, _ = load_split('who_grouped')

    best_who_query = pd.read_csv(who_data_best_fpath, sep=';')
    # train_data, test_data, _, _ = load_split('who')
    train_data, test_data, _, _ = load_split('who_grouped')
    train_data = merge_tables(train_data, best_who_query, 0, 0, 5)
    test_data = merge_tables(test_data, best_who_query, 0, 0, 5)

    train_data, test_data = remove_query(train_data, test_data, QUERY_WHO)
    train_data, test_data = remove_query(train_data, test_data, 'Adult Mortality')
    if retrain:
        predictor = TabularPredictor(label=LABEL_WHO, path=save_dst, problem_type='regression')
        predictor.fit(train_data=train_data, time_limit=time_limit, presets=presets)
    else:
        predictor = TabularPredictor.load(str(save_dst))

    performance = predictor.evaluate(test_data)
    print(performance)
    leaderboard = predictor.leaderboard(test_data)
    print(leaderboard)

    # test_data_no_label = test_data.drop(columns=[LABEL_WHO])
    # y_pred = predictor.predict(test_data_no_label)
    # y_true = test_data[LABEL_WHO]

    print('Done')
