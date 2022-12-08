from pathlib import Path
from typing import Tuple, List

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from ml_utils import generate_indices

model_path = Path(__file__).parent / 'models'
model_path.mkdir(exist_ok=True, parents=True)
who_data_fpath = Path(__file__).parent / 'data' / 'Life_Expectancy_Data.csv'

def load_who_data() -> pd.DataFrame:
    df_who = pd.read_csv(who_data_fpath, sep=';')
    df_who = df_who[df_who.iloc[:, 3].notna()]  # target cannot be NaN
    # df_who = df_who[df_who.notnull().all(axis=1)]  # this would drop all NaN, but loses a lot of data
    return df_who

def compute_splits(data: pd.DataFrame) -> Tuple[TabularDataset, TabularDataset, str]:
    idx_train, idx_test = generate_indices(len(df))
    data_train, data_test = data.iloc[idx_train], data.iloc[idx_test]
    # remove country categorical value (query)
    data_train = TabularDataset(data_train.loc[:, data_train.columns != 'Country'])
    data_test = TabularDataset(data_test.loc[:, data_test.columns != 'Country'])
    lbl = 'Life expectancy '
    return data_train, data_test, lbl

def load_who_query() -> pd.DataFrame:
    df_who = load_who_data()
    query_with_duplicates = df_who.iloc[:, [0, 3]]
    return query_with_duplicates

if __name__ == '__main__':
    df = load_who_data()
    train_data, test_data, label = compute_splits(df)
    save_dst = model_path / 'who_medium'
    time_limit = 1200  # 20min is plenty
    retrain = True
    query = load_who_query()

    if retrain:
        predictor = TabularPredictor(label=label, path=save_dst, problem_type='regression')
        predictor.fit(train_data=train_data, time_limit=time_limit)
    else:
        predictor = TabularPredictor.load(str(save_dst))

    performance = predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data)

    test_data_no_label = test_data.drop(columns=[label])
    y_pred = predictor.predict(test_data_no_label)
    y_true = test_data[label]

    print('Done')
