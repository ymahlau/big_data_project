from pathlib import Path

import pandas as pd
from autogluon.tabular import TabularPredictor

from machine_learning.ml_utils import prepare_datasets, generate_indices, compute_splits, save_split, load_split, \
    remove_query

model_path = Path(__file__).parent / 'models'
model_path.mkdir(exist_ok=True, parents=True)
who_data_fpath = Path(__file__).parent.parent / 'data' / 'real_data' / 'Life_Expectancy_Data.csv'

QUERY_WHO = 'Country'
LABEL_WHO = 'Life expectancy '

def load_who_data() -> pd.DataFrame:
    df_who = pd.read_csv(who_data_fpath, sep=';')
    df_who = df_who[df_who.iloc[:, 3].notna()]  # target cannot be NaN
    return df_who

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
    time_limit = 600  # units are seconds
    retrain = True
    counter = 0
    save_dst = model_path / f'who_medium_{counter}'
    query = load_who_query()

    train_data, test_data, _, _ = load_split('who')
    train_data, test_data = remove_query(train_data, test_data, QUERY_WHO)

    if retrain:
        predictor = TabularPredictor(label=LABEL_WHO, path=save_dst, problem_type='regression')
        predictor.fit(train_data=train_data, time_limit=time_limit)
    else:
        predictor = TabularPredictor.load(str(save_dst))

    performance = predictor.evaluate(test_data)
    leaderboard = predictor.leaderboard(test_data)

    test_data_no_label = test_data.drop(columns=[LABEL_WHO])
    y_pred = predictor.predict(test_data_no_label)
    y_true = test_data[LABEL_WHO]

    print('Done')
