from pathlib import Path
from typing import Tuple

import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor

from ml_utils import generate_indices

model_path = Path(__file__).parent / 'models'
model_path.mkdir(exist_ok=True, parents=True)
who_data_fpath = Path(__file__).parent / 'data' / 'song_data.csv'

def load_song_data() -> pd.DataFrame:
    df_song = pd.read_csv(who_data_fpath, sep=',')
    return df_song

def load_song_query() -> pd.DataFrame:
    df_song = load_song_data()
    query = df_song.iloc[:, [0, 1]]
    return query

def compute_splits(data: pd.DataFrame) -> Tuple[TabularDataset, TabularDataset, str]:
    idx_train, idx_test = generate_indices(len(df))
    data_train, data_test = data.iloc[idx_train], data.iloc[idx_test]
    # remove country categorical value (query)
    data_train = TabularDataset(data_train.iloc[:, 1:])
    data_test = TabularDataset(data_test.iloc[:, 1:])
    lbl = 'song_popularity'
    return data_train, data_test, lbl

if __name__ == '__main__':
    df = load_song_data()
    query = load_song_query()
    train_split, val_split, label = compute_splits(df)
    save_dst = model_path / 'song_medium'
    time_limit = 1200
    retrain = True

    if retrain:
        predictor = TabularPredictor(label=label, path=save_dst, problem_type='regression')
        predictor.fit(train_data=train_split, time_limit=time_limit)
    else:
        predictor = TabularPredictor.load(str(save_dst))

    performance = predictor.evaluate(val_split)
    leaderboard = predictor.leaderboard(val_split)

    test_data_no_label = val_split.drop(columns=[label])
    y_pred = predictor.predict(test_data_no_label)
    y_true = val_split[label]

    print('Done')

