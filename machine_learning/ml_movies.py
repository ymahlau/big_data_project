from pathlib import Path

import pandas as pd

model_path = Path(__file__).parent / 'models'
model_path.mkdir(exist_ok=True, parents=True)
who_data_fpath = Path(__file__).parent.parent / 'data' / 'real_data' / 'movie_dataset.csv'

QUERY_MOVIES = 'Popularity'
LABEL_MOVIES = 'Title'

def load_movie_data() -> pd.DataFrame:
    df_movies = pd.read_csv(who_data_fpath, sep=',')
    df_movies = df_movies[df_movies[QUERY_MOVIES].notna()]  # target cannot be NaN
    df_movies = df_movies.iloc[:, [0, 1, 2, 3, 7, 8]]  # remove unnecessary columns
    return df_movies

def load_movie_query() -> pd.DataFrame:
    df_movie = load_movie_data()
    df_movie = df_movie.iloc[:, [2, 4]]
    return df_movie

if __name__ == '__main__':
    query = load_movie_query()
    print(query.shape)
