from typing import List

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy import stats

import utils


def naive_search(
        query_df: pd.DataFrame,
        input_col_name: str,
        target_col_name: str,
        data: List[pd.DataFrame],
        join_threshold: float = 0.0,
) -> np.ndarray:
    input_col = query_df[input_col_name]
    target_col = query_df[target_col_name]

    score_arr = np.zeros(shape=(len(data), 2))  # join and corr score per df

    for i, df in enumerate(data):
        # joinability score
        best_col_name = None
        best_col_score = -1
        for col_name in df.columns:
            col = df[col_name]
            join_score = len(set(col).intersection(set(input_col))) / len(set(input_col))

            if join_score > best_col_score:
                best_col_name = col_name
                best_col_score = join_score
        score_arr[i, 0] = best_col_score

        # correlation score
        if best_col_score > join_threshold:
            joined_table = pd.merge(query, df, left_on=input_col_name, right_on=best_col_name)
            target = joined_table[target_col_name]
            best_corr_score = -2.
            best_col_name = None

            for col_name in df.columns:
                if not is_numeric_dtype(df[col_name]) or col_name in [target_col_name, input_col_name]:
                    continue

                corr_score, _ = stats.pearsonr(target, joined_table[col_name])
                if corr_score > best_corr_score:
                    best_corr_score = corr_score
                    best_col_name = col_name

            score_arr[i, 1] = best_corr_score
    return score_arr


if __name__ == '__main__':
    df_list = utils.load_test_data()
    query = df_list[0]
    input_col_id = 0

    score_arr = naive_search(
        query_df=query,
        input_col_name='Country',
        target_col_name='Seats in European Parliament',
        data=df_list[1:],
    )
