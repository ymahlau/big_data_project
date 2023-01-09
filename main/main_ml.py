from pathlib import Path

import pandas as pd

from machine_learning.ml_utils import merge_tables
from machine_learning.ml_who import load_who_data, who_data_best_fpath, who_data_best_merged_fpath

def merge_who_best():
    original = load_who_data()
    original_query_idx = 0
    # new = pd.read_csv(who_data_best_fpath, sep=';')
    new = pd.read_parquet(Path(__file__).parent.parent / 'data' / 'zip_cache' / 'HDI_2015.parquet')
    col_query_idx = 0

    # Life Expectancy
    col_numeric_idx = 5
    merged = merge_tables(original, new, original_query_idx, col_query_idx, col_numeric_idx)

    # hdi score
    col_numeric_idx = 2
    merged = merge_tables(merged, new, original_query_idx, col_query_idx, col_numeric_idx)

    # percentage children
    col_numeric_idx = 20
    merged = merge_tables(merged, new, original_query_idx, col_query_idx, col_numeric_idx)

    # fertility rate
    col_numeric_idx = 22
    merged = merge_tables(merged, new, original_query_idx, col_query_idx, col_numeric_idx)

    # hdi rank
    col_numeric_idx = 1
    merged = merge_tables(merged, new, original_query_idx, col_query_idx, col_numeric_idx)

    merged.to_csv(Path(__file__).parent.parent / 'data' / 'results' / 'who_5.csv')

if __name__ == '__main__':
    merge_who_best()
