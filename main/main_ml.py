import pandas as pd

from machine_learning.ml_utils import merge_tables
from machine_learning.ml_who import load_who_data, who_data_best_fpath, who_data_best_merged_fpath

def merge_who_best():
    original = load_who_data()
    new = pd.read_csv(who_data_best_fpath, sep=';')
    original_query_idx = 0
    col_query_idx = 0
    col_numeric_idx = 5
    merged = merge_tables(original, new, original_query_idx, col_query_idx, col_numeric_idx)
    merged.to_csv(who_data_best_merged_fpath)

if __name__ == '__main__':
    merge_who_best()
