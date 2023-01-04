import math
import random
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd
from autogluon.core import TabularDataset

split_path = Path(__file__).parent / 'splits'
split_path.mkdir(parents=True, exist_ok=True)

# Select rows as 90%-train, 10%-test split
def generate_indices(n: int) -> Tuple[List[int], List[int]]:
    n_train = math.floor(0.9 * n)
    row_idx = list(range(n))
    row_idx_train = random.sample(row_idx, k=n_train)
    row_idx_test = list(set(row_idx) - set(row_idx_train))
    return row_idx_train, row_idx_test

def remove_query(
        data_train: pd.DataFrame,
        data_test: pd.DataFrame,
        query: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_train = data_train.loc[:, data_train.columns != query]
    data_test = data_test.loc[:, data_test.columns != query]
    return data_train, data_test

def compute_splits(
        data: pd.DataFrame,
        idx_train: List[int],
        idx_test: List[int],
        query: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_train, data_test = data.iloc[idx_train], data.iloc[idx_test]
    # remove country categorical value (query)
    if query is not None:
        data_train, data_test = remove_query(data_train, data_test, query)
    return data_train, data_test

def prepare_datasets(
        data: pd.DataFrame,
        query: str,
) -> Tuple[TabularDataset, TabularDataset]:
    idx_train, idx_test = generate_indices(len(data))
    data_train, data_test = compute_splits(data, idx_train, idx_test, query)
    dataset_train, dataset_test = TabularDataset(data_train), TabularDataset(data_test)
    return dataset_train, dataset_test

def save_split(
        name: str,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        label: str,
        query: str,
):
    cur_path = split_path / name
    cur_path.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(cur_path / 'train.csv')
    test_data.to_csv(cur_path / 'test.csv')
    with open(cur_path / 'label.txt', 'w') as f:
        f.write(label)
    with open(cur_path / 'query.txt', 'w') as f:
        f.write(query)

def merge_tables(
        original_table: pd.DataFrame,
        new_table: pd.DataFrame,
        original_query_idx,
        new_query_idx: int,
        new_numeric_idx: int,
) -> pd.DataFrame:
    new_table_reduced = new_table.iloc[:, [new_query_idx, new_numeric_idx]]
    original_query_name = original_table.columns[original_query_idx]
    new_query_name = new_table_reduced.columns[0]

    merged = original_table.merge(new_table_reduced, how='left', left_on=original_query_name, right_on=new_query_name)
    merged = merged.drop(new_query_name, axis=1)  # remove duplicate query
    return merged

def load_split(
        name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    cur_path = split_path / name
    train_data = pd.read_csv(cur_path / 'train.csv', index_col=0)
    test_data = pd.read_csv(cur_path / 'test.csv', index_col=0)
    with open(cur_path / 'label.txt', 'r') as f:
        label = f.read()
    with open(cur_path / 'query.txt', 'r') as f:
        query = f.read()
    return train_data, test_data, label, query
