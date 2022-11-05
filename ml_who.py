import math
import random
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml_utils import train

who_data_fpath = Path(__file__).parent / 'data' / 'Life_Expectancy_Data.csv'

def load_who_data() -> pd.DataFrame:
    df_who = pd.read_csv(who_data_fpath)
    # exclude all rows that have at least one nan value
    df_filtered = df_who[df_who.notnull().all(axis=1)]
    # convert categorical 'developing' and 'developed' to numerical 0/1 flag
    df_filtered = df_filtered.replace('Developing', 0)
    df_filtered = df_filtered.replace('Developed', 1)
    return df_filtered

# Select rows as 80%-train, 10%-validation, 10%-test split
def generate_indices(n: int) -> Tuple[List[int], List[int], List[int]]:
    n_train, n_val = math.floor(0.8 * n), math.floor(0.1 * n)
    n_test = n - n_train - n_val
    row_idx = list(range(n))
    row_idx_train = random.sample(row_idx, k=n_train)
    remaining_idx = list(set(row_idx) - set(row_idx_train))
    row_idx_val = random.sample(remaining_idx, k=n_val)
    row_idx_test = list(set(remaining_idx) - set(row_idx_val))
    return row_idx_train, row_idx_val, row_idx_test


# Pytorch Dataset class
class WHODataset(Dataset):
    def __init__(self, data: pd.DataFrame, indices: List[int]):
        data_select = data.iloc[indices]
        self.y_df = data_select[['Life expectancy ']]  # life expectancy is the target
        self.Y = torch.tensor(self.y_df.values).squeeze().float()
        self.x_df = data_select.loc[:, data_select.columns != 'Life expectancy ']  # remove target
        self.x_df = self.x_df.loc[:, self.x_df.columns != 'Country']  # remove country categorical value (query)
        self.X = torch.tensor(self.x_df.values).float()

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.X.shape[0]


def build_loaders(
        batch_size: int = 64,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    df_who = load_who_data()
    n = len(df_who)
    idx_train, idx_val, idx_test = generate_indices(n)

    data_train = WHODataset(df_who, idx_train)
    data_val = WHODataset(df_who, idx_val)
    data_test = WHODataset(df_who, idx_test)

    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)
    return loader_train, loader_val, loader_test

if __name__ == '__main__':
    train_loader, val_loader, test_loader = build_loaders()
    input_size = 20
    output_size = 1
    hidden_size = 10
    name = 'who_normal'
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.ReLU(),
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        name=name,
    )


