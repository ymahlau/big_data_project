from typing import List

import pandas as pd
from pathlib import Path

def load_test_data() -> List[pd.DataFrame]:
    paths = [
        Path(__file__).parent / 'data' / 'test_data_00.csv',
        Path(__file__).parent / 'data' / 'test_data_01.csv',
        Path(__file__).parent / 'data' / 'test_data_02.csv',
        Path(__file__).parent / 'data' / 'test_data_03.csv',
    ]

    df_list = [pd.read_csv(path, sep=';') for path in paths]
    return df_list

