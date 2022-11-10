from typing import List
from pathlib import Path

import pandas as pd
import numpy as np
import duckdb

def load_test_data() -> List[pd.DataFrame]:
    paths = [
        Path(__file__).parent / 'data' / 'test_data_00.csv',
        Path(__file__).parent / 'data' / 'test_data_01.csv',
        Path(__file__).parent / 'data' / 'test_data_02.csv',
        Path(__file__).parent / 'data' / 'test_data_03.csv',
    ]

    df_list = [pd.read_csv(path, sep=';') for path in paths]
    return df_list


def load_dataframe(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=';')

def load_dataframe_to_db(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, view_name: str) -> None:
    d = {'CellValue': [], 'TableId': [], 'ColumnId': [], 'RowId': []}
    
    for (row_id, column_id), value in np.ndenumerate(df.values):
        d["CellValue"].append(str(value))
        d["TableId"].append("")
        d["ColumnId"].append(column_id)
        d["RowId"].append(row_id)
    
    transformed = pd.DataFrame(data=d)
    
    con.register(view_name, transformed)


def query_db(con: duckdb.DuckDBPyConnection, query: str) -> pd.core.frame.DataFrame:
    """
    query db
    :param query: what query to execute
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :return: pandas dataframe with resulting table
    """
    df = con.execute(query).fetch_df()
    print(df)
    return df
