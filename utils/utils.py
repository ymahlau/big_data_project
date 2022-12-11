from typing import List, Union
from pathlib import Path

import pandas as pd
import numpy as np
import duckdb


def load_csvs_to_dataframes(path: Path) -> List[pd.DataFrame]:
    """
    load all csv files in a directory into a list of pandas dataframes
    :param path: path to directory containing csv files
    :return: list of pandas dataframes
    """
    frames = []
    for file in path.iterdir():
        if file.suffix == ".csv":
            df = load_csv_to_dataframe(file)
            frames.append(df)

    return frames


def load_csvs_to_db(
    con: duckdb.DuckDBPyConnection,
    parts=[Path(__file__).parent / "toy_data"],
    big_table_name="AllTables",
) -> None:
    """
    load all csv files in a directory into a database
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :param path: path to directory containing csv files
    :param big_table_name: name of the table to create containing all tables
    """
    frames = load_csvs_to_dataframes(parts[0])  # Todo: Iterate over all parts
    load_dataframes_to_db(con, frames, big_table_name)


def load_csv_to_dataframe(path: Union[Path, str]) -> pd.DataFrame:
    """
    load a csv file into a pandas dataframe
    :param path: path to csv file
    :return: pandas dataframe
    """
    path = Path(path)
    df = pd.read_csv(path, sep=";")
    df.columns.name = path.stem

    return df


def load_dataframes_to_db(
    con: duckdb.DuckDBPyConnection, frames: List[pd.DataFrame], table_name: str
) -> None:
    """
    load a list of dataframes into a database
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :param frames: list of dataframes to load
    :param table_name: name of the table to create containing all tables
    """
    d = {"CellValue": [], "TableId": [], "ColumnId": [], "RowId": []}
    for df in frames:
        for (row_id, column_id), value in np.ndenumerate(df.values):
            d["CellValue"].append(str(value))
            d["TableId"].append(df.columns.name)
            d["ColumnId"].append(column_id)
            d["RowId"].append(row_id)

    transformed = pd.DataFrame(data=d)

    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * from transformed")


def save_database(con: duckdb.DuckDBPyConnection, path: str = "data/database") -> None:
    """
    save a table to disk
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :param path: path to save the database to
    """
    con.execute(f"EXPORT DATABASE '{path}' (FORMAT PARQUET)")


def load_database(con: duckdb.DuckDBPyConnection, path: str = "data/database") -> None:
    """
    load a database from disk
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :param path: path to load the database from
    """
    if Path(path).exists():
        con.execute(f"IMPORT DATABASE '{path}'")


def query_db(con: duckdb.DuckDBPyConnection, query: str) -> pd.DataFrame:
    """
    query the database and print and return the result as a pandas dataframe
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :param query: what query to execute
    :return: pandas dataframe with resulting table
    """
    df = con.execute(query).fetch_df()
    print(df)
    return df


def fetch_table(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.core.frame.DataFrame:
    """
    query db for given table name and transform columnar representation ['CellValue', 'TableId', 'ColumnId', 'RowId']
    to actual tabular format. Ignores column headers contained in data.
    :param table_name: identifier of table
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :return: pandas dataframe with actual columns and rows
    author: ML
    """
    query = f"""
        SELECT * 
        FROM AllTables
        WHERE TableId == '{table_name}'

        """
    df = con.execute(query).fetch_df()
    print()
    print(df['RowId'].max() + 1, "columns")
    print(df['ColumnId'].max() + 1, "rows")
    grouped_table = df.sort_values(by=['TableId', 'RowId', 'ColumnId']).groupby(
        ['TableId', 'RowId']).CellValue.apply(list).reset_index()
    tbl_dict = grouped_table.drop(['TableId'], axis=1).set_index('RowId').to_dict()['CellValue']
    df_tbl = pd.DataFrame.from_dict(tbl_dict, orient='index')
    df_tbl.columns = [str(x) for x in np.arange(len(df_tbl.columns.values))]
    print(table_name)
    print(df_tbl)

    return df_tbl

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df_obj = df.select_dtypes(['object'])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip().str.lower())
    return df
