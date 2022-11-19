import sys
from pathlib import Path
from tqdm import tqdm

from typing import Callable, List

import duckdb
from datadiscoverybench.utils import load_dresden_db, load_git_tables_db

from utils import (
    load_csvs_to_db,
    load_csv_to_dataframe,
    load_dataframes_to_db,
    load_database,
    save_database,
    query_db,
)
from qcr_sql import create_inverted_index, search_correlated


def main():
    load_mode = ["web"]
    args = sys.argv[1:]
    load_index = True
    loader = None
    parts = None

    con = duckdb.connect(database=":memory:")

    if "debug" in load_mode:
        loader = load_csvs_to_db
        parts = [Path(__file__).parent / "toy_tables"]
    if "web" in load_mode:
        loader = load_dresden_db
        parts = [1, 2, 3, 4, 5]
    if "git" in load_mode:
        loader = load_git_tables_db
        parts = [
            "allegro_con_spirito_tables_licensed",
            "abstraction_tables_licensed",
        ]

    if load_index or args:
        load_database(con)

    if not args:
        create_iterativ(con, loader, parts, "AllTables", "TermIndex", not load_index)
        save_database(con)

    if args:
        df = load_csv_to_dataframe(args[0])
        print(search_correlated(con, df, "TermIndex"))


def create_iterativ(
    con: duckdb.DuckDBPyConnection,
    loader: Callable[[duckdb.DuckDBPyConnection, List], None],
    parts: List[List[any]],
    table_name: str,
    index_name: str,
    new: bool = True,
):
    if new:
        con.execute(f"DROP TABLE IF EXISTS {index_name}")

    for part in tqdm(parts):
        loader(con, parts=[part])
        create_inverted_index(con, table_name, index_name, union_old=True)
        con.execute(f"DROP TABLE {table_name}")


if __name__ == "__main__":
    main()
