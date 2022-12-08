import sys
from pathlib import Path
from tqdm import tqdm

from typing import Callable, List

import duckdb
from datadiscoverybench.utils import load_dresden_db, load_git_tables_db

from utils.utils import (
    load_csvs_to_db,
    load_csv_to_dataframe,
    load_database,
    save_database,
)
from algorithms.qcr.qcr_sql import create_inverted_index, search_correlated


def main():
    """
    Loads database and if any valid arg is given computes the most joinable tables according to the qcr algorithm.
    Supports different statically defined load_modes.
    :param: args[0] = table to query with
            "-k <integer>" to set a new k for the top-k tables to find. Default: 10
            "-w <float> any float between 0 and 1 to manually determine the weighting between correlation and joinability. Default: 0.5
    """
    load_mode = ["web"]
    args = sys.argv[1:]
    load_index = True
    loader = None
    parts = None    
    con = duckdb.connect(database=":memory:")
    k_limit = 10
    weighting = 0.5


    for i in range(1,args.__len__()):                                    #gets integers stored behind k and w flags for later use
        if args[i] == "-k" and i+1 in range(args.__len__()):
            try:
                k_limit = int(args[i+1])
            except:
                print("Unable to parse value after '-k' as integer. Continue using default...")
            i+=1
        elif args[i] == "-w" and i+1 in range(args.__len__()):
            try:
                weighting = float(args[i+1])
            except:
                print("Unable to parse value after '-w' as float. Continue using default...")
            
            i+=1

    if args:
        args=args[0]

    if "debug" in load_mode:
        loader = load_csvs_to_db
        parts = [Path(__file__).parent / "toy_data"]
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
        print("""SPECIFY table to query with. Arguments have to be formatted\n
        like this: ~main.py <Path/Table> -k <Integer> -w <Float>\n
        For k use any Integer. For w choose a value between roughly 0.1 and 0.9\n
        Note that k and w values are optional.\n""")
        create_iterativ(con, loader, parts, "AllTables", "TermIndex", not load_index)
        save_database(con)
        
    if args:
        df = load_csv_to_dataframe(args)
        print(search_correlated(con, df, "TermIndex",k_limit, weighting))



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
