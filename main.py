import sys
from pathlib import Path

import duckdb
from datadiscoverybench.utils import load_dresden_db, load_git_tables_db

from utils import load_csvs_to_db, load_csv_to_dataframe, load_dataframes_to_db, load_database, save_database, query_db
from qcr_sql import create_inverted_index, search_correlated


def main():
    load_mode = ["debug"]
    args = sys.argv[1:]
    build_index = None

    con = duckdb.connect(database=':memory:')

    if "load" in load_mode:
        load_database(con)
    else:
        if "debug" in load_mode:
            load_csvs_to_db(con, Path(__file__).parent / 'toy_tables', "AllTables")
        if "dresden" in load_mode:
            load_dresden_db(con, parts=[0, 1])
        if "git" in load_mode:
            load_git_tables_db(con, parts=['allegro_con_spirito_tables_licensed', "abstraction_tables_licensed"])

    if (len(args) <= 0 or build_index) and build_index is not False:
        create_inverted_index(con, "AllTables", "TermIndex")
    if len(args) >= 1:
        df = load_csv_to_dataframe(args[0])
        print(search_correlated(con, df, "TermIndex"))

    save_database(con)


if __name__ == '__main__':
    main()
