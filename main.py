import sys
import time

import duckdb
from datadiscoverybench.utils import load_dresden_db, load_git_tables_db

from utils import load_dataframe, load_dataframe_to_db
from qcr_sql import create_inverted_index, search_inverted_index



def main():
    con = duckdb.connect(database=':memory:')
    # load_dresden_db(con, parts=[0,1])
    load_git_tables_db(con, parts=['allegro_con_spirito_tables_licensed', "abstraction_tables_licensed"])

    if len(sys.argv) < 1:
        create_inverted_index(con, "AllTables", "TermIndex")
    else:
        df = load_dataframe(sys.argv[1])
        load_dataframe_to_db(con, df)
        search_inverted_index(con, "Query", "TermIndex")



if __name__ == '__main__':
    main()