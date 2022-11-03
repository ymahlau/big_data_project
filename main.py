import duckdb
import pandas as pd
import numpy as np
from datadiscoverybench.utils import load_dresden_db, load_git_tables_db
import time

from qcr_sql import create_inverted_index


def main():
    con = duckdb.connect(database=':memory:')
    # load_dresden_db(con, parts=[0,1])
    load_git_tables_db(con, parts=['allegro_con_spirito_tables_licensed', "abstraction_tables_licensed"])

    start = time.time()
    create_inverted_index(con)
    query = "select * from TermIndex limit 10"
    print(time.time() - start)

    query_db(con, query)
    print(time.time() - start)



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




if __name__ == '__main__':
    main()