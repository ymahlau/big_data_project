import duckdb
import pandas as pd
import numpy as np
from datadiscoverybench.utils import load_dresden_db, load_git_tables_db


def main():
    con = duckdb.connect(database=':memory:')
    load_dresden_db(con, parts=[0, 1])
    # load_git_tables_db(con)

    query_db(con, 'SELECT DISTINCT TableId FROM AllTables LIMIT 100')

    # fetch_table(con, "allegro_con_spirito_tables_licensed_dictionary")


def query_db(con: duckdb.DuckDBPyConnection, query: str) -> pd.core.frame.DataFrame:
    """
    query db
    :param query: what query to execute
    :param con: connection to database (eg: con = duckdb.connect(database=':memory:') )
    :return: pandas dataframe with resulting table
    """
    df = con.execute(query).fetch_df()
    pd.set_option('display.max_rows', df.shape[0] + 1)
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


if __name__ == '__main__':
    main()
