import duckdb
from datadiscoverybench.utils import load_dresden_db, load_git_tables_db

def main():
    con = duckdb.connect(database=':memory:')
    # load_dresden_db(con, parts=[0,1])
    load_git_tables_db(con)
    df = con.execute('SELECT * FROM AllTables LIMIT 5').fetch_df()
    print(df)
if __name__ == '__main__':
    main()
