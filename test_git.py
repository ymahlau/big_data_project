import time

import duckdb

from datadiscoverybench.utils import load_git_tables_db


def main():
    with open('gittable_parts.txt', 'r') as f:
        parts = f.read().splitlines()
    start = time.time()
    con = duckdb.connect(database=':memory:')
    load_git_tables_db(con, parts=parts[2:3])
    print("Connected:  " + str(time.time() - start))

    start = time.time()
    print(con.execute(f"select * from AllTables Limit 100;").fetch_df().to_markdown())
    print("Executed:  " + str(time.time() - start))



if __name__ == '__main__':
    main()
