import duckdb

# from datadiscoverybench.utils import dir_path

import time

from qcr_sql import get_column_pairs

def main():

    start = time.time()
    con = duckdb.connect(database="/home/groupb/database_v051.db")
    print("Connected:  " + str(time.time() - start))

    print(con.execute("summarize select * from AllTables;").fetch_df().to_markdown())
    start = time.time()
    # res = con.execute("select * from AllTables where tableID >= 100 and tableID < 1100;").fetch_df()
    print("Executed:  " + str(time.time() - start))
    start = time.time()

    # con.execute("PRAGMA memory_limit='400GB'; Pragma force_index_join;")
    #con.execute("create index TableID_IDX on AllTables (TableID);")
    #con.execute("import database 'exported_db'")
    # print(con.execute(get_column_pairs('AllTables')))
    #print(con.execute("select count(*) from AllTables;").fetch_df())
    

    print('time: ' + str(time.time() - start))


if __name__ == "__main__":

    main()
