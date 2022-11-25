import math
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Any, Tuple
import itertools
import pandas as pd
from tqdm import tqdm
import duckdb
import numpy as np

from qcr import save_index, get_kc, get_c, create_sketch, hash_function, generate_term_keys, get_table_id, \
    add_to_inverted_index, load_index

con = None
SKETCH_SIZE = 100
MIN_NUM_COLS = 20

def get_column_pairs(table) -> List[pd.DataFrame]:
    kc_list = table.select_dtypes(include=["object"]).columns.to_list()
    c_list = table.select_dtypes(include=["float64", "int64"]).columns.to_list()
    column_pairs = list(itertools.product(kc_list, c_list))
    pair_list = []
    for kc, c in column_pairs:
        pair_list.append(table[[kc, c]])
    return pair_list


def build_index_part(table_id_lower: int, table_id_upper: int) -> dict:
    load_df_query = f"select * from allTables where tableID >= {table_id_lower} and tableID < {table_id_upper}"
    df = con.execute(load_df_query).fetch_df()

    # grouped = df.sort_values(by=['TableId', 'RowId', 'ColumnId']).set_index(
    #     ['TableId', 'ColumnId', 'RowId']).unstack(-1)

    grouped = df.groupby('TableId')
    table_dict = {}
    for table_id in range(table_id_lower, table_id_upper):
        table_stacked = grouped.get_group(table_id)
        table = table_stacked.drop('TableId', axis=1).set_index(['RowId', 'ColumnId']).unstack(-1).droplevel(-2, axis=1)
        if table.shape[0] >= MIN_NUM_COLS:
            table_dict[table_id] = table

    inverted_index = defaultdict(set)
    for table_id, table in table_dict.items():
        col_pair_tables = get_column_pairs(table)

        for col_pair_tbl in col_pair_tables:
            col_id = col_pair_tbl.columns[0]
            sketch = create_sketch(col_pair_tbl.iloc[:, 0], col_pair_tbl.iloc[:, 1], hash_function, n=SKETCH_SIZE)
            terms = generate_term_keys(sketch)
            add_to_inverted_index(inverted_index, terms, (table_id, col_id))

    return inverted_index


def build_index_parallel(con, num_workers: int, task_args, index_path: str):
    print('starting building index')
    start_time = time.time()
    p = Pool(num_workers)
    result_dict = defaultdict(set)

    result = p.starmap(build_index_part, task_args)
    try:
        task = p._cache[result._job]
        while task._number_left > 0:
              # track progress
            print("Tasks remaining = {0}".format(task._number_left * task._chunksize))
            time.sleep(10)
    except:
        print('done with index building, starting merge')

    for dict_part in result:
        for k, v in dict_part:
            result_dict[k].add(v)
    save_index(result_dict)

    end_time = time.time()
    runtime = end_time - start_time
    print(f'runtime: {runtime}')
    with open('/home/groupb/big_data_project/stats_temp.txt', 'w') as f:
        f.write(f'Finished in {runtime}s')


def build_dresden():
    database_path = "/home/groupb/database_v051.db"
    index_path = "/home/groupb/python_parallel_index.pickle"
    # num_tables = 145533822
    # partition_size = 1000
    # num_workers = 64
    num_tables = 100000
    partition_size = 100
    num_workers = 4
    num_partitions = math.floor(num_tables / partition_size) + 1

    partition_bounds = list(partition_size * np.arange(num_partitions - 1))
    partition_bounds.append(num_tables)
    partitions_lower = partition_bounds[:-1]
    partitions_upper = partition_bounds[1:]

    print('started connecting to db')
    global con
    con = duckdb.connect(database=database_path)
    tasks = list(zip(partitions_lower, partitions_upper))
    build_index_parallel(con, num_workers, tasks, index_path)


if __name__ == '__main__':
    # build_dresden()
    idx = load_index()
    print(len(idx))
    pass