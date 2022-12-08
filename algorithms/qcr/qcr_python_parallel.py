import math
import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Any, Tuple
import itertools

import numpy as np
import pandas as pd
from tqdm import tqdm
import duckdb

# from datadiscoverybench.utils import load_dresden_db
from qcr import save_index, get_kc, get_c, create_sketch, hash_function, generate_term_keys, get_table_id, \
    add_to_inverted_index, load_index

SKETCH_SIZE = 100
MIN_NUM_ROWS = 1
NUM_WORKER = 4
PARTITION_SIZE = 100

stats_file_path = Path(__file__).parent / 'stats.txt'
# database connection
print(f'Process: {os.getpid()} started connecting to db')
database_path = "/home/groupb/database_v051.db"
con = duckdb.connect(database=database_path)
# con = duckdb.connect(database=':memory:')
# load_dresden_db(con, parts=[0, 1])

def get_column_pairs(table) -> List[pd.DataFrame]:
    kc_list = table.select_dtypes(include=["object"]).columns.to_list()
    c_list = table.select_dtypes(include=["float64", "int64", np.number]).columns.to_list()
    column_pairs = list(itertools.product(kc_list, c_list))
    pair_list = []
    for kc, c in column_pairs:
        pair_list.append(table[[kc, c]])
    return pair_list


def build_index_part(partitions: pd.DataFrame) -> dict:
    table_id_list = partitions.iloc[:, 0].values.tolist()
    table_id_str = str(table_id_list).replace('[', '(').replace(']', ')')
    load_df_query = f"select * from allTables where tableID in {table_id_str};"
    global con
    print('a')
    df = con.execute(load_df_query).fetch_df()
    print('b')
    # grouped = df.sort_values(by=['TableId', 'RowId', 'ColumnId']).set_index(
    #     ['TableId', 'ColumnId', 'RowId']).unstack(-1)

    grouped = df.groupby('TableId')
    table_dict = {}
    for table_id in table_id_list:
        table_stacked = grouped.get_group(table_id)
        table = table_stacked.drop('TableId', axis=1).set_index(['RowId', 'ColumnId']).unstack(-1).droplevel(-2, axis=1)
        if table.shape[0] >= MIN_NUM_ROWS:
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


def build_index_parallel(partitions, index_path: str):
    print('starting building index')
    start_time = time.time()
    with Pool(NUM_WORKER) as pool:
        inverted_index = defaultdict(set)
        result_list = [pool.apply_async(build_index_part, (part,)) for part in partitions]

        for res_dict_wrapper in tqdm(result_list):
            res_dict_wrapper.wait()
            res_dict = res_dict_wrapper.get()
            for k, v in res_dict:
                inverted_index[k].add(v)
    with open(index_path, "wb") as f:
        pickle.dump(inverted_index, f)

    end_time = time.time()
    runtime = end_time - start_time
    print(f'runtime: {runtime}')
    with open(stats_file_path, 'w') as f:
        f.write(f'Finished in {runtime}s')


def build_dresden():
    # index_path = "/home/groupb/python_parallel_index.pickle"
    index_path = "index.pickle"

    print('started fetching part list')
    global con
    id_df = con.execute("select distinct TableId from AllTables where TableId < 100000").fetch_df()
    num_tables = id_df.shape[0]
    if num_tables % PARTITION_SIZE == 0:
        num_partitions = int(num_tables / PARTITION_SIZE)
    else:
        num_partitions = math.floor(num_tables / PARTITION_SIZE) + 1
    partitions = []
    for i in range(num_partitions):
        partitions.append(id_df.iloc[i*PARTITION_SIZE:(i+1)*PARTITION_SIZE, :])

    build_index_parallel(partitions[:100], index_path)


if __name__ == '__main__':
    build_dresden()
    idx = load_index()
    print(len(idx))
    pass