from pathlib import Path

import duckdb
import pandas as pd

from algorithms.naive.naive_callback import NaiveCallback
from machine_learning.ml_who import load_who_query
from utils.table_mapper import map_chunks
from utils.chunk import DresdenChunk


def main():
    # con = duckdb.connect(':memory:')
    con = duckdb.connect(database="/home/groupb/big_data_project/data/indices/dresdentables_naive_who.db")
    query = load_who_query()
    # table = pd.read_csv('../data/real_data/countries_of_the_world.csv', sep=';')
    # table = pd.read_csv('../data/real_data/countries_of_the_world_cleaned.csv')
    callback = NaiveCallback(query)
    # result = callback(table)
    # print(result)

    map_chunks(
        con,
        'result_table',
        DresdenChunk,
        DresdenChunk.get_chunk_labels()[:1],
        callback
    )
    # print(con.execute('SELECT * FROM result_table where joinability > 0').fetchdf())


if __name__ == '__main__':
    main()
