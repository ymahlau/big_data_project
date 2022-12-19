from pathlib import Path

import duckdb
import pandas as pd

from algorithms.naive.naive_callback import NaiveCallback
from machine_learning.ml_who import load_who_query
from utils.table_mapper import map_chunks


def main():
    # con = duckdb.connect(':memory:')
    con = duckdb.connect(database="/home/groupb/big_data_project/data/indices/gittables_naive_who.db")
    query = load_who_query()
    # table = pd.read_csv('../data/real_data/countries_of_the_world.csv', sep=';')
    # table = pd.read_csv('../data/real_data/countries_of_the_world_cleaned.csv')
    callback = NaiveCallback(query)
    # result = callback(table)
    # print(result)
    with open('/home/groupb/big_data_project/data/gittable_parts.txt') as f:
        parts = f.read().split('\n')
        parts.remove('')

    map_chunks(
        con,
        'result_table',
        '/home/groupb/big_data_project/data/zip_cache',
        # [
        #     # 'abstraction_tables_licensed',
        #     # 'allegro_con_spirito_tables_licensed',
        #     # 'beats_per_minute_tables_licensed', 'cease_tables_licensed',
        #     'centripetal_acceleration_tables_licensed'
        # ],
        parts[42:44],
        callback
    )
    # print(con.execute('SELECT * FROM result_table where joinability > 0').fetchdf())


if __name__ == '__main__':
    main()
