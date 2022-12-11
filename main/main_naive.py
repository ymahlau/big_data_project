from pathlib import Path

import duckdb
import pandas as pd

from algorithms.naive.naive_callback import NaiveCallback
from machine_learning.ml_who import load_who_query
from utils.table_mapper import map_parts


def main():
    con = duckdb.connect(':memory:')
    query = load_who_query()
    # table = pd.read_csv('../data/real_data/countries_of_the_world.csv', sep=';')
    # table = pd.read_csv('../data/real_data/countries_of_the_world_cleaned.csv')
    callback = NaiveCallback(query)
    # result = callback(table)
    # print(result)

    map_parts(
        con,
        'result_table',
        '../data/real_data/',
        [
            # 'abstraction_tables_licensed',
            # 'allegro_con_spirito_tables_licensed',
            # 'beats_per_minute_tables_licensed', 'cease_tables_licensed',
            'centripetal_acceleration_tables_licensed'
        ],
        callback
    )
    print(con.execute('SELECT * FROM result_table where joinability > 0').fetchdf())


if __name__ == '__main__':
    main()
