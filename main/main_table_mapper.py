import duckdb
import pandas as pd

from utils.table_mapper import map_parts


def count_rows(df, only_shape=False):
    # Workaround to ensure correct inference of column types
    if only_shape:
        return pd.DataFrame({'count': [0]})
    return pd.DataFrame({'count': [len(df)]})


def main():
    con = duckdb.connect(':memory:')
    map_parts(con, 'result_table', 'data/zip_cache/', ['abstraction_tables_licensed',
                                                          'allegro_con_spirito_tables_licensed',
                                                          'beats_per_minute_tables_licensed', 'cease_tables_licensed', 'centripetal_acceleration_tables_licensed', 'id_tables_licensed'], count_rows)
    print(con.execute('SELECT * FROM result_table').fetchdf())


if __name__ == "__main__":
    main()
