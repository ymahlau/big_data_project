import duckdb
import pandas as pd
from pathlib import Path

from utils.chunk import GitChunk, DresdenChunk

from algorithms.qcr.qcr import get_kc, get_c, create_sketch, hash_function, key_labeling, cross_product_tables
from utils.table_mapper import map_chunks


def count_rows(df, only_shape=False):
    # Workaround to ensure correct inference of column types
    if only_shape:
        return pd.DataFrame({'count': [0]})
    return pd.DataFrame({'count': [len(df)]})


def callback_qcr(df_in: pd.DataFrame, only_shape=False) -> pd.DataFrame:
    if only_shape:
        df_out = pd.DataFrame(zip({"sample_term-1"}, {"sample_table_combination"}),
                              columns=['term_id', 'table_id_catcol_numcol'])
        return df_out  # Emtpy Dataframe and continuosly appending unadvised
    else:
        c_col = get_kc(df_in)
        n_col = get_c(df_in)
        cross_product_tables_list = cross_product_tables(c_col, n_col, df_in.columns.name)
        list1, list2 = [], []
        for i in cross_product_tables_list:
            sketch = create_sketch(i.iloc[:, 0], i.iloc[:, 1], hash_function, n=128)
            labels = key_labeling(sketch, hash_function)
            for term in labels:
                #list1.append(hash_function(term))
                list1.append(term)
                list2.append(i.columns.name)
                # df_out.append({'termid': term, 'table_id_catcol_numcol': i.columns.name})
        df_out = pd.DataFrame(zip(list1, list2), columns=['term_id', 'table_id_catcol_numcol'])
        return df_out


def main():
    con = duckdb.connect(database=":memory:")
    map_chunks(con, 'result_table', DresdenChunk, DresdenChunk.get_chunk_labels()[:1], callback=count_rows)
    print(con.execute('SELECT * FROM result_table').fetchdf())


if __name__ == "__main__":
    main()
