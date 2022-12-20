from typing import List, Callable, Type

import duckdb
import pandas as pd
import multiprocessing
from functools import partial
from tqdm import tqdm
from utils.chunk import Chunk

process_unique_zipfile = None

def chunk2result(callback: Callable[[pd.DataFrame], pd.DataFrame], part: any) -> pd.DataFrame:
    global process_unique_chunk
    df = process_unique_chunk.get_part(part)
    if df is not None:
        try:
            return callback(df)
        except Exception as e:
            print(e)
            return None
    return None


def init_worker(chunk_cls: Type[Chunk], chunk_label: any) -> None:
    global process_unique_chunk
    process_unique_chunk = chunk_cls(chunk_label)  # Todo: Check whether this is closed properly


def process_chunk(con: duckdb.DuckDBPyConnection, result_table_name: str, chunk_cls: Type[Chunk], chunk_label: any, callback: Callable[[pd.DataFrame], pd.DataFrame], cache_and_store_limit: int):
    with chunk_cls(chunk_label) as chunk:
        parts = chunk.get_part_labels()

    with multiprocessing.Pool(
            processes=multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(chunk_cls, chunk_label, )
    ) as pool:

        def cache_and_store(item,  limit, last=False, item_cache=[]):
            if item is not None:
                item_cache.append(item)
            if len(item_cache) > limit or (last and item_cache):
                result_merged_df = pd.concat(item_cache, axis=0)
                con.register('result_merged_df', result_merged_df)
                con.execute(f"INSERT INTO {result_table_name} SELECT * FROM result_merged_df")
                item_cache.clear()

        for result_table_df in tqdm(pool.imap_unordered(partial(chunk2result, callback), parts, chunksize=16), total=len(parts), leave=False):
            if result_table_df is None:
                continue
            cache_and_store(result_table_df, limit=cache_and_store_limit)
        cache_and_store(None, limit=cache_and_store_limit, last=True)


def map_chunks(con: duckdb.DuckDBPyConnection,
              result_table_name: str,
              chunk_cls: Type[Chunk],
              chunks: List[any],
              callback: Callable[[pd.DataFrame], pd.DataFrame],
              cache_and_store_limit: int = 500,
              ):
    """
    :param con: DuckDB connection that the result table will be inserted into
    :param result_table_name: Name of the result table
    :param chunk_cls: Chunk class that is used to load the data
    :param chunks: List of chunks to be loaded
    :param parts: List of parts to be loaded
    :param cache_and_store_limit: Limit for the cache_and_store item cache size
    :param callback: Callback function that takes a DataFrame and returns a DataFrame
    """

    # Create table that all results are inserted to
    shape_table_df = callback(pd.DataFrame(), only_shape=True)
    con.register('shape_table_df', shape_table_df)
    con.execute(f"CREATE OR REPLACE TABLE {result_table_name} AS SELECT * FROM shape_table_df LIMIT 1")
    con.execute(f"DELETE FROM {result_table_name}")

    # For all chunks calculate the results in parallel (parallelization by table file)
    print('Calculating results...')
    for chunk_label in tqdm(chunks):
        process_chunk(con, result_table_name, chunk_cls, chunk_label, callback, cache_and_store_limit)
