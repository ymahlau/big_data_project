import io
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import List, Callable

import duckdb
from zipfile import ZipFile
import pyarrow.parquet as pq
import pandas as pd
import multiprocessing
import concurrent.futures
from functools import partial
import os
from tqdm import tqdm

import urllib.request
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

process_unique_zipfile = None

def file2result(part: str, callback: Callable[[pd.DataFrame], pd.DataFrame], file_name: str):
    global process_unique_zipfile
    if not Path(file_name).suffix != 'parquet':  # optional filtering by filetype
        return None
    try:
        table_name = str(part.replace("'", '') + '_' + Path(file_name).stem)
        pq_bytes = process_unique_zipfile.read(file_name)
        pq_file_like = io.BytesIO(pq_bytes)
        df = pd.read_parquet(pq_file_like, engine="fastparquet")
        df.columns.name = table_name
        result_table = callback(df)
        return result_table

    except Exception as e:
        print(f"Error: {part=}, {file_name=} ->", e)


def init_worker(zip_path):
    global process_unique_zipfile
    process_unique_zipfile = ZipFile(zip_path)  # Todo: Check whether this is closed properly


def process_zip(con: duckdb.DuckDBPyConnection, result_table_name: str, zip_path: Path, callback: Callable[[pd.DataFrame], pd.DataFrame]):
    with ZipFile(zip_path) as zf:
        file_names = zf.namelist()

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=multiprocessing.cpu_count(),
            initializer=init_worker,
            initargs=(zip_path, )
    ) as pool:

        def cache_and_store(item, last=False, limit=500, item_cache=[]):
            if item is not None:
                item_cache.append(item)
            if len(item_cache) > limit or last:
                result_merged_df = pd.concat(item_cache, axis=0)
                con.execute(f"INSERT INTO {result_table_name} SELECT * FROM result_merged_df")
                item_cache.clear()

        for result_table_df in tqdm(pool.map(partial(file2result, str(zip_path.stem), callback), file_names), total=len(file_names), leave=False):
            if result_table_df is None:
                continue
            cache_and_store(result_table_df)

        cache_and_store(None, last=True)


def map_parts(con: duckdb.DuckDBPyConnection,
              result_table_name: str,
              zip_folder_path: str,
              parts: List[str],
              callback: Callable[[pd.DataFrame], pd.DataFrame]
              ):
    """
    :param con: DuckDB connection that the result table will be inserted into
    :param result_table_name: Name of the result table
    :param zip_folder_path: Path to the folder containing the cached zip files
    :param parts: List of parts to be loaded
    :param callback: Callback function that takes a DataFrame and returns a DataFrame
    """

    # Transform zip_folder_path to absolute Path object if it is not already
    zip_folder_path = Path(zip_folder_path)
    zip_folder_path.mkdir(parents=True, exist_ok=True)

    # Download zip files that do not yet exist and add them to the paths list
    zip_paths: List[Path] = []
    print('Downloading zip files...')
    for part in tqdm(parts, leave=False):
        zip_path = (zip_folder_path / part).with_suffix(".zip")
        zip_paths.append(zip_path)
        if not zip_path.exists():
            urllib.request.urlretrieve(
                "https://zenodo.org/record/6517052/files/" + part + ".zip",
                zip_path)

    # Create table that all results are inserted to
    shape_table_df = callback(pd.DataFrame(), only_shape=True)
    con.execute(f"CREATE TABLE {result_table_name} AS SELECT * FROM shape_table_df LIMIT 1")
    con.execute(f"DELETE FROM {result_table_name}")

    # For all parts calculate the results in parallel (parallelization by table file)
    print('Calculating results...')
    for zip_path in tqdm(zip_paths):
        process_zip(con, result_table_name, zip_path, callback)
