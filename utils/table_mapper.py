import io
import sys
import time

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


def file2result(zf, zip_path, callback, zip_lock: multiprocessing.Lock, file):
    if not file.endswith('.parquet'):  # optional filtering by filetype
        return None
    try:
        # Zipdatei folder/myzip.zip: file1.parquet, file2.parquet, file3.parquet ||||| file1.parquet -> myzip_file1
        table_name = zip_path.split('/')[-1].split('.')[0].replace('\'', '') + '_' + file.split('.')[0]
        zip_lock.acquire()
        pq_bytes = zf.read(file)
        zip_lock.release()
        pq_file_like = io.BytesIO(pq_bytes)
        df: pd.DataFrame = pq.read_table(pq_file_like).to_pandas()
        df.columns.name = table_name
        result_table = callback(df)
        return result_table

    except Exception as e:
        print("error: " + str(zip_path + file) + ' -> ' + str(e))


def zip2result(zip_path, pool, callback, con, result_table_name):
    with ZipFile(zip_path) as zf:
        zip_lock = multiprocessing.Lock()
        for result_table_df in pool.map(partial(file2result, zf, zip_path, callback, zip_lock), zf.namelist()):
            if result_table_df is None:
                continue
            con.execute(f"INSERT INTO {result_table_name} SELECT * FROM result_table_df")


def map_parts(con, result_table_name, zip_folder_path, parts, callback):
    # Download zip files that do not yet exist
    print('Downloading zip files...')
    for p in tqdm(parts):
        if not os.path.isfile(zip_folder_path + p + '.zip'):
            urllib.request.urlretrieve(
                "https://zenodo.org/record/6517052/files/" + p + ".zip",
                zip_folder_path + p + '.zip')

    # Generate a list of all zip files of the required parts
    paths = [zip_folder_path + p + '.zip' for p in parts]

    # Create table that all results are inserted to
    shape_table_df = callback(pd.DataFrame())
    con.execute(f"CREATE TABLE {result_table_name} AS SELECT * FROM shape_table_df LIMIT 0")

    # For all parts calculate the results in parallel (parallelization by table file)
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as pool:
        print('Calculating results...')
        for path in tqdm(paths):
            zip2result(path, pool, callback, con, result_table_name)
