import io
from zipfile import ZipFile
from timeit import timeit

import pandas as pd
import pyarrow.parquet as pq


def main():
    with ZipFile('data/zip_cache/abstraction_tables_licensed.zip') as zip_file:
        for file_name in zip_file.namelist():
            pq_bytes = zip_file.read(file_name)
            pq_file_like = io.BytesIO(pq_bytes)
            pd.read_parquet(pq_file_like, engine='fastparquet')


if __name__ == "__main__":
    print(timeit(main, number=1))
