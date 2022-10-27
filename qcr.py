from typing import Callable, Tuple, List, DefaultDict, Set
import heapq
from collections import defaultdict
import pickle
import hashlib
from unicodedata import numeric
import pandas as pd
from pathlib import Path
from collections import Counter


#############################################
# Paper related code                        #
#############################################


def create_hash_functions(
    KC: List[str], C: List[numeric]
) -> Tuple[Callable[[str], int], Callable[[int], int]]:
    """
    Create hash functions h and hu
    """

    return (
        lambda x: int.from_bytes(
            hashlib.sha256(x.encode("utf-8")).digest(), "big", signed=False
        ),
        lambda x: int.from_bytes(
            hashlib.sha256(str(x).encode("utf-8")).digest(), "big", signed=False
        ),
    )


def create_sketch(
    KC: List[str],
    C: List[numeric],
    h: Callable[[str], int],
    hu: Callable[[int], int],
    n=1000,
) -> List[Tuple[str, numeric]]:
    """
    Create sketch
    """

    return heapq.nsmallest(n, zip(KC, C), key=lambda x: hu(h(x[0])))


def tk(sketch: List[Tuple[str, numeric]], h: Callable[[str], int]) -> List[int]:
    """
    Generate term keys for sketch
    """

    mue = sum([value for key, value in sketch]) / len(sketch)
    return [h(f'{h(key)}{"+1" if value > mue else "-1"}') for key, value in sketch]


def add_to_inverted_index(
    inverted_index: DefaultDict[int, Set[str]], terms: List[int], value: str
) -> None:
    """
    Add value to inverted index
    """
    for term in terms:
        inverted_index[term].add(value)


def build_index() -> None:
    inverted_index = load_index()
    tables = load_tables()

    for table in tables:
        KC = get_kc(table)
        C = get_c(table)
        h, hu = create_hash_functions(KC, C)
        sketch = create_sketch(KC, C, h, hu, 20)
        terms = tk(sketch, h)
        table_id = get_table_id(table)
        add_to_inverted_index(inverted_index, terms, table_id)

    save_index(inverted_index)


def find_tables(query: pd.DataFrame) -> List[str]:
    KC = get_kc(query)
    C = get_c(query)
    h, hu = create_hash_functions(KC, C)
    sketch = create_sketch(KC, C, h, hu, 20)
    terms = tk(sketch, h)
    anti_terms = tk(
        list(map((lambda key_value: (key_value[0], -key_value[1])), sketch)), h
    )
    inverted_index = load_index()
    result = Counter()
    result.update(
        "+:" + table_id for term in terms for table_id in inverted_index[term]
    )
    result.update(
        "-:" + table_id for term in anti_terms for table_id in inverted_index[term]
    )
    return result.most_common(10)


#############################################
# Utils                                     #
#############################################


def load_index() -> DefaultDict[int, Set[str]]:
    try:
        with open("index.pickle", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return defaultdict(set)


def save_index(index: DefaultDict[int, Set[str]]) -> None:
    with open("index.pickle", "wb") as f:
        pickle.dump(index, f)


def load_tables() -> List[pd.DataFrame]:
    # Loads all tables as pandas dataframe from csv files

    tables = []
    for path in Path("toy_tables").glob("*.csv"):
        table = pd.read_csv(path, sep=";")
        table.columns.name = path.stem
        tables.append(table)

    return tables


def load_query() -> pd.DataFrame:
    # Loads query table as pandas dataframe from csv file

    return pd.read_csv("toy_tables/A_1.csv", sep=";")


def get_kc(table: pd.DataFrame) -> List[str]:
    KC_column_name = table.select_dtypes(include=["object"]).columns[0]
    return table[KC_column_name].values.tolist()


def get_c(table: pd.DataFrame) -> List[numeric]:
    C_column_name = table.select_dtypes(include=["float64", "int64"]).columns[0]
    return table[C_column_name].values.tolist()


def get_table_id(table: pd.DataFrame) -> str:
    return table.columns.name


#############################################
# Test it                                   #
#############################################


if __name__ == "__main__":
    build_index()
    print(find_tables(load_query()))

    # Delete index file for debug purposes
    # Usually you would build the index once and then use it
    # for multiple queries
    Path("index.pickle").unlink()
