from typing import Callable, Tuple, List, DefaultDict, Set, Union
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




def create_hash_functions() -> Tuple[Callable[[str], int], Callable[[int], int]]:
    """
    Create hash functions h and hu
    """

    return (
        lambda x: int.from_bytes(
            hashlib.md5(x.encode("utf-8")).digest(), "little", signed=True
        ),
        lambda x: int.from_bytes(
            hashlib.md5(str(x).encode("utf-8")).digest(), "little", signed=False
        ) / 2 ** 256,
    )

def hash_function(obj: str) -> float:
    """
    Hashes a string to a value between 0 and 1
    :param obj: String to hash
    :return: hash value
    """
    return int.from_bytes(
        hashlib.md5(str(obj).encode("utf-8")).digest(), "little", signed=True
    ) / 2 ** 256

def create_sketch(
    KC: List[str],
    C: List[numeric],
    hash_function: Callable[[str], int],
    n=100
) -> List[Tuple[str, numeric]]:
    """
    Create sketch
    """

    return heapq.nsmallest(n, zip(KC, C), key=lambda x: hash_function(x[0]))


def generate_term_keys(sketch: List[Tuple[str, numeric]], h: Callable[[str], int] = lambda x: x) -> List[Union[int, str]]:
    """
    Generate term keys for sketch
    """

    mue = sum([value for key, value in sketch]) / len(sketch)
    return [h(f'{h(key)}{"+1" if value > mue else "-1"}') for key, value in sketch]


def add_to_inverted_index(
    inverted_index: DefaultDict[int, Set[str]], terms: List[Union[int, str]], value: str
) -> None:
    """
    Add value to inverted index
    """
    for term in terms:
        inverted_index[term].add(value)


def build_index(tables: List[pd.DataFrame]) -> None:
    inverted_index = load_index()

    for table in tables:
        KC = get_kc(table)
        C = get_c(table)
        h, hu = create_hash_functions()
        sketch = create_sketch(KC, C, hash_function)
        terms = generate_term_keys(sketch, h)
        table_id = get_table_id(table)
        add_to_inverted_index(inverted_index, terms, table_id)

    save_index(inverted_index)


def find_tables(query: pd.DataFrame) -> List[str]:
    KC = get_kc(query)
    C = get_c(query)
    h, hu = create_hash_functions()
    sketch = create_sketch(KC, C, hash_function)
    terms = generate_term_keys(sketch, h)
    anti_terms = generate_term_keys(
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
    matches = result.most_common(10)
    return matches


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


def load_tables(folder_name) -> List[pd.DataFrame]:
    # Loads all tables as pandas dataframe from csv files

    tables = []
    for path in Path(folder_name).glob("*.csv"):
        table = pd.read_csv(path, sep=";")
        table.columns.name = path.stem
        tables.append(table)

    return tables


def load_query() -> pd.DataFrame:
    # Loads query table as pandas dataframe from csv file

    return pd.read_csv("toy_tables/A_0.csv", sep=";")


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
    build_index(load_tables("toy_tables"))
    print(find_tables(load_query()))

    # Delete index file for debug purposes
    # Usually you would build the index once and then use it
    # for multiple queries
    Path("index.pickle").unlink()
