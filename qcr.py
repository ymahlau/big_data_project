from typing import Callable, Tuple, List, DefaultDict, Set, Union, Any, Dict
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
    :return:
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
        kc: List[str],
        c: List[numeric],
        hash_funct: Callable[[str], int],
        n=100
) -> List[Tuple[str, numeric]]:
    """

    :param kc: kategorical key
    :param c: numeric value
    :param hash_funct: collision free hash function string -> int/float
    :param n: size of sketch, default 100
    :return: sketch of size n for given coulumns
    """
    sketch = heapq.nsmallest(n, zip(kc, c), key=lambda x: hash_funct(x[0]))

    return sketch


def key_labeling(sketch: List[Tuple[str, numeric]], h: Callable[[str], int] = lambda x: x) \
        -> List[Union[int, str]]:
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


def build_index(tables: List[pd.DataFrame], n=100) -> None:
    inverted_index = load_index()

    for table in tables:
        kc = get_kc(table)
        c = get_c(table)
        two_col_tables = cross_product_tables(kc, c)
        table_id = get_table_id(table)
        for tbl in two_col_tables:
            this_table_id = f"{table_id}_{tbl.columns.name}"
            sketch = create_sketch(tbl.iloc[:, 0].tolist(), tbl.iloc[:, 1].tolist(), hash_function, n)
            terms = key_labeling(sketch)  # (sketch, h) # we currently don't use the hash values for the sketch
            add_to_inverted_index(inverted_index, terms, this_table_id)

    save_index(inverted_index)


def find_tables(query: pd.DataFrame) -> List[Tuple[Any, int]]:
    kc = get_kc(query)
    kc = list(kc.values())[0]  # we only work wit 2 col queries
    c = get_c(query)
    c = list(c.values())[0]    # therefore we only have one num- and one cat-column
    sketch = create_sketch(kc, c, hash_function)
    terms = key_labeling(sketch)  # (sketch, h) # we currently don't use the hash values for the sketch
    anti_terms = key_labeling(
        list(map((lambda key_value: (key_value[0], -key_value[1])), sketch)))  # , h)
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
    """
    Loads all tables as pandas dataframe from csv files
    :param folder_name:
    :return:
    """
    tables = []
    for path in Path(folder_name).glob("*.csv"):
        table = pd.read_csv(path, sep=";")
        table.columns.name = path.stem
        tables.append(table)

    return tables


def load_query() -> pd.DataFrame:
    """
    # Loads query table as pandas dataframe from csv file
    :return:
    """

    return pd.read_csv("toy_tables/A_0.csv", sep=";")


def cross_product_tables(cat_col: DefaultDict[str, List[str]], num_col: DefaultDict[str, List[numeric]]) -> List[pd.DataFrame]:
    """
    combines all numerical and categorical colums like a cross-product.
    eg: c1, c2 x n1, n2, n3 -> ['c1_n1', c1_n2', 'c1_n3','c2_n1', 'c2_n2', c2_n3']
    :param cat_col: default dict with column name as key and list of categorical-column-values as value.
    :param num_col: default dict with column name as key and list of numerical-column-values as value.
    :return: list of named tables.
    """
    tables = []
    for cat_header in cat_col:
        for num_header in num_col:
            table = pd.DataFrame(list(zip(cat_col[cat_header], num_col[num_header])), columns =[cat_header, num_header])
            table.columns.name = f"{cat_header}_{num_header}"   # here be use the column names as name for the new table
            tables.append(table)
    return tables


def get_kc(table: pd.DataFrame) -> DefaultDict[str, List[str]]:
    """
    returns categorical column of dataframe
    :param table:
    :return:
    """
    kc_column_name = table.select_dtypes(include=["object"]).columns
    columns = {}
    for col in kc_column_name:
        columns[col] = (table[col].values.tolist())
    return columns


def get_c(table: pd.DataFrame) -> DefaultDict[str, List[numeric]]:
    """
    # returns numerical column of dataframe
    :param table:
    :return:
    """
    c_column_name = table.select_dtypes(include=["float64", "int64"]).columns
    columns = {}
    for col in c_column_name:
        columns[col] = (table[col].values.tolist())
    return columns


def get_table_id(table: pd.DataFrame) -> str:
    return table.columns.name


#############################################
# Test it                                   #
#############################################


if __name__ == "__main__":

    # sample with table from notebook
    tbl = pd.read_csv('data/test_table.csv')
    tbl.columns.name = 'testTable'
    print(tbl)
    build_index([tbl], n=3)
    q = pd.read_csv('data/q.csv')
    print(find_tables(q))

    # sample with generated toy tables
    #build_index(load_tables("toy_tables"))

    # queries may only consist of 1 numerical and 1 categorical column.
    #print(find_tables(load_query()))

    # Delete index file for debugging and testing purposes
    # Usually you would build the index once and then use it
    # for multiple queries
    Path("index.pickle").unlink()
