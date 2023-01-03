import hashlib
import heapq
import os
import pandas as pd
import pickle
import textwrap
from collections import defaultdict, Counter
from pathlib import Path
from typing import Callable, Tuple, List, DefaultDict, Set, Union, Any, Dict
from unicodedata import numeric


#############################################
# Paper related code                        #
#############################################


def hash_function(obj: str) -> float:
    """
    Hashes a string to a value between 0 and 1
    :param obj: String that needs to be hashed
    :return: hashed value
    """
    return int.from_bytes(
        hashlib.md5(str(obj).encode("utf-8")).digest(), "big", signed=False
    ) / 2 ** 128

def hash_md5(obj: str) -> int:
    """
    Hashes a string to an integer value
    :param obj: String that needs to be hashed
    :return: hashed value
    """
    return int.from_bytes(
        hashlib.md5(str(obj).encode("utf-8")).digest(), "big", signed=False
    )


def create_sketch(
        kc: List[str],
        c: List[numeric],
        hash_funct: Callable[[str], int],
        n=100
) -> List[Tuple[str, numeric]]:
    """
    This function creates a sketch of size n from two columns (one with numeric values, one with categorical values).
    It hashes the categorical column and builds a table (list of tuples) with the hashes and the corresponding values
    from the numerical column. This table is sorted by the hash-column and the rows with the n-smallest hash-values are
    kept for form the sketch
    :param kc: list of categorical keys (key column)
    :param c: list of numeric values (value column)
    :param hash_funct: collision free hash function string -> int/float
    :param n: size of sketch, default 100
    :return: sketch of size n for given columns
    """
    grouped = pd.DataFrame({'kc': kc, 'c': c}).groupby('kc').mean(numeric_only=True).reset_index()
    sketch = heapq.nsmallest(n, zip(grouped["kc"], grouped["c"]), key=lambda x: hash_funct(x[0]))
    return sketch


def key_labeling(sketch: List[Tuple[str, numeric]], h: Callable[[str], int] = lambda x: int.from_bytes(str(x).encode("utf-8"), "big", signed=False), inner_hash: bool = False) \
        -> List[Union[int, str]]:
    """
    labels keys according to their values' distribution. +key or -key
    :param sketch: table with keys and their values
    :param h: hash function if hashed keys shall be labeled, nothing if literal keys shall be labeled
    :return: returns a two col table of labeled keys and values
    """
    mue = sum([value for key, value in sketch]) / len(sketch)
    return [format(h(f'{f"{h(key):x}" if inner_hash else key}{"+1" if value > mue else "-1"}'), "x") for key, value in sketch]


def add_to_inverted_index(
        inverted_index: DefaultDict[int, Set[str]], terms: List[Union[int, str]], value: str
) -> None:
    """
    Add labeled key to inverted index, the table_id it originates from is the value of the dictionary entry.
    :param inverted_index: inverted index to add table content to
    :param terms: list of labeled keys from sketch
    :param value: name of table
    :return: none. Dictionary is edited
    """
    for key in terms:
        inverted_index[key].add(value)


def build_index(tables: List[pd.DataFrame], n=100) -> None:
    """
    builds inverted index of labeled keys from all given tables and the corresponding table and column ids
    :param tables: tables to be included in the index
    :param n: size of sketch per table (sample size)
    :return: none. inverted index is build and stored on disc.
    """
    inverted_index = load_index()

    for table in tables:
        kc = get_kc(table)
        c = get_c(table)
        original_table_id = get_table_id(table)
        two_col_tables = cross_product_tables(kc, c, original_table_id)
        for tbl in two_col_tables:
            sketch = create_sketch(tbl.iloc[:, 0].tolist(), tbl.iloc[:, 1].tolist(), hash_function, n)
            terms = key_labeling(sketch)  # (sketch, h) # we currently don't use the hash values for the sketch
            sub_table_id = get_table_id(tbl)
            add_to_inverted_index(inverted_index, terms, sub_table_id)

    save_index(inverted_index)


def find_tables(query: pd.DataFrame) -> List[Tuple[Any, int]]:
    """
    finds correlated & joinable tables/columns from inverted index to the given query
    :param query: query table, consists of 2 columns only. One numeric, one categorical
    :return: top 10 most joinable & correlated columns from index
    """
    kc = get_kc(query)
    kc = list(kc.values())[0]  # we only work wit 2 col queries
    c = get_c(query)
    c = list(c.values())[0]  # therefore we only have one num- and one cat-column
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
    """
    load index from disc if it exists. creates new, empty index otherwise
    :return: inverted index of type dictionary.
    """
    try:
        with open("index.pickle", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return defaultdict(set)


def save_index(index: DefaultDict[int, Set[str]]) -> None:
    """
    stores index on disc
    :param index: index of type dictionary
    :return: none. Stores on disc
    """
    with open("index.pickle", "wb") as f:
        pickle.dump(index, f)


def load_tables(folder_name: str) -> List[pd.DataFrame]:
    """
    Loads tables as pandas dataframe from all csv files in the folder
    :param folder_name: String of relative path from this script to the folder
    :return: tables (dataframes) contained in given folder
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
    :return: query table as dataframe. 2 columns only!
    """

    return pd.read_csv("../../data/toy_data/A_0.csv", sep=";")


def cross_product_tables(cat_col: DefaultDict[str, List[str]], num_col: DefaultDict[str, List[numeric]],
                         table_id: str) -> List[pd.DataFrame]:
    """
    combines all numerical and categorical columns like a cross-product.
    eg: c1, c2 x n1, n2, n3 -> ['c1_n1', c1_n2', 'c1_n3','c2_n1', 'c2_n2', c2_n3']
    :param cat_col: default dict with column name as key and list of categorical-column-values as value.
    :param num_col: default dict with column name as key and list of numerical-column-values as value.
    :param table_id: name of table, that is to be split
    :return: list of named tables.
    """
    tables = []
    for cat_header in cat_col:
        for num_header in num_col:
            table = pd.DataFrame(list(zip(cat_col[cat_header], num_col[num_header])), columns=[cat_header, num_header])
            table.columns.name = f"{table_id}_{cat_header}_{num_header}"  # here we use the column names as name for the new table
            tables.append(table)
    return tables


def get_kc(table: pd.DataFrame) -> DefaultDict[str, List[str]]:
    """
    extract categorical columns from dataframe
    :param table: input table
    :return: dict of categorical columns by column name
    """
    kc_column_name = table.select_dtypes(include=["object"]).columns
    columns = defaultdict(List[str])
    for col in kc_column_name:
        columns[col] = (table[col].values.tolist())
    return columns


def get_c(table: pd.DataFrame) -> DefaultDict[str, List[numeric]]:
    """
    extract numerical columns from dataframe
    :param table: input table
    :return: dict of numerical columns by column name
    """
    c_column_name = table.select_dtypes(include=["float64", "int64"]).columns
    columns = defaultdict(List[str])
    for col in c_column_name:
        columns[col] = (table[col].values.tolist())
    return columns


def get_table_id(table: pd.DataFrame) -> str:
    """
    extract name from pandas dataFrame
    :param table: pandas dataFrame
    :return: name (string)
    """
    return table.columns.name


def print_dict(dictionary: DefaultDict[Any, List[Any]], identifier="") -> None:
    """
    pretty print version for dictionaries with string or numeric values
    :param dictionary: dict with list of String or list of numeric values
    :param identifier: a title for the dictionary that follows
    :return: none. prints to the console
    """
    if identifier != "":
        print(f"{identifier}:")

    for key in dictionary:
        if isinstance(dictionary[key], str):
            value = ', '.join(dictionary[key])
        else:
            value = str(dictionary[key]).strip('[]')

        prefix = f"{key} "
        wrapper = textwrap.TextWrapper(initial_indent=prefix, width=70,
                                       subsequent_indent=' ' * len(prefix))
        print(wrapper.fill(f"{value}"))


#############################################
# Test it                                   #
#############################################


if __name__ == "__main__":
    # Delete index file for debugging and testing purposes (if it exists)
    # Usually you would build the index once and then use it
    # for multiple queries
    if os.path.exists("index.pickle"):
        Path("index.pickle").unlink()

    # sample with table from notebook
    tbl = pd.read_csv('../../expert_review/test_table.csv')
    tbl.columns.name = 'testTable'
    print(tbl)
    build_index([tbl], n=3)#TODO: make param
    q = pd.read_csv('../../expert_review/q.csv')
    print(find_tables(q))

    # sample with generated toy tables
    # build_index(load_tables("toy_data"))
    # print(find_tables(load_query()))

    # remove connection to inverted index for next test-run
    Path("index.pickle").unlink()
