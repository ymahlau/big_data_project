import pandas as pd
import duckdb

from utils import load_dataframe_to_db

def get_column_pairs(table_name) -> str:
    query = f"""
    select categorical.TableID, categorical_column_id, numerical_column_id
    from ({get_categorical_columns(table_name)}) as categorical
        join ({get_numerical_columns(table_name)}) as numerical
            on categorical.TableID = numerical.TableID
    """
    return query


def get_numerical_columns(table_name) -> str:
    query = f"""
    select TableID, ColumnID as numerical_column_id
    from {table_name}
    group by TableID, ColumnID
    having count(*) = count(try_cast(CellValue as DOUBLE))
    """
    return query

def get_categorical_columns(table_name) -> str:
    query = f"""
    select TableID, ColumnID as categorical_column_id
    from {table_name}
    group by TableID, ColumnID
    having count(try_cast(CellValue as DOUBLE)) = 0
    """
    return query

def get_sketches(table_name, sketch_size = 100):
    full = f"""
    select Categorical.TableId as TableID, Categorical.CellValue as Category, avg(CAST(Numerical.CellValue as DOUBLE)) as Value,
        row_number() over (Partition by Categorical.TableId order by md5_number(Categorical.CellValue)) as RowNumber
    from {table_name} Categorical
        join {table_name} Numerical
            on Categorical.TableId = Numerical.TableId
                and Categorical.RowId = Numerical.RowId
        join ({get_column_pairs(table_name)}) Pairs
            on Categorical.TableId = Pairs.TableID
                and Categorical.ColumnID = Pairs.categorical_column_id
                and Numerical.ColumnID = Pairs.numerical_column_id

    where Numerical.CellValue != 'nan'

    group by Categorical.TableId, Categorical.CellValue

    """

    query = f"""
    select TableID, Category, Value
    from ({full}) as t
    where RowNumber <= {sketch_size}
    order by Value
    """

    return query


def get_terms(table_name):
    query = f"""
    select md5(concat(Category, (CASE WHEN Mue > Value THEN '+1' ELSE '-1' END))) as term, Sketches.TableID
    from ({get_sketches(table_name)}) as Sketches
        join (select TableID, avg(Value) as Mue from ({get_sketches(table_name)}) group by TableID) as Mues
            on Sketches.TableID = Mues.TableID
    """
    return query

def create_inverted_index(con, table_name, index_name):
    query = get_terms(table_name)
    query = f"create or replace view {index_name} as \n" + query
    con.execute(query)


def search_inverted_index(con, table_name, index_name):
    create_inverted_index(con, table_name, table_name + "Index")

    query = f"""
    select TableID + {table_name}, count(*) as Count
    from {index_name}
    where term in (select term from {table_name + "Index"})
    group by TableID
    """

    return query

def search_correlated(con: duckdb.DuckDBPyConnection,df: pd.DataFrame):
    load_dataframe_to_db(con, df, "TmpTablePlus")
    t_plus = search_inverted_index(con, "TmpTablePlus", "TermIndex")

    df = df.copy()
    # Invert all numerical columns
    for col in df.select_dtypes(include=['number']).columns:
        df[col] = -df[col]
    

    load_dataframe_to_db(con, df, "TmpTableMinus")
    t_minus = search_inverted_index(con, "TmpTableMinus", "TermIndex")

    query = f"""
    {t_plus}
    union all
    {t_minus}
    order by Count desc
    """

    con.execute(query).fetch_df()
