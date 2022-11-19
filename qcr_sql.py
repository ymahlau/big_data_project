import pandas as pd
import duckdb

from utils import load_dataframes_to_db


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


def get_sketches(table_name, sketch_size=100):
    full = f"""
    select concat(Categorical.TableId, '__', Categorical.ColumnID, '_', Numerical.ColumnID) as TableID, Categorical.CellValue as Category, avg(CAST(Numerical.CellValue as DOUBLE)) as Value,
        row_number() over (Partition by Categorical.TableId, Categorical.ColumnID, Numerical.ColumnID order by md5_number(Categorical.CellValue)) as RowNumber
    from {table_name} Categorical
        join {table_name} Numerical
            on Categorical.TableId = Numerical.TableId
                and Categorical.RowId = Numerical.RowId
        join ({get_column_pairs(table_name)}) Pairs
            on Categorical.TableId = Pairs.TableID
                and Categorical.ColumnID = Pairs.categorical_column_id
                and Numerical.ColumnID = Pairs.numerical_column_id

    where isfinite(TRY_CAST(Numerical.CellValue as DOUBLE))

    group by Categorical.TableId, Categorical.ColumnID, Numerical.ColumnID, Categorical.CellValue

    """  # Todo: Check why try_cast is needed

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


def create_inverted_index(
    con: duckdb.DuckDBPyConnection, table_name, index_name, union_old=False
):
    query = get_terms(table_name)
    index_exists = None
    if union_old:
        index_exists = (
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                [index_name],
            )
            .df()
            .size
        )
        print(index_exists)
        if index_exists:
            query = f"""
            insert into {index_name}
            {query}
            """

    if not union_old or not index_exists:
        query = f"create or replace table {index_name} as \n" + query

    con.execute(query)


def search_inverted_index(con, table_name, index_name):
    create_inverted_index(con, table_name, table_name + "Index")

    query = f"""
    select concat(TableID, '-{table_name}') as ID, count(*) as Count
    from {index_name}
    where term in (select term from {table_name + "Index"})
    group by TableID
    """

    return query


def create_joined_table(
    con: duckdb.DuckDBPyConnection, df: pd.DataFrame, joinables, master_table_name
):
    table_name = df.columns.name + "_result"
    query_table_query = f"""
            create or replace table {table_name} as
            select a.cellvalue as ID, b.cellvalue as value
            from {master_table_name} a join {master_table_name} b on a.TableId=b.Tableid
            and a.RowId = b.RowId and a.ColumnId < b.ColumnId
            where a.TableId = '{df.columns.name}'
            """
    con.execute(query_table_query)
    for x in joinables:
        if x != df.columns.name:  # alltables(cellvalue, rowId, columnID, tableID)
            print(x)
            sub_query = f"""
            select a.cellvalue as ID, b.cellvalue as value
            from {master_table_name} a join {master_table_name} b on a.TableId=b.Tableid
            and a.RowId = b.RowId and a.ColumnId < b.ColumnId
            where a.TableId = '{x}'
            """
            full_query = f"""
            create or replace table {table_name} as 
            select q.*, sq.value from {table_name} q left join ({sub_query}) sq on q.ID=sq.ID """
            con.execute(full_query)
    print(con.execute(f"select * from {table_name}").fetch_df())
    con.execute(f"COPY {table_name} TO '{table_name}.csv' (DELIMITER ';', HEADER)")


def search_correlated(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    index_name: str,
    master_table_name: str = None,
    limit=10,
) -> pd.DataFrame:
    load_dataframes_to_db(con, [df], "QPlus")

    s_plus = search_inverted_index(con, "QPlus", index_name)

    # Invert all numerical columns
    df = df.copy()
    for col in df.select_dtypes(include=["number"]).columns:
        df[col] = -df[col]

    load_dataframes_to_db(con, [df], "QMinus")
    s_minus = search_inverted_index(con, "QMinus", index_name)

    query = f"""
    {s_plus}
    union all
    {s_minus}
    order by Count desc
    limit {limit}
    """

    result = con.execute(query).fetch_df()

    con.execute("DROP TABLE QPlus")
    con.execute("DROP TABLE QMinus")
    con.execute("DROP TABLE QPlusIndex")
    con.execute("DROP TABLE QMinusIndex")

    if master_table_name:
        to_join = []
        for x in result.ID:
            end = x.index("__")
            to_join.append(x[:end])
        create_joined_table(con, df, to_join, master_table_name)

    return result
