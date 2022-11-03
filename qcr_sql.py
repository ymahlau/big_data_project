

def get_column_pairs() -> str:
    query = f"""
    select categorical.TableID, categorical_column_id, numerical_column_id
    from ({get_categorical_columns()}) as categorical
        join ({get_numerical_columns()}) as numerical
            on categorical.TableID = numerical.TableID
    """
    return query


def get_numerical_columns() -> str:
    query = """
    select TableID, ColumnID as numerical_column_id
    from AllTables
    group by TableID, ColumnID
    having count(*) = count(try_cast(CellValue as DOUBLE))
    """
    return query

def get_categorical_columns() -> str:
    query = """
    select TableID, ColumnID as categorical_column_id
    from AllTables
    group by TableID, ColumnID
    having count(try_cast(CellValue as DOUBLE)) = 0
    """
    return query

def get_sketches(sketch_size = 100):
    full = f"""
    select Categorical.TableId as TableID, Categorical.CellValue as Category, avg(CAST(Numerical.CellValue as DOUBLE)) as Value,
        row_number() over (Partition by Categorical.TableId order by md5_number(Categorical.CellValue)) as RowNumber
    from AllTables Categorical
        join AllTables Numerical
            on Categorical.TableId = Numerical.TableId
                and Categorical.RowId = Numerical.RowId
        join ({get_column_pairs()}) Pairs
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


def get_terms():
    query = f"""
    select md5(concat(Category, (CASE WHEN Mue > Value THEN '+1' ELSE '-1' END))) as term, Sketches.TableID
    from ({get_sketches()}) as Sketches
        join (select TableID, avg(Value) as Mue from ({get_sketches()}) group by TableID) as Mues
            on Sketches.TableID = Mues.TableID
    """
    return query

def create_inverted_index(con):
    query = get_terms()
    query = "create or replace view TermIndex as \n" + query
    con.execute(query)

