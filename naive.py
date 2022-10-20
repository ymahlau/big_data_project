import pandas as pd
from pathlib import Path
from scipy import stats
from itertools import product


def main():
    query = load_query()
    tables = load_tables()
    joinable_tables = get_joinable_tables(query, tables)
    reorderd_tables = rank_tables(query, joinable_tables)
    
    result = pd.DataFrame()
    result["result_table_name"] = list(map(lambda x: x[1].columns.name, reorderd_tables))
    result["correlation"] = list(map(lambda x: x[0], reorderd_tables))
    result["query_join_column"] = list(map(lambda x: x[2], reorderd_tables))
    result["result_join_column"] = list(map(lambda x: x[3], reorderd_tables))
    result["query_correlated_column"] = list(map(lambda x: x[4], reorderd_tables))
    result["result_correlated_column"] = list(map(lambda x: x[5], reorderd_tables))

    print(result)


def load_query():
    # Loads query table as pandas dataframe from csv file

    return pd.read_csv('data/test_data_01.csv', sep=";")

def load_tables():
    # Loads all tables as pandas dataframe from csv files
    
    tables = []
    for path in Path('data').glob('*.csv'):
        table = pd.read_csv(path, sep=";")
        table.columns.name = path.stem
        tables.append(table)
    
    return tables

def get_joinable_tables(query: pd.DataFrame, tables: list, k=20):
    # Gets top k tables that are joinable with query table
    
    # Get all string columns from query table
    query_cols = query.select_dtypes(include=['object']).columns
    joinability_with_column = []
    
    # Test for joinability with all tables and their columns
    for table in tables:
        for query_column in query_cols:
            for result_column in table.select_dtypes(include=['object']).columns:
                joinability = len(set(query[query_column]) & set(table[result_column]))
                if joinability > 0: # Todo: maybe add threshold
                    joinability_with_column.append((joinability, table, query_column, result_column))

    return sorted(joinability_with_column, key=lambda x: x[0], reverse=True)[:k]


def rank_tables(query, tables):
    # Reorders tables based on correlation with query table

    reorderd = []
    for joinability, table, col, col2 in tables:
        joined = query.add_suffix("_q").merge(table.add_suffix("_t"), left_on=col+"_q", right_on=col2+"_t", how='inner')

        
        # Group numeric columns by suffix
        joined_cols = joined.select_dtypes(include=['float64', 'int64']).columns
        left_cols = [col for col in joined_cols if col.endswith('_q')]
        right_cols = [col for col in joined_cols if col.endswith('_t')]


        # Calculate correlation
        for query_col, table_col in product(left_cols, right_cols):
            corr = stats.pearsonr(joined[query_col], joined[table_col])[0]
            reorderd.append((corr, table, col, col2, query_col, table_col))


    return sorted(reorderd, key=lambda x: abs(x[0]), reverse=True)


if __name__ == '__main__':
    main()
