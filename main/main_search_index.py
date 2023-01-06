import argparse
from pathlib import Path
import pandas as pd
from typing import List, Optional
import duckdb
import sys

try:
    from algorithms.qcr.qcr import get_labels_for_table
except ImportError:
    print("Please run 'python -m main.main_search_index [arguments]' from big_data_project folder")
    sys.exit(1)


def search_naive(
    con: duckdb.DuckDBPyConnection, result_size: int,
) -> List[str]:
    result_df = con.execute(
        f"SELECT *, sqrt(joinability * abs(correlation)) as score FROM result_table ORDER BY score DESC LIMIT {result_size}"
    ).df()
    print(result_df.to_markdown())
    return result_df["column_headers"].tolist()


def search_qcr(
    query_df: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    result_size: int,
    query_sketch_size: Optional[int] = None,
) -> List[str]:
    # Prepare positive correlation query
    labels, _ = get_labels_for_table(query_df, query_sketch_size)
    search_df = pd.DataFrame({"term_id": labels})
    con.register("search_df", search_df)

    # Prepare negative correlation query
    query_df_anti = query_df.copy()
    query_df_anti.iloc[:, 1] = -query_df_anti.iloc[:, 1]
    labels_anti, _ = get_labels_for_table(query_df_anti, query_sketch_size)
    search_df_anti = pd.DataFrame({"term_id": labels_anti})
    con.register("search_df_anti", search_df_anti)

    # Execute query
    result_df = con.execute(
        f"""
        (SELECT table_id_catcol_numcol, COUNT(*) as qcr_score, COUNT(*) as abs_qcr_score
        FROM result_table WHERE term_id IN (SELECT term_id FROM search_df)
        GROUP BY table_id_catcol_numcol)
        UNION
        (SELECT table_id_catcol_numcol, -COUNT(*) as qcr_score, COUNT(*) as abs_qcr_score
        FROM result_table WHERE term_id IN (SELECT term_id FROM search_df_anti)
        GROUP BY table_id_catcol_numcol)

        order by abs_qcr_score desc LIMIT {result_size}
        """
    ).df()
    print(result_df.to_markdown())
    return result_df["table_id_catcol_numcol"].tolist()


if __name__ == "__main__":
    # Create the top-level parser
    parser = argparse.ArgumentParser()

    # Add the datalake and result size arguments
    parser.add_argument(
        "datalake",
        choices=["git", "dresden"],
        type=str,
        help="the name of the datalake",
    )
    parser.add_argument("result_size", type=int, help="the result size (top k results)")
    # Create a subcommand parser
    subparsers = parser.add_subparsers(title="subcommands", required=True, dest="search_method")
    # Create a parser for the "naive" subcommand
    parser_naive = subparsers.add_parser("naive", help="use the naive search method")
    parser_naive.add_argument(
        "query", type=str, choices=["song", "who"], help="the query to evaluate"
    )
    # Create a parser for the "qcr" subcommand
    parser_qcr = subparsers.add_parser("qcr", help="use the qcr search method")
    parser_qcr.add_argument("query_path", type=str, help="the path to the query file")
    parser_qcr.add_argument("categorical_column", type=str, help="the categorical column")
    parser_qcr.add_argument("numerical_column", type=str, help="the numerical column")
    parser_qcr.add_argument(
        "query_sketch_size",
        type=int,
        help="the sketch size of the query (only for qcr)",
    )
    parser_qcr.add_argument(
        "index_sketch_size",
        type=int,
        help="the sketch size of the index (only for qcr)",
    )
    parser_qcr.add_argument(
        "-c",
        "--compare_naive",
        choices=["song", "who"],
        type=str,
        dest="compare_naive",
        help="compare the qcr search method with the naive search method",
    )
    # Parse the command-line arguments
    args = parser.parse_args()

    # Load the index to a duckdb database
    

    # Check the search method and execute the appropriate code
    if args.search_method == "qcr":
        # Load the index to a duckdb database
        database_path = (
            Path(__file__).parent.parent
            / "data/indices"
            / f"{args.datalake}tables_{args.search_method}_{args.index_sketch_size}.db"
        )
        con = duckdb.connect(str(database_path))
        # Load the query dataframe
        query_df = pd.read_csv(args.query_path, sep=";")
        query_df = query_df[[args.categorical_column, args.numerical_column]]
        # Execute the qcr search method
        search_qcr(query_df, con, args.result_size, args.query_sketch_size)
    if args.search_method == "naive" or args.compare_naive:
        # Execute the naive search method
        database_path = (
            Path(__file__).parent.parent
            / "data/indices"
            / f"{args.datalake}tables_{args.search_method}_{args.query if args.search_method == 'naive' else args.compare_naive}.db"
        )
        con = duckdb.connect(str(database_path))
        search_naive(con, args.result_size)
    
