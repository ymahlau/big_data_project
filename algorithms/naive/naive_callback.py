from itertools import product

import pandas as pd
from scipy import stats

from utils.utils import clean_dataframe


class NaiveCallback:
    def __init__(
            self,
            query: pd.DataFrame,
            agg_method: str = 'avg',
    ):
        self.agg_method = agg_method
        if self.agg_method == 'avg':
            query_cat_col_name = query.select_dtypes(include=["object"]).columns[0]
            self.query = clean_dataframe(query).groupby(query_cat_col_name).mean(numeric_only=True).reset_index()
        else:
            raise ValueError('Unknown Aggregation method')

    def __call__(self, df: pd.DataFrame, only_shape: bool = False) -> pd.DataFrame:
        if only_shape:
            return pd.DataFrame({"column_headers": ["temp"], "joinability": [1.0], "correlation": [1.0]})
        result = self.get_correlation_and_joinability(df)
        return result

    def get_correlation_and_joinability(self, table: pd.DataFrame):
        """

        :param table:
        :type table:
        :return:
        :rtype:
        """
        result = pd.DataFrame({"column_header": [], "joinability": [], "correlation": []})
        query_col_name = self.query.select_dtypes(include=["object"]).columns[0]
        table = clean_dataframe(table)

        for cat_column in table.select_dtypes(include=["object"]).columns:
            table_grouped = table.groupby(cat_column).mean(numeric_only=True).reset_index()

            joined = self.query.add_suffix("_q").merge(
                table_grouped.add_suffix("_t"),
                left_on=str(query_col_name) + "_q",
                right_on=str(cat_column) + "_t",
                how="inner",
            )

            # Group numeric columns by suffix
            joined_cols = joined.select_dtypes(include=["float64", "int64"]).columns
            left_cols = [col for col in joined_cols if col.endswith("_q")]
            right_cols = [col for col in joined_cols if col.endswith("_t")]

            # Calculate correlation
            for query_col, table_col in product(left_cols, right_cols):
                joined_no_nan = joined.dropna(subset=table_col)
                joinability = len(joined_no_nan[table_col])
                if joinability < 2 or joined_no_nan[table_col].var() <= 1e-10:
                    corr = 0
                else:
                    corr = stats.pearsonr(joined_no_nan[query_col], joined_no_nan[table_col])[0]
                percentual_joinability = joinability / len(self.query)
                result.loc[len(result)] = [
                    f"{cat_column}_|_{table_col}_|_{table.columns.name}",
                    percentual_joinability,
                    corr
                ]

        return result
