SELECT 
  Md5(
    Concat(
      category, 
      (
        CASE WHEN mue > value THEN '+1' ELSE '-1' END
      )
    )
  ) AS term, 
  sketches.tableid 
FROM 
  (
    SELECT 
      tableid, 
      category, 
      value 
    FROM 
      (
        SELECT 
          categorical.tableid AS tableid, 
          categorical.cellvalue AS category, 
          Avg(
            Cast(numerical.cellvalue AS DOUBLE)
          ) AS value, 
          Row_number() OVER (
            partition BY categorical.tableid 
            ORDER BY 
              Md5_number(categorical.cellvalue)
          ) AS rownumber 
        FROM 
          alltables Categorical 
          JOIN alltables Numerical ON categorical.tableid = numerical.tableid 
          AND categorical.rowid = numerical.rowid 
          JOIN (
            SELECT 
              categorical.tableid, 
              categorical_column_id, 
              numerical_column_id 
            FROM 
              (
                SELECT 
                  tableid, 
                  columnid AS categorical_column_id 
                FROM 
                  alltables 
                GROUP BY 
                  tableid, 
                  columnid 
                HAVING 
                  count(
                    try_cast(cellvalue as DOUBLE)
                  ) = 0
              ) AS categorical 
              JOIN (
                SELECT 
                  tableid, 
                  columnid AS numerical_column_id 
                FROM 
                  alltables 
                GROUP BY 
                  tableid, 
                  columnid 
                HAVING 
                  count(*) = count(
                    try_cast(cellvalue AS DOUBLE)
                  )
              ) AS numerical ON categorical.tableid = numerical.tableid
          ) pairs ON categorical.tableid = pairs.tableid 
          AND categorical.columnid = pairs.categorical_column_id 
          AND numerical.columnid = pairs.numerical_column_id 
        WHERE 
          numerical.cellvalue != 'nan' 
        GROUP BY 
          categorical.tableid, 
          categorical.cellvalue
      ) AS t 
    WHERE 
      rownumber <= 3 
    ORDER BY 
      value
  ) AS sketches 
  JOIN (
    SELECT 
      tableid, 
      avg(value) AS mue 
    FROM 
      (
        SELECT 
          tableid, 
          category, 
          value 
        FROM 
          (
            SELECT 
              categorical.tableid AS tableid, 
              categorical.cellvalue AS category, 
              avg(
                cast(numerical.cellvalue AS DOUBLE)
              ) AS value, 
              row_number() OVER (
                partition BY categorical.tableid 
                ORDER BY 
                  md5_number(categorical.cellvalue)
              ) AS rownumber 
            FROM 
              alltables categorical 
              JOIN alltables numerical ON categorical.tableid = numerical.tableid 
              AND categorical.rowid = numerical.rowid 
              JOIN (
                SELECT 
                  categorical.tableid, 
                  categorical_column_id, 
                  numerical_column_id 
                FROM 
                  (
                    SELECT 
                      tableid, 
                      columnid AS categorical_column_id 
                    FROM 
                      alltables 
                    GROUP BY 
                      tableid, 
                      columnid 
                    HAVING 
                      count(
                        try_cast(cellvalue AS DOUBLE)
                      ) = 0
                  ) AS categorical 
                  JOIN (
                    SELECT 
                      tableid, 
                      columnid AS numerical_column_id 
                    FROM 
                      alltables 
                    GROUP BY 
                      tableid, 
                      columnid 
                    HAVING 
                      count(*) = count(
                        try_cast(cellvalue AS DOUBLE)
                      )
                  ) AS numerical ON categorical.tableid = numerical.tableid
              ) pairs ON categorical.tableid = pairs.tableid 
              AND categorical.columnid = pairs.categorical_column_id 
              AND numerical.columnid = pairs.numerical_column_id 
            WHERE 
              numerical.cellvalue != 'nan' 
            GROUP BY 
              categorical.tableid, 
              categorical.cellvalue
          ) AS t 
        WHERE 
          rownumber <= 3 
        ORDER BY 
          value
      ) 
    GROUP BY 
      tableid
  ) AS mues ON sketches.tableid = mues.tableid
