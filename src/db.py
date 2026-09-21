from pathlib import Path

import duckdb


def get_conn(path: str | Path = "data/retail.duckdb") -> duckdb.DuckDBPyConnection:
    """Open DuckDB with built-in pandas registration and fetchdf integration."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path))


def load_raw(path: str | Path = "data/retail.duckdb", raw_dir: str | Path = "data/raw") -> int:
    """Persist M5, ranking items by sales in selected stores from 2015 onward."""
    files = [Path(raw_dir) / name for name in (
        "sales_train_evaluation.csv", "sell_prices.csv", "calendar.csv")]
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(f"Missing M5 input: {file}")
    with get_conn(path) as conn:
        for name, file in zip(("raw_sales", "raw_prices", "raw_calendar"), files):
            conn.execute(f"CREATE TEMP VIEW {name} AS SELECT * FROM read_csv_auto({quote_literal(str(file))})")
        conn.execute("""
            CREATE TEMP TABLE long_sales AS
            WITH melted AS (
                UNPIVOT (SELECT * FROM raw_sales WHERE store_id IN ('CA_1', 'CA_2', 'TX_1'))
                ON COLUMNS('^d_[0-9]+$') INTO NAME d VALUE units
            )
            SELECT s.item_id, s.store_id, s.cat_id, CAST(c.date AS DATE) AS date,
                   c.wm_yr_wk, s.units
            FROM melted s JOIN raw_calendar c USING (d)
            WHERE CAST(c.date AS DATE) >= DATE '2015-01-01'
        """)
        conn.execute("""
            CREATE OR REPLACE TABLE sales_clean AS
            WITH top_items AS (
                SELECT item_id FROM long_sales GROUP BY item_id
                ORDER BY SUM(units) DESC, item_id LIMIT 200
            )
            SELECT s.*, p.sell_price FROM long_sales s
            JOIN top_items USING (item_id)
            LEFT JOIN raw_prices p USING (store_id, item_id, wm_yr_wk)
            ORDER BY s.store_id, s.item_id, s.date
        """)
        return conn.execute("SELECT count(*) FROM sales_clean").fetchone()[0]


def connect(database: str | Path = ":memory:", read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection for profiling and local transformations."""
    return duckdb.connect(str(database), read_only=read_only)


def register_parquet(
    connection: duckdb.DuckDBPyConnection,
    table_name: str,
    parquet_path: str | Path,
) -> duckdb.DuckDBPyConnection:
    """Register a Parquet file as a DuckDB view."""
    table = quote_identifier(table_name)
    path = quote_literal(str(parquet_path))
    connection.execute(
        f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet({path})",
    )
    return connection


def profile_table(connection: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, object]:
    """Return basic table-level and column-level profile metadata."""
    table = quote_identifier(table_name)
    row_count = connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    columns = connection.execute(f"DESCRIBE {table}").fetchdf().to_dict(orient="records")
    nulls = {}
    for column in columns:
        name = column["column_name"]
        quoted_name = quote_identifier(name)
        nulls[name] = connection.execute(
            f"SELECT count(*) FROM {table} WHERE {quoted_name} IS NULL"
        ).fetchone()[0]

    return {"row_count": row_count, "columns": columns, "null_counts": nulls}


def quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def quote_literal(value: str) -> str:
    return f"'{value.replace(chr(39), chr(39) * 2)}'"
