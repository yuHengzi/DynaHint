
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import psycopg2
from psycopg2 import OperationalError, ProgrammingError, sql
from psycopg2.extras import execute_values


SOURCE_DB = "tpcds10g"
CUT_SPECS: List[Tuple[str, str]] = [
    ("199806", "1998-06-30"),
    ("199812", "1998-12-31"),
    ("199906", "1999-06-30"),
    ("199912", "1999-12-31"),
    ("200006", "2000-06-30"),
    ("200012", "2000-12-31"),
    ("200106", "2001-06-30"),
    ("200112", "2001-12-31"),
    ("200206", "2002-06-30"),
]

PG_HOST = "localhost"
PG_HOST = os.getenv("DYNAHINT_PG_HOST", PG_HOST)
PG_PORT = os.getenv("DYNAHINT_PG_PORT", "5433")
PG_USER = os.getenv("DYNAHINT_PG_USER", "")
PG_PASSWORD = os.getenv("DYNAHINT_PG_PASSWORD") or None
SCHEMA = "public"
BATCH_SIZE = 5000
RESTORE_FOREIGN_KEYS = False

FACT_DATE_KEYS = {
    "store_sales": "ss_sold_date_sk",
    "catalog_sales": "cs_sold_date_sk",
    "web_sales": "ws_sold_date_sk",
    "store_returns": "sr_returned_date_sk",
    "catalog_returns": "cr_returned_date_sk",
    "web_returns": "wr_returned_date_sk",
    "inventory": "inv_date_sk",
}

ALL_TABLES = [
    "call_center",
    "catalog_page",
    "catalog_returns",
    "catalog_sales",
    "customer",
    "customer_address",
    "customer_demographics",
    "date_dim",
    "dbgen_version",
    "household_demographics",
    "income_band",
    "inventory",
    "item",
    "promotion",
    "reason",
    "ship_mode",
    "store",
    "store_returns",
    "store_sales",
    "time_dim",
    "warehouse",
    "web_page",
    "web_returns",
    "web_sales",
    "web_site",
]


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def target_db_name(source_db: str, cut_label: str) -> str:
    return f"{source_db}_{cut_label}"


def make_db_config(dbname: str) -> Dict[str, str]:
    config = {
        "dbname": dbname,
        "user": PG_USER,
        "host": PG_HOST,
        "port": PG_PORT,
    }
    if PG_PASSWORD:
        config["password"] = PG_PASSWORD
    return config


def connect_postgres(config: Dict[str, str]):
    conn = None
    try:
        conn = psycopg2.connect(**config)
        conn.autocommit = False
        log(f"Connected to database: {config['dbname']}")
        return conn
    except OperationalError as exc:
        log(f"Database connection failed: {exc}")
        return None


def create_database_if_needed(dbname: str) -> bool:
    conn = connect_postgres(make_db_config("postgres"))
    if conn is None:
        return False
    conn.autocommit = True
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s;", (dbname,))
        exists = cursor.fetchone() is not None
        if exists:
            log(f"Database already exists: {dbname}")
            return True
        cursor.execute(sql.SQL("CREATE DATABASE {};").format(sql.Identifier(dbname)))
        log(f"Created database: {dbname}")
        return True
    except ProgrammingError as exc:
        log(f"Failed to create database: {exc}")
        return False
    finally:
        cursor.close()
        conn.close()


def selected_dates_subquery(cutoff: str) -> str:
    return f"SELECT d_date_sk FROM {SCHEMA}.date_dim WHERE d_date <= DATE '{cutoff}'"


def build_select_query(table_name: str, cutoff: str) -> str:
    if table_name in FACT_DATE_KEYS:
        date_key = FACT_DATE_KEYS[table_name]
        return (
            f"SELECT * FROM {SCHEMA}.{table_name} "
            f"WHERE {date_key} IN ({selected_dates_subquery(cutoff)})"
        )
    return f"SELECT * FROM {SCHEMA}.{table_name}"


def get_table_columns(source_conn, table_name: str) -> List[Tuple[str, str, bool, Optional[str]]]:
    cursor = source_conn.cursor()
    try:
        cursor.execute(
            """
            SELECT
                a.attname,
                pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
                a.attnotnull,
                pg_get_expr(ad.adbin, ad.adrelid) AS default_expr
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            LEFT JOIN pg_attrdef ad
                ON ad.adrelid = a.attrelid
               AND ad.adnum = a.attnum
            WHERE n.nspname = %s
              AND c.relname = %s
              AND a.attnum > 0
              AND NOT a.attisdropped
            ORDER BY a.attnum;
            """,
            (SCHEMA, table_name),
        )
        return cursor.fetchall()
    finally:
        cursor.close()


def recreate_table_structure(source_conn, target_conn, table_name: str) -> None:
    columns = get_table_columns(source_conn, table_name)
    if not columns:
        raise RuntimeError(f"Source database does not contain table: {table_name}")

    target_cursor = target_conn.cursor()
    try:
        target_cursor.execute(
            sql.SQL("DROP TABLE IF EXISTS {}.{} CASCADE;").format(
                sql.Identifier(SCHEMA),
                sql.Identifier(table_name),
            )
        )

        column_defs = []
        for col_name, data_type, is_not_null, default_expr in columns:
            parts = [sql.Identifier(col_name), sql.SQL(data_type)]
            if default_expr and not default_expr.startswith("nextval("):
                parts.extend([sql.SQL("DEFAULT"), sql.SQL(default_expr)])
            if is_not_null:
                parts.append(sql.SQL("NOT NULL"))
            column_defs.append(sql.SQL(" ").join(parts))

        create_sql = sql.SQL("CREATE TABLE {}.{} ({});").format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table_name),
            sql.SQL(", ").join(column_defs),
        )
        target_cursor.execute(create_sql)
        log(f"Copied table schema: {table_name}")
    finally:
        target_cursor.close()


def stream_copy_data(
    source_conn,
    target_conn,
    table_name: str,
    select_query: str,
    column_names: List[str],
) -> int:
    source_cursor_name = f"src_{table_name}_{int(time.time() * 1000)}"
    source_cursor = source_conn.cursor(name=source_cursor_name)
    target_cursor = target_conn.cursor()
    total_rows = 0
    try:
        source_cursor.itersize = BATCH_SIZE
        source_cursor.execute(select_query)
        insert_sql = sql.SQL("INSERT INTO {}.{} ({}) VALUES %s").format(
            sql.Identifier(SCHEMA),
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in column_names),
        ).as_string(target_conn)

        while True:
            rows = source_cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break
            execute_values(target_cursor, insert_sql, rows, page_size=1000)
            total_rows += len(rows)
        return total_rows
    finally:
        source_cursor.close()
        target_cursor.close()


def restore_primary_key(source_conn, target_conn, table_name: str) -> int:
    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()
    restored = 0
    try:
        source_cursor.execute(
            """
            SELECT con.conname, pg_get_constraintdef(con.oid)
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = rel.relnamespace
            WHERE n.nspname = %s
              AND rel.relname = %s
              AND con.contype = 'p';
            """,
            (SCHEMA, table_name),
        )
        for constraint_name, constraint_def in source_cursor.fetchall():
            target_cursor.execute(
                sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} {};").format(
                    sql.Identifier(SCHEMA),
                    sql.Identifier(table_name),
                    sql.Identifier(constraint_name),
                    sql.SQL(constraint_def),
                )
            )
            restored += 1
        return restored
    finally:
        source_cursor.close()
        target_cursor.close()


def restore_nonpk_indexes(source_conn, target_conn, table_name: str) -> int:
    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()
    restored = 0
    try:
        source_cursor.execute(
            """
            SELECT pg_get_indexdef(i.indexrelid)
            FROM pg_index i
            JOIN pg_class tbl ON tbl.oid = i.indrelid
            JOIN pg_namespace n ON n.oid = tbl.relnamespace
            JOIN pg_class idx ON idx.oid = i.indexrelid
            LEFT JOIN pg_constraint con ON con.conindid = i.indexrelid
            WHERE n.nspname = %s
              AND tbl.relname = %s
              AND con.oid IS NULL
            ORDER BY idx.relname;
            """,
            (SCHEMA, table_name),
        )
        for (index_def,) in source_cursor.fetchall():
            statement = index_def
            if statement.startswith("CREATE UNIQUE INDEX "):
                statement = statement.replace("CREATE UNIQUE INDEX ", "CREATE UNIQUE INDEX IF NOT EXISTS ", 1)
            elif statement.startswith("CREATE INDEX "):
                statement = statement.replace("CREATE INDEX ", "CREATE INDEX IF NOT EXISTS ", 1)
            target_cursor.execute(statement)
            restored += 1
        return restored
    finally:
        source_cursor.close()
        target_cursor.close()


def restore_foreign_keys(source_conn, target_conn, table_name: str) -> int:
    if not RESTORE_FOREIGN_KEYS:
        return 0

    source_cursor = source_conn.cursor()
    target_cursor = target_conn.cursor()
    restored = 0
    try:
        source_cursor.execute(
            """
            SELECT con.conname, pg_get_constraintdef(con.oid)
            FROM pg_constraint con
            JOIN pg_class rel ON rel.oid = con.conrelid
            JOIN pg_namespace n ON n.oid = rel.relnamespace
            WHERE n.nspname = %s
              AND rel.relname = %s
              AND con.contype = 'f';
            """,
            (SCHEMA, table_name),
        )
        for constraint_name, constraint_def in source_cursor.fetchall():
            target_cursor.execute(
                sql.SQL("ALTER TABLE {}.{} ADD CONSTRAINT {} {};").format(
                    sql.Identifier(SCHEMA),
                    sql.Identifier(table_name),
                    sql.Identifier(constraint_name),
                    sql.SQL(constraint_def),
                )
            )
            restored += 1
        return restored
    finally:
        source_cursor.close()
        target_cursor.close()


def analyze_table(target_conn, table_name: str) -> None:
    cursor = target_conn.cursor()
    try:
        cursor.execute(
            sql.SQL("ANALYZE {}.{};").format(
                sql.Identifier(SCHEMA),
                sql.Identifier(table_name),
            )
        )
    finally:
        cursor.close()


def copy_single_table(source_conn, target_conn, table_name: str, cutoff: str) -> int:
    select_query = build_select_query(table_name, cutoff)
    columns = get_table_columns(source_conn, table_name)
    recreate_table_structure(source_conn, target_conn, table_name)
    row_count = stream_copy_data(
        source_conn,
        target_conn,
        table_name,
        select_query,
        [col[0] for col in columns],
    )
    pk_count = restore_primary_key(source_conn, target_conn, table_name)
    index_count = restore_nonpk_indexes(source_conn, target_conn, table_name)
    fk_count = restore_foreign_keys(source_conn, target_conn, table_name)
    analyze_table(target_conn, table_name)
    target_conn.commit()
    log(
        f"Copied table data: {table_name} ({row_count} rows, primary keys={pk_count}, indexes={index_count}, foreign keys={fk_count})"
    )
    return row_count


def fetch_scalar(conn, query: str, params: Optional[Tuple] = None):
    cursor = conn.cursor()
    try:
        cursor.execute(query, params or ())
        row = cursor.fetchone()
        return None if row is None else row[0]
    finally:
        cursor.close()


def fetch_row(conn, query: str, params: Optional[Tuple] = None):
    cursor = conn.cursor()
    try:
        cursor.execute(query, params or ())
        return cursor.fetchone()
    finally:
        cursor.close()


def validate_target_database(target_conn, cutoff: str) -> Dict[str, object]:
    table_count = fetch_scalar(
        target_conn,
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s;",
        (SCHEMA,),
    )
    fact_counts = {}
    invalid_fact_counts = {}
    for table_name, date_key in FACT_DATE_KEYS.items():
        fact_counts[table_name] = fetch_scalar(
            target_conn,
            sql.SQL("SELECT count(*) FROM {}.{};").format(
                sql.Identifier(SCHEMA),
                sql.Identifier(table_name),
            ).as_string(target_conn),
        )
        invalid_fact_counts[table_name] = fetch_scalar(
            target_conn,
            f"""
            SELECT count(*)
            FROM {SCHEMA}.{table_name} f
            JOIN {SCHEMA}.date_dim d ON d.d_date_sk = f.{date_key}
            WHERE d.d_date > %s::date;
            """,
            (cutoff,),
        )

    primary_key_count = fetch_scalar(
        target_conn,
        """
        SELECT count(*)
        FROM pg_constraint con
        JOIN pg_namespace n ON n.oid = con.connamespace
        WHERE n.nspname = %s
          AND con.contype = 'p';
        """,
        (SCHEMA,),
    )
    index_count = fetch_scalar(
        target_conn,
        "SELECT count(*) FROM pg_indexes WHERE schemaname = %s;",
        (SCHEMA,),
    )
    return {
        "table_count": table_count,
        "fact_counts": fact_counts,
        "invalid_fact_counts": invalid_fact_counts,
        "primary_key_count": primary_key_count,
        "index_count": index_count,
    }


def build_single_cut_database(source_db: str, cut_label: str, cutoff: str, target_db: str) -> Dict[str, object]:
    summary = {
        "cut_label": cut_label,
        "cutoff": cutoff,
        "source_db": source_db,
        "target_db": target_db,
        "status": "failed",
        "failed_stage": None,
        "error": None,
        "table_rows": {},
    }

    if not create_database_if_needed(target_db):
        summary["failed_stage"] = "create_database"
        summary["error"] = "failed to create database"
        return summary

    source_conn = connect_postgres(make_db_config(source_db))
    target_conn = connect_postgres(make_db_config(target_db))
    if source_conn is None or target_conn is None:
        summary["failed_stage"] = "connect_database"
        summary["error"] = "failed to connect source or target database"
        if source_conn:
            source_conn.close()
        if target_conn:
            target_conn.close()
        return summary

    try:
        for table_name in ALL_TABLES:
            log(f"Processing table: {table_name}")
            try:
                row_count = copy_single_table(source_conn, target_conn, table_name, cutoff)
            except Exception as exc:
                target_conn.rollback()
                summary["failed_stage"] = f"copy_table:{table_name}"
                summary["error"] = str(exc)
                raise
            summary["table_rows"][table_name] = row_count

        summary.update(validate_target_database(target_conn, cutoff))
        summary["status"] = "success"
        summary["failed_stage"] = None
        return summary
    except Exception as exc:
        log(f"Failed to build database: {target_db}, stage={summary['failed_stage']}, error={exc}")
        return summary
    finally:
        source_conn.close()
        target_conn.close()


def print_summary(results: Iterable[Dict[str, object]]) -> None:
    divider = "=" * 80
    print("\n" + divider)
    print("TPC-DS half-year cut build result summary")
    print(divider)
    for result in results:
        if result["status"] == "success":
            invalid_total = sum(result.get("invalid_fact_counts", {}).values())
            fact_counts = result.get("fact_counts", {})
            print(
                f"{result['target_db']}: SUCCESS | cutoff={result.get('cutoff')} | "
                f"store_sales={fact_counts.get('store_sales')} | "
                f"catalog_sales={fact_counts.get('catalog_sales')} | "
                f"web_sales={fact_counts.get('web_sales')} | "
                f"inventory={fact_counts.get('inventory')} | "
                f"invalid_facts={invalid_total} | "
                f"pks={result.get('primary_key_count')} | indexes={result.get('index_count')}"
            )
        else:
            print(
                f"{result['target_db']}: FAILED | "
                f"stage={result.get('failed_stage')} | error={result.get('error')}"
            )


def build_cut_databases(
    source_db: str,
    cut_specs: List[Tuple[str, str]],
) -> List[Dict[str, object]]:
    results = []
    for cut_label, cutoff in cut_specs:
        target_db = target_db_name(source_db, cut_label)
        divider = "=" * 80
        print("\n" + divider)
        print(f"Start building TPC-DS half-year drift database: {target_db}，cutoff={cutoff}")
        print(divider)
        result = build_single_cut_database(source_db, cut_label, cutoff, target_db)
        results.append(result)
        if result["status"] == "success":
            log(
                f"{target_db} build completed: table_count={result['table_count']}，"
                f"store_sales={result['fact_counts'].get('store_sales')}，"
                f"inventory={result['fact_counts'].get('inventory')}"
            )
        else:
            log(f"{target_db} build failed. Continue with the next cut.")
    print_summary(results)
    return results


if __name__ == "__main__":
    build_cut_databases(SOURCE_DB, CUT_SPECS)
