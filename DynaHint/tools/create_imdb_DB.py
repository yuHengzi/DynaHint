
from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

import psycopg2
from psycopg2 import OperationalError, ProgrammingError, sql
from psycopg2.extras import execute_values


SOURCE_DB = "imdb"
SPLIT_YEARS = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
TARGET_SUFFIX = "cut"
PG_HOST = "localhost"
PG_HOST = os.getenv("DYNAHINT_PG_HOST", PG_HOST)
PG_PORT = os.getenv("DYNAHINT_PG_PORT", "5433")
PG_USER = os.getenv("DYNAHINT_PG_USER", "")
PG_PASSWORD = os.getenv("DYNAHINT_PG_PASSWORD") or None
SCHEMA = "public"
BATCH_SIZE = 5000
RESTORE_FOREIGN_KEYS = False
TITLE_EXTRA_FILTER = None

ALL_TABLES = [
    "aka_name",
    "aka_title",
    "cast_info",
    "char_name",
    "comp_cast_type",
    "company_name",
    "company_type",
    "complete_cast",
    "info_type",
    "keyword",
    "kind_type",
    "link_type",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
    "movie_link",
    "name",
    "person_info",
    "role_type",
    "title",
]

FACT_TABLES_BY_MOVIE_ID = {
    "aka_title",
    "cast_info",
    "complete_cast",
    "movie_companies",
    "movie_info",
    "movie_info_idx",
    "movie_keyword",
}

DICTIONARY_TABLES = {
    "comp_cast_type",
    "company_type",
    "info_type",
    "kind_type",
    "link_type",
    "role_type",
}

ENTITY_TABLES = {
    "aka_name",
    "char_name",
    "company_name",
    "keyword",
    "name",
    "person_info",
}


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


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
    config = make_db_config("postgres")
    conn = connect_postgres(config)
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


def title_filter_sql(split_year: int) -> str:
    clauses = [f"production_year BETWEEN 1880 AND {int(split_year)}"]
    if TITLE_EXTRA_FILTER:
        clauses.append(TITLE_EXTRA_FILTER)
    return " AND ".join(clauses)


def build_select_query(table_name: str, split_year: int) -> str:
    title_filter = title_filter_sql(split_year)
    title_subquery = f"SELECT id FROM {SCHEMA}.title WHERE {title_filter}"
    person_id_subquery = (
        f"SELECT DISTINCT person_id FROM {SCHEMA}.cast_info "
        f"WHERE person_id IS NOT NULL AND movie_id IN ({title_subquery})"
    )
    person_role_subquery = (
        f"SELECT DISTINCT person_role_id FROM {SCHEMA}.cast_info "
        f"WHERE person_role_id IS NOT NULL AND movie_id IN ({title_subquery})"
    )
    company_id_subquery = (
        f"SELECT DISTINCT company_id FROM {SCHEMA}.movie_companies "
        f"WHERE company_id IS NOT NULL AND movie_id IN ({title_subquery})"
    )
    keyword_id_subquery = (
        f"SELECT DISTINCT keyword_id FROM {SCHEMA}.movie_keyword "
        f"WHERE keyword_id IS NOT NULL AND movie_id IN ({title_subquery})"
    )

    if table_name == "title":
        return f"SELECT * FROM {SCHEMA}.title WHERE {title_filter}"
    if table_name in FACT_TABLES_BY_MOVIE_ID:
        return (
            f"SELECT * FROM {SCHEMA}.{table_name} "
            f"WHERE movie_id IN ({title_subquery})"
        )
    if table_name == "movie_link":
        return (
            f"SELECT * FROM {SCHEMA}.movie_link "
            f"WHERE linked_movie_id IN ({title_subquery})"
        )
    if table_name in DICTIONARY_TABLES:
        return f"SELECT * FROM {SCHEMA}.{table_name}"
    if table_name == "name":
        return f"SELECT * FROM {SCHEMA}.name WHERE id IN ({person_id_subquery})"
    if table_name in {"aka_name", "person_info"}:
        return (
            f"SELECT * FROM {SCHEMA}.{table_name} "
            f"WHERE person_id IN ({person_id_subquery})"
        )
    if table_name == "char_name":
        return f"SELECT * FROM {SCHEMA}.char_name WHERE id IN ({person_role_subquery})"
    if table_name == "company_name":
        return f"SELECT * FROM {SCHEMA}.company_name WHERE id IN ({company_id_subquery})"
    if table_name == "keyword":
        return f"SELECT * FROM {SCHEMA}.keyword WHERE id IN ({keyword_id_subquery})"
    raise ValueError(f"Undefined table copy rule: {table_name}")


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


def copy_single_table(source_conn, target_conn, table_name: str, split_year: int) -> int:
    select_query = build_select_query(table_name, split_year)
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


def validate_target_database(source_conn, target_conn, split_year: int) -> Dict[str, object]:
    table_count = fetch_scalar(
        target_conn,
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = %s;",
        (SCHEMA,),
    )
    title_stats = fetch_row(
        target_conn,
        """
        SELECT min(production_year), max(production_year), count(*)
        FROM public.title;
        """,
    )
    invalid_year_count = fetch_scalar(
        target_conn,
        "SELECT count(*) FROM public.title WHERE production_year > %s;",
        (split_year,),
    )
    invalid_kind_count = None
    if TITLE_EXTRA_FILTER == "kind_id = 1":
        invalid_kind_count = fetch_scalar(
            target_conn,
            "SELECT count(*) FROM public.title WHERE kind_id <> 1 OR kind_id IS NULL;",
        )
    movie_link_count = fetch_scalar(target_conn, "SELECT count(*) FROM public.movie_link;")
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
    smoke_cursor = target_conn.cursor()
    try:
        smoke_cursor.execute(
            """
            EXPLAIN
            SELECT t.title, lt.link
            FROM public.movie_link AS ml
            JOIN public.title AS t
              ON ml.linked_movie_id = t.id
            JOIN public.link_type AS lt
              ON ml.link_type_id = lt.id
            LIMIT 5;
            """
        )
        smoke_ok = True
    except Exception:
        smoke_ok = False
    finally:
        smoke_cursor.close()

    shrink_checks = {}
    for table_name in sorted(ENTITY_TABLES):
        source_count = fetch_scalar(source_conn, f"SELECT count(*) FROM {SCHEMA}.{table_name};")
        target_count = fetch_scalar(target_conn, f"SELECT count(*) FROM {SCHEMA}.{table_name};")
        shrink_checks[table_name] = {
            "source": source_count,
            "target": target_count,
            "shrunk": target_count <= source_count,
        }

    return {
        "table_count": table_count,
        "title_min_year": title_stats[0],
        "title_max_year": title_stats[1],
        "title_count": title_stats[2],
        "invalid_year_count": invalid_year_count,
        "invalid_kind_count": invalid_kind_count,
        "movie_link_count": movie_link_count,
        "primary_key_count": primary_key_count,
        "index_count": index_count,
        "smoke_ok": smoke_ok,
        "entity_shrink_checks": shrink_checks,
    }


def build_single_cut_database(source_db: str, split_year: int, target_db: str) -> Dict[str, object]:
    summary = {
        "split_year": split_year,
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
                row_count = copy_single_table(source_conn, target_conn, table_name, split_year)
            except Exception as exc:
                target_conn.rollback()
                summary["failed_stage"] = f"copy_table:{table_name}"
                summary["error"] = str(exc)
                raise
            summary["table_rows"][table_name] = row_count

        summary.update(validate_target_database(source_conn, target_conn, split_year))
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
    print("Build result summary")
    print(divider)
    for result in results:
        if result["status"] == "success":
            print(
                f"{result['target_db']}: SUCCESS | "
                f"title={result.get('title_count')} | "
                f"movie_link={result.get('movie_link_count')} | "
                f"pks={result.get('primary_key_count')} | "
                f"indexes={result.get('index_count')} | "
                f"title_year={result.get('title_min_year')}..{result.get('title_max_year')}"
            )
        else:
            print(
                f"{result['target_db']}: FAILED | "
                f"stage={result.get('failed_stage')} | "
                f"error={result.get('error')}"
            )


def build_cut_databases(source_db: str, split_years: List[int], target_suffix: str = "cut") -> List[Dict[str, object]]:
    results = []
    for split_year in split_years:
        target_db = f"{source_db}{split_year}{target_suffix}"
        divider = "=" * 80
        print("\n" + divider)
        print(f"Start building drift database: {target_db}")
        print(divider)
        result = build_single_cut_database(source_db, split_year, target_db)
        results.append(result)
        if result["status"] == "success":
            log(
                f"{target_db} build completed: table_count={result['table_count']}，"
                f"title={result['title_count']}, movie_link={result['movie_link_count']}"
            )
        else:
            log(f"{target_db} build failed. Continue with the next year.")
    print_summary(results)
    return results


if __name__ == "__main__":
    build_cut_databases(SOURCE_DB, SPLIT_YEARS, TARGET_SUFFIX)
