
from __future__ import annotations

from typing import Dict, Iterable, List

import create_stats_DB as base


SOURCE_DB = base.SOURCE_DB
SCHEMA = base.SCHEMA
ALL_TABLES = base.ALL_TABLES
TARGET_SUFFIX = ""

CUT_SPECS: List[tuple[str, str]] = [
    ("200912", "2009-12-31 23:59:59"),
    ("201006", "2010-06-30 23:59:59"),
    ("201012", "2010-12-31 23:59:59"),
    ("201106", "2011-06-30 23:59:59"),
    ("201112", "2011-12-31 23:59:59"),
    ("201206", "2012-06-30 23:59:59"),
    ("201212", "2012-12-31 23:59:59"),
    ("201306", "2013-06-30 23:59:59"),
    ("201312", "2013-12-31 23:59:59"),
]


def log(message: str) -> None:
    base.log(message)


def target_db_name(source_db: str, cut_label: str) -> str:
    return f"{source_db}_{cut_label}"


def build_select_query(table_name: str, cutoff: str) -> str:
    return base.build_select_query(table_name, cutoff)


def copy_single_table(source_conn, target_conn, table_name: str, cutoff: str) -> int:
    select_query = build_select_query(table_name, cutoff)
    columns = base.get_table_columns(source_conn, table_name)
    base.recreate_table_structure(source_conn, target_conn, table_name)
    row_count = base.stream_copy_data(
        source_conn,
        target_conn,
        table_name,
        select_query,
        [col[0] for col in columns],
    )
    pk_count = base.restore_primary_key(source_conn, target_conn, table_name)
    index_count = base.restore_nonpk_indexes(source_conn, target_conn, table_name)
    fk_count = base.restore_foreign_keys(source_conn, target_conn, table_name)
    base.analyze_table(target_conn, table_name)
    target_conn.commit()
    log(
        f"Copied table data: {table_name} ({row_count} rows, primary keys={pk_count}, indexes={index_count}, foreign keys={fk_count})"
    )
    return row_count


def validate_target_database(target_conn, cutoff: str) -> Dict[str, object]:
    return base.validate_target_database(target_conn, cutoff)


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

    if not base.create_database_if_needed(target_db):
        summary["failed_stage"] = "create_database"
        summary["error"] = "failed to create database"
        return summary

    source_conn = base.connect_postgres(base.make_db_config(source_db))
    target_conn = base.connect_postgres(base.make_db_config(target_db))
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
    print("STATS half-year cut build result summary")
    print(divider)
    for result in results:
        if result["status"] == "success":
            print(
                f"{result['target_db']}: SUCCESS | "
                f"cutoff={result.get('cutoff')} | "
                f"posts={result.get('post_count')} | "
                f"post_time={result.get('post_min_time')}..{result.get('post_max_time')} | "
                f"leak_postlinks={result.get('postlink_leak_count')} | "
                f"pks={result.get('primary_key_count')} | "
                f"indexes={result.get('index_count')}"
            )
        else:
            print(
                f"{result['target_db']}: FAILED | "
                f"stage={result.get('failed_stage')} | "
                f"error={result.get('error')}"
            )


def build_cut_databases(source_db: str, cut_specs: List[tuple[str, str]]) -> List[Dict[str, object]]:
    results = []
    for cut_label, cutoff in cut_specs:
        target_db = target_db_name(source_db, cut_label)
        divider = "=" * 80
        print("\n" + divider)
        print(f"Start building half-year drift database: {target_db}，cutoff={cutoff}")
        print(divider)
        result = build_single_cut_database(source_db, cut_label, cutoff, target_db)
        results.append(result)
        if result["status"] == "success":
            log(
                f"{target_db} build completed: table_count={result['table_count']}，"
                f"posts={result['post_count']}，max_time={result['post_max_time']}"
            )
        else:
            log(f"{target_db} build failed. Continue with the next cut.")
    print_summary(results)
    return results


if __name__ == "__main__":
    build_cut_databases(SOURCE_DB, CUT_SPECS)
