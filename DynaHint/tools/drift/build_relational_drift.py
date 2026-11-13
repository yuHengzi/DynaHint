#!/usr/bin/env python3
"""Build relation-safe drifted PostgreSQL databases from built-in profiles.

The tool samples a profile-specific root entity by strata, then materializes the
dependent rows needed to keep joins meaningful.  It supports IMDb/JOB, STATS,
TPC-DS, and TPC-H without depending on the NeurBench source tree.

This is relational resampling, not independent per-table DDPM synthesis.  That
distinction prevents generated foreign keys from silently changing join graphs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


class TargetDatabaseExistsError(RuntimeError):
    """Raised before an existing target database could be replaced implicitly."""


@dataclass(frozen=True)
class Relationship:
    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    nullable: bool = False


@dataclass(frozen=True)
class DriftProfile:
    name: str
    default_source_db: str
    tables: tuple[str, ...]
    relationships: tuple[Relationship, ...]
    root_from_sql: str
    root_id_sql: str
    root_value_sql: str
    root_category_sql: str
    default_bins: str
    bin_type: str
    root_table_is_filtered: bool = True


IMDB_TABLES = (
    "comp_cast_type", "company_type", "info_type", "kind_type", "link_type",
    "role_type", "title", "name", "char_name", "company_name", "keyword",
    "aka_name", "person_info", "aka_title", "cast_info", "complete_cast",
    "movie_companies", "movie_info", "movie_info_idx", "movie_keyword", "movie_link",
)

IMDB_RELATIONSHIPS = (
    Relationship("aka_name", "person_id", "name", "id"),
    Relationship("aka_title", "movie_id", "title", "id"),
    Relationship("aka_title", "episode_of_id", "title", "id", True),
    Relationship("aka_title", "kind_id", "kind_type", "id"),
    Relationship("cast_info", "person_id", "name", "id"),
    Relationship("cast_info", "movie_id", "title", "id"),
    Relationship("cast_info", "person_role_id", "char_name", "id", True),
    Relationship("cast_info", "role_id", "role_type", "id"),
    Relationship("complete_cast", "movie_id", "title", "id"),
    Relationship("complete_cast", "subject_id", "comp_cast_type", "id"),
    Relationship("complete_cast", "status_id", "comp_cast_type", "id"),
    Relationship("movie_companies", "movie_id", "title", "id"),
    Relationship("movie_companies", "company_id", "company_name", "id"),
    Relationship("movie_companies", "company_type_id", "company_type", "id"),
    Relationship("movie_info", "movie_id", "title", "id"),
    Relationship("movie_info", "info_type_id", "info_type", "id"),
    Relationship("movie_info_idx", "movie_id", "title", "id"),
    Relationship("movie_info_idx", "info_type_id", "info_type", "id"),
    Relationship("movie_keyword", "movie_id", "title", "id"),
    Relationship("movie_keyword", "keyword_id", "keyword", "id"),
    Relationship("movie_link", "movie_id", "title", "id"),
    Relationship("movie_link", "linked_movie_id", "title", "id"),
    Relationship("movie_link", "link_type_id", "link_type", "id"),
    Relationship("person_info", "person_id", "name", "id"),
    Relationship("person_info", "info_type_id", "info_type", "id"),
    Relationship("title", "kind_id", "kind_type", "id"),
    Relationship("title", "episode_of_id", "title", "id", True),
)

STATS_TABLES = (
    "users", "posts", "comments", "posthistory", "postlinks", "votes", "badges", "tags",
)

STATS_RELATIONSHIPS = (
    Relationship("posts", "owneruserid", "users", "id", True),
    Relationship("posts", "lasteditoruserid", "users", "id", True),
    Relationship("comments", "postid", "posts", "id"),
    Relationship("comments", "userid", "users", "id", True),
    Relationship("posthistory", "postid", "posts", "id"),
    Relationship("posthistory", "userid", "users", "id", True),
    Relationship("postlinks", "postid", "posts", "id"),
    Relationship("postlinks", "relatedpostid", "posts", "id"),
    Relationship("votes", "postid", "posts", "id"),
    Relationship("votes", "userid", "users", "id", True),
    Relationship("badges", "userid", "users", "id"),
)

TPCDS_FACT_DATE_KEYS = {
    "store_sales": "ss_sold_date_sk",
    "catalog_sales": "cs_sold_date_sk",
    "web_sales": "ws_sold_date_sk",
    "store_returns": "sr_returned_date_sk",
    "catalog_returns": "cr_returned_date_sk",
    "web_returns": "wr_returned_date_sk",
    "inventory": "inv_date_sk",
}

TPCDS_TABLES = (
    "call_center", "catalog_page", "customer", "customer_address",
    "customer_demographics", "date_dim", "dbgen_version", "household_demographics",
    "income_band", "item", "promotion", "reason", "ship_mode", "store", "time_dim",
    "warehouse", "web_page", "web_site", "catalog_returns", "catalog_sales", "inventory",
    "store_returns", "store_sales", "web_returns", "web_sales",
)

TPCDS_RELATIONSHIPS = tuple(
    Relationship(table, column, "date_dim", "d_date_sk", True)
    for table, column in TPCDS_FACT_DATE_KEYS.items()
)

TPCH_TABLES = ("region", "nation", "supplier", "customer", "part", "partsupp", "orders", "lineitem")

TPCH_RELATIONSHIPS = (
    Relationship("nation", "n_regionkey", "region", "r_regionkey"),
    Relationship("supplier", "s_nationkey", "nation", "n_nationkey"),
    Relationship("customer", "c_nationkey", "nation", "n_nationkey"),
    Relationship("partsupp", "ps_partkey", "part", "p_partkey"),
    Relationship("partsupp", "ps_suppkey", "supplier", "s_suppkey"),
    Relationship("orders", "o_custkey", "customer", "c_custkey"),
    Relationship("lineitem", "l_orderkey", "orders", "o_orderkey"),
    Relationship("lineitem", "l_partkey", "part", "p_partkey"),
    Relationship("lineitem", "l_suppkey", "supplier", "s_suppkey"),
)

PROFILES = {
    "imdb": DriftProfile(
        "imdb", "imdb", IMDB_TABLES, IMDB_RELATIONSHIPS,
        "public.title AS root LEFT JOIN public.kind_type AS category ON category.id = root.kind_id",
        "root.id", "root.production_year", "category.kind",
        "1950,1980,2000,2010", "integer",
    ),
    "stats": DriftProfile(
        "stats", "stats", STATS_TABLES, STATS_RELATIONSHIPS,
        "public.posts AS root", "root.id", "root.creationdate", "root.posttypeid::text",
        "2011-12-23,2012-09-11,2013-04-21,2014-03-25", "date",
    ),
    "tpcds": DriftProfile(
        "tpcds", "tpcds10g", TPCDS_TABLES, TPCDS_RELATIONSHIPS,
        "(SELECT * FROM public.date_dim WHERE d_date BETWEEN "
        "DATE '1998-01-01' AND DATE '2003-12-31') AS root",
        "root.d_date_sk", "root.d_date", "root.d_moy::text",
        "1998-12-31,1999-12-31,2000-12-31,2001-12-31,2002-12-31", "date", False,
    ),
    "tpch": DriftProfile(
        "tpch", "tpch", TPCH_TABLES, TPCH_RELATIONSHIPS,
        "public.orders AS root", "root.o_orderkey", "root.o_orderdate", "root.o_orderpriority",
        "1993-12-31,1994-12-31,1995-12-31,1996-12-31", "date",
    ),
}


def parse_bins(profile: DriftProfile, value: str) -> tuple[Any, ...]:
    raw = tuple(item.strip() for item in value.split(",") if item.strip())
    if not raw:
        raise ValueError("At least one bin boundary is required")
    try:
        parsed = tuple(int(item) for item in raw) if profile.bin_type == "integer" else tuple(
            dt.date.fromisoformat(item) for item in raw
        )
    except ValueError as error:
        expected = "integers" if profile.bin_type == "integer" else "ISO dates (YYYY-MM-DD)"
        raise ValueError(f"{profile.name} bin boundaries must be {expected}") from error
    if any(left >= right for left, right in zip(parsed, parsed[1:])):
        raise ValueError("Bin boundaries must be strictly increasing")
    return parsed


def _sql_literal(value: Any) -> str:
    if isinstance(value, dt.date):
        return f"DATE '{value.isoformat()}'"
    return str(int(value))


def stratum_expression(profile: DriftProfile, boundaries: Sequence[Any]) -> str:
    parts = [f"WHEN {profile.root_value_sql} IS NULL THEN 'unknown'"]
    previous = None
    for boundary in boundaries:
        label = str(boundary) if previous is None else f"{previous}..{boundary}"
        parts.append(
            f"WHEN {profile.root_value_sql} <= {_sql_literal(boundary)} THEN 'le_{label}'"
        )
        previous = boundary
    parts.append(f"ELSE 'gt_{boundaries[-1]}'")
    bucket = "CASE " + " ".join(parts) + " END"
    return f"({bucket} || '|' || COALESCE(({profile.root_category_sql})::text, 'unknown'))"


def allocate_stratum_quotas(
    capacities: Mapping[str, int], root_count: int, target_weights: Mapping[str, float] | None = None
) -> dict[str, int]:
    """Allocate roots by largest deficit while respecting each stratum capacity."""
    if root_count < 0:
        raise ValueError("root_count must not be negative")
    ordered = sorted(capacities)
    if any(not isinstance(capacities[key], int) or capacities[key] < 0 for key in ordered):
        raise ValueError("Stratum capacities must be non-negative integers")
    if root_count > sum(capacities.values()):
        raise ValueError("Requested root count exceeds available capacity")
    if not ordered:
        if root_count:
            raise ValueError("Cannot sample from an empty profile")
        return {}
    if target_weights is None:
        weights = {key: float(capacities[key]) for key in ordered}
    else:
        unknown = sorted(set(target_weights) - set(ordered))
        if unknown:
            raise ValueError(f"Target weights contain unknown strata: {', '.join(unknown)}")
        weights = {key: float(target_weights.get(key, 0.0)) for key in ordered}
        if any(weight < 0 for weight in weights.values()):
            raise ValueError("Target weights must not be negative")
    total_weight = sum(weights.values())
    if root_count and total_weight <= 0:
        raise ValueError("At least one target weight must be positive")
    ideal = {
        key: root_count * weights[key] / total_weight if total_weight else 0.0
        for key in ordered
    }
    quotas = {key: 0 for key in ordered}
    for _ in range(root_count):
        available = [key for key in ordered if quotas[key] < capacities[key]]
        chosen = max(available, key=lambda key: (ideal[key] - quotas[key], key))
        quotas[chosen] += 1
    return quotas


def jensen_shannon_divergence(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    keys = set(left) | set(right)
    left_total, right_total = sum(left.values()), sum(right.values())
    if left_total == right_total == 0:
        return 0.0
    if not left_total or not right_total:
        raise ValueError("Both distributions must be non-empty")
    result = 0.0
    for key in keys:
        p, q = left.get(key, 0) / left_total, right.get(key, 0) / right_total
        midpoint = (p + q) / 2
        if p:
            result += p * math.log(p / midpoint) / 2
        if q:
            result += q * math.log(q / midpoint) / 2
    return result


def build_anti_join_sql(relation: Relationship) -> str:
    nullable = f"child.{relation.child_column} IS NOT NULL AND " if relation.nullable else ""
    return (
        f"SELECT COUNT(*) FROM public.{relation.child_table} AS child "
        f"LEFT JOIN public.{relation.parent_table} AS parent "
        f"ON child.{relation.child_column} = parent.{relation.parent_column} "
        f"WHERE {nullable}parent.{relation.parent_column} IS NULL"
    )


def _selected_roots(column: str) -> str:
    return f"{column} IN (SELECT id FROM selected_root_ids)"


def _imdb_select(table: str, columns: Sequence[str] | None) -> str:
    static = {"comp_cast_type", "company_type", "info_type", "kind_type", "link_type", "role_type"}
    if table in static:
        return f"SELECT * FROM public.{table}"
    if table == "title":
        if columns and "episode_of_id" in columns:
            selected = [
                "CASE WHEN episode_of_id IS NULL OR " + _selected_roots("episode_of_id")
                + " THEN episode_of_id ELSE NULL END AS episode_of_id"
                if column == "episode_of_id" else column for column in columns
            ]
            return f"SELECT {', '.join(selected)} FROM public.title WHERE {_selected_roots('id')}"
        return f"SELECT * FROM public.title WHERE {_selected_roots('id')}"
    if table in {"cast_info", "complete_cast", "movie_companies", "movie_info", "movie_info_idx", "movie_keyword"}:
        return f"SELECT * FROM public.{table} WHERE {_selected_roots('movie_id')}"
    if table == "movie_link":
        return f"SELECT * FROM public.movie_link WHERE {_selected_roots('movie_id')} AND {_selected_roots('linked_movie_id')}"
    if table == "aka_title":
        return f"SELECT * FROM public.aka_title WHERE {_selected_roots('movie_id')} AND (episode_of_id IS NULL OR {_selected_roots('episode_of_id')})"
    people = "SELECT DISTINCT person_id FROM public.cast_info WHERE " + _selected_roots("movie_id")
    if table in {"name", "aka_name", "person_info"}:
        column = "id" if table == "name" else "person_id"
        return f"SELECT * FROM public.{table} WHERE {column} IN ({people})"
    if table == "char_name":
        return "SELECT * FROM public.char_name WHERE id IN (SELECT DISTINCT person_role_id FROM public.cast_info WHERE " + _selected_roots("movie_id") + " AND person_role_id IS NOT NULL)"
    if table == "company_name":
        return "SELECT * FROM public.company_name WHERE id IN (SELECT DISTINCT company_id FROM public.movie_companies WHERE " + _selected_roots("movie_id") + ")"
    if table == "keyword":
        return "SELECT * FROM public.keyword WHERE id IN (SELECT DISTINCT keyword_id FROM public.movie_keyword WHERE " + _selected_roots("movie_id") + ")"
    raise ValueError(f"Unsupported IMDb table: {table}")


def _stats_select(table: str) -> str:
    if table == "posts":
        return f"SELECT * FROM public.posts WHERE {_selected_roots('id')}"
    if table in {"comments", "posthistory", "votes"}:
        return f"SELECT * FROM public.{table} WHERE {_selected_roots('postid')}"
    if table == "postlinks":
        return f"SELECT * FROM public.postlinks WHERE {_selected_roots('postid')} AND {_selected_roots('relatedpostid')}"
    if table == "users":
        return "SELECT * FROM public.users WHERE id IN (SELECT id FROM selected_related_user_ids)"
    if table == "badges":
        return "SELECT * FROM public.badges WHERE userid IN (SELECT id FROM selected_related_user_ids)"
    if table == "tags":
        return "SELECT * FROM public.tags"
    raise ValueError(f"Unsupported STATS table: {table}")


def build_table_select(profile: DriftProfile, table: str, columns: Sequence[str] | None = None) -> str:
    if table not in profile.tables:
        raise ValueError(f"Table {table!r} is not part of profile {profile.name!r}")
    if profile.name == "imdb":
        return _imdb_select(table, columns)
    if profile.name == "stats":
        return _stats_select(table)
    if profile.name == "tpcds" and table in TPCDS_FACT_DATE_KEYS:
        return f"SELECT * FROM public.{table} WHERE {_selected_roots(TPCDS_FACT_DATE_KEYS[table])}"
    if profile.name == "tpch" and table in {"orders", "lineitem"}:
        key = "o_orderkey" if table == "orders" else "l_orderkey"
        return f"SELECT * FROM public.{table} WHERE {_selected_roots(key)}"
    return f"SELECT * FROM public.{table}"


def connection_kwargs(args: argparse.Namespace, dbname: str) -> dict[str, Any]:
    result: dict[str, Any] = {"host": args.host, "port": args.port, "dbname": dbname}
    if args.user:
        result["user"] = args.user
    if args.password is not None:
        result["password"] = args.password
    return result


def database_exists(args: argparse.Namespace, dbname: str) -> bool:
    with psycopg2.connect(**connection_kwargs(args, "postgres")) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT EXISTS (SELECT 1 FROM pg_database WHERE datname = %s)", (dbname,))
            return bool(cursor.fetchone()[0])


def list_column_order(connection) -> dict[str, tuple[str, ...]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema='public' ORDER BY table_name, ordinal_position"
        )
        result: dict[str, list[str]] = {}
        for table, column in cursor.fetchall():
            result.setdefault(table, []).append(column)
    return {table: tuple(columns) for table, columns in result.items()}


def validate_source_schema(connection, profile: DriftProfile) -> dict[str, tuple[str, ...]]:
    columns = list_column_order(connection)
    missing_tables = sorted(set(profile.tables) - set(columns))
    if missing_tables:
        raise ValueError(f"Source is missing {profile.name} table(s): {', '.join(missing_tables)}")
    missing_columns = []
    for relation in profile.relationships:
        if relation.child_column not in columns[relation.child_table]:
            missing_columns.append(f"{relation.child_table}.{relation.child_column}")
        if relation.parent_column not in columns[relation.parent_table]:
            missing_columns.append(f"{relation.parent_table}.{relation.parent_column}")
    if missing_columns:
        raise ValueError(f"Source is missing relationship column(s): {', '.join(sorted(set(missing_columns)))}")
    return columns


def fetch_stratum_counts(connection, profile: DriftProfile, boundaries: Sequence[Any]) -> dict[str, int]:
    expression = stratum_expression(profile, boundaries)
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT {expression} AS stratum, COUNT(*) FROM {profile.root_from_sql} "
            "GROUP BY stratum ORDER BY stratum"
        )
        return {stratum: count for stratum, count in cursor.fetchall()}


def select_seed_ids(
    connection, profile: DriftProfile, boundaries: Sequence[Any], quotas: Mapping[str, int], seed: int
) -> list[int]:
    expression = stratum_expression(profile, boundaries)
    selected = []
    with connection.cursor() as cursor:
        for stratum, quota in sorted(quotas.items()):
            if not quota:
                continue
            cursor.execute(
                f"SELECT {profile.root_id_sql} FROM {profile.root_from_sql} "
                f"WHERE {expression} = %s ORDER BY md5(({profile.root_id_sql})::text || %s), "
                f"{profile.root_id_sql} LIMIT %s",
                (stratum, str(seed), quota),
            )
            selected.extend(row[0] for row in cursor.fetchall())
    if len(selected) != sum(quotas.values()):
        raise RuntimeError("Source changed while root rows were being selected")
    return selected


def create_selection_tables(connection, profile: DriftProfile, selected: Sequence[int]) -> None:
    with connection.cursor() as cursor:
        cursor.execute("CREATE TEMP TABLE selected_root_ids (id bigint PRIMARY KEY) ON COMMIT PRESERVE ROWS")
        execute_values(cursor, "INSERT INTO selected_root_ids (id) VALUES %s", [(value,) for value in selected])
        if profile.name == "stats":
            cursor.execute("CREATE TEMP TABLE selected_related_user_ids (id bigint PRIMARY KEY) ON COMMIT PRESERVE ROWS")
            cursor.execute(
                "INSERT INTO selected_related_user_ids "
                "SELECT DISTINCT id FROM ("
                "SELECT owneruserid AS id FROM public.posts WHERE id IN (SELECT id FROM selected_root_ids) "
                "UNION SELECT lasteditoruserid FROM public.posts WHERE id IN (SELECT id FROM selected_root_ids) "
                "UNION SELECT userid FROM public.comments WHERE postid IN (SELECT id FROM selected_root_ids) "
                "UNION SELECT userid FROM public.posthistory WHERE postid IN (SELECT id FROM selected_root_ids) "
                "UNION SELECT userid FROM public.votes WHERE postid IN (SELECT id FROM selected_root_ids)"
                ") AS users WHERE id IS NOT NULL"
            )


def _pg_program(args: argparse.Namespace, name: str) -> str:
    return str(args.pg_bin_dir / name) if args.pg_bin_dir else name


def _pg_env(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    if args.pg_bin_dir is not None:
        pg_lib_dir = args.pg_bin_dir.parent / "lib"
        current_library_path = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = str(pg_lib_dir)
        if current_library_path:
            environment["LD_LIBRARY_PATH"] += os.pathsep + current_library_path
    if args.password is not None:
        environment["PGPASSWORD"] = args.password
    return environment


def _pg_connection_args(args: argparse.Namespace, dbname: str) -> list[str]:
    result = ["--host", args.host, "--port", str(args.port), "--dbname", dbname]
    if args.user:
        result.extend(["--username", args.user])
    return result


def create_target_database(args: argparse.Namespace) -> None:
    if args.source_db == args.target_db:
        raise ValueError("Source and target databases must differ")
    exists = database_exists(args, args.target_db)
    if exists and not args.replace_target:
        raise TargetDatabaseExistsError(
            "Target database already exists; choose another name or pass --replace-target"
        )
    admin = psycopg2.connect(**connection_kwargs(args, "postgres"))
    admin.autocommit = True
    try:
        with admin.cursor() as cursor:
            if exists:
                cursor.execute(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s AND pid<>pg_backend_pid()",
                    (args.target_db,),
                )
                cursor.execute(sql.SQL("DROP DATABASE {}").format(sql.Identifier(args.target_db)))
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(args.target_db)))
    finally:
        admin.close()


def dump_schema_sections(args: argparse.Namespace) -> tuple[Path, Path]:
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    pre_data = args.artifact_dir / "schema_pre_data.sql"
    post_data = args.artifact_dir / "schema_post_data.sql"
    common = [_pg_program(args, "pg_dump"), *_pg_connection_args(args, args.source_db), "--no-owner", "--no-privileges"]
    for section, destination in (("pre-data", pre_data), ("post-data", post_data)):
        with destination.open("w", encoding="utf-8") as handle:
            subprocess.run(
                [*common, "--section", section], check=True, stdout=handle, env=_pg_env(args)
            )
    return pre_data, post_data


def apply_sql_file(args: argparse.Namespace, path: Path) -> None:
    subprocess.run(
        [_pg_program(args, "psql"), *_pg_connection_args(args, args.target_db),
         "--set", "ON_ERROR_STOP=1", "--file", str(path)],
        check=True, env=_pg_env(args),
    )


def copy_table(
    source_connection, target_connection, profile: DriftProfile, table: str, columns: Sequence[str]
) -> int:
    select_query = build_table_select(profile, table, columns)
    copy_out = f"COPY ({select_query}) TO STDOUT WITH CSV HEADER"
    copy_in = sql.SQL("COPY public.{} FROM STDIN WITH CSV HEADER").format(
        sql.Identifier(table)
    ).as_string(target_connection)
    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b") as handle:
        with source_connection.cursor() as cursor:
            cursor.copy_expert(copy_out, handle)
        handle.seek(0)
        with target_connection.cursor() as cursor:
            cursor.copy_expert(copy_in, handle)
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM public.{}").format(sql.Identifier(table)))
            count = cursor.fetchone()[0]
        target_connection.commit()
    return count


def relation_orphan_counts(connection, profile: DriftProfile) -> dict[str, int]:
    result = {}
    with connection.cursor() as cursor:
        for relation in profile.relationships:
            cursor.execute(build_anti_join_sql(relation))
            key = f"{relation.child_table}.{relation.child_column}->{relation.parent_table}.{relation.parent_column}"
            result[key] = cursor.fetchone()[0]
    return result


def extract_explain_metrics(payload: Any) -> dict[str, float | int]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    root = payload[0]["Plan"]
    largest_join_rows = 0
    join_node_count = 0
    pending = [root]
    while pending:
        node = pending.pop()
        if node.get("Node Type") in {"Hash Join", "Merge Join", "Nested Loop"}:
            join_node_count += 1
            largest_join_rows = max(
                largest_join_rows,
                int(node.get("Actual Rows", 0)) * int(node.get("Actual Loops", 1)),
            )
        pending.extend(item for item in node.get("Plans", []) if isinstance(item, dict))
    return {
        "execution_ms": float(root.get("Actual Total Time", 0.0)),
        "largest_join_rows": largest_join_rows,
        "join_node_count": join_node_count,
    }


def benchmark_queries(args: argparse.Namespace, databases: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    results = {}
    for database in databases:
        database_results = []
        with psycopg2.connect(**connection_kwargs(args, database)) as connection:
            for path in sorted(args.query_dir.glob("*.sql")):
                query = path.read_text(encoding="utf-8").strip().rstrip(";")
                if not query:
                    continue
                try:
                    with connection.cursor() as cursor:
                        cursor.execute("SET LOCAL statement_timeout=%s", (str(args.statement_timeout_ms),))
                        cursor.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query)
                        plan = cursor.fetchone()[0]
                    database_results.append({"query": path.name, "metrics": extract_explain_metrics(plan), "plan": plan})
                except psycopg2.Error as error:
                    connection.rollback()
                    database_results.append({"query": path.name, "error": str(error).strip()})
        results[database] = database_results
    return results


def load_weights(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Target weights must be a JSON object")
    return {str(key): float(value) for key, value in payload.items()}


def write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def preflight(args: argparse.Namespace, profile: DriftProfile) -> dict[str, Any]:
    boundaries = parse_bins(profile, args.bins)
    weights = load_weights(args.target_weights)
    with psycopg2.connect(**connection_kwargs(args, args.source_db)) as connection:
        validate_source_schema(connection, profile)
        source_strata = fetch_stratum_counts(connection, profile, boundaries)
    available = sum(source_strata.values())
    root_count = args.root_count if args.root_count is not None else math.ceil(available * args.sample_scale)
    if args.scenario == "distribution" and weights is None:
        raise ValueError("--target-weights is required for the distribution scenario")
    quotas = allocate_stratum_quotas(source_strata, root_count, weights)
    return {
        "tool": "build_relational_drift", "phase": "preflight", "profile": profile.name,
        "source_db": args.source_db, "target_db": args.target_db, "scenario": args.scenario,
        "seed": args.seed, "sample_scale": args.sample_scale, "requested_root_count": root_count,
        "bins": [str(value) for value in boundaries], "target_weights": weights,
        "source_strata": source_strata, "requested_quotas": quotas,
        "target_already_exists": database_exists(args, args.target_db),
    }


def materialize(args: argparse.Namespace, profile: DriftProfile, manifest: dict[str, Any]) -> dict[str, Any]:
    boundaries = parse_bins(profile, ",".join(manifest["bins"]))
    with psycopg2.connect(**connection_kwargs(args, args.source_db)) as source:
        columns = validate_source_schema(source, profile)
        selected = select_seed_ids(source, profile, boundaries, manifest["requested_quotas"], args.seed)
        create_selection_tables(source, profile, selected)
        create_target_database(args)
        pre_data, post_data = dump_schema_sections(args)
        apply_sql_file(args, pre_data)
        with psycopg2.connect(**connection_kwargs(args, args.target_db)) as target:
            counts = {}
            for table in profile.tables:
                print(f"Copying {table}", flush=True)
                counts[table] = copy_table(source, target, profile, table, columns[table])
        apply_sql_file(args, post_data)
    manifest.update({"phase": "materialized", "selected_root_count": len(selected), "table_counts": counts})
    return manifest


def validate_target(args: argparse.Namespace, profile: DriftProfile, manifest: dict[str, Any]) -> dict[str, Any]:
    boundaries = parse_bins(profile, ",".join(manifest["bins"]))
    with psycopg2.connect(**connection_kwargs(args, args.target_db)) as target:
        orphans = relation_orphan_counts(target, profile)
        target_strata = (
            fetch_stratum_counts(target, profile, boundaries) if profile.root_table_is_filtered else None
        )
    validation: dict[str, Any] = {
        "phase": "validated", "relation_orphans": orphans,
        "all_relationships_valid": not any(orphans.values()), "target_strata": target_strata,
    }
    if target_strata is not None:
        validation["requested_target_jsd"] = jensen_shannon_divergence(
            manifest["requested_quotas"], target_strata
        )
        validation["source_target_jsd"] = jensen_shannon_divergence(
            manifest["source_strata"], target_strata
        )
    if args.query_dir is not None:
        databases = list(dict.fromkeys([args.target_db, *args.comparison_dbs]))
        results = benchmark_queries(args, databases)
        benchmark_path = args.artifact_dir / "query_results.json"
        benchmark_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validation["query_results_path"] = str(benchmark_path)
    manifest.update(validation)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), required=True)
    parser.add_argument("--phase", choices=("preflight", "materialize", "validate", "all"), default="all")
    parser.add_argument("--scenario", choices=("shrink", "distribution"), default="shrink")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--source-db")
    parser.add_argument("--target-db")
    parser.add_argument("--user", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--sample-scale", type=float, default=0.01)
    parser.add_argument("--root-count", type=int)
    parser.add_argument("--bins", help="Comma-separated integer or ISO-date boundaries")
    parser.add_argument("--target-weights", type=Path)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--query-dir", type=Path)
    parser.add_argument("--comparison-dbs", default="")
    parser.add_argument("--statement-timeout-ms", type=int, default=120000)
    parser.add_argument("--pg-bin-dir", type=Path)
    parser.add_argument("--replace-target", action="store_true")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    args.source_db = args.source_db or profile.default_source_db
    args.target_db = args.target_db or f"{args.source_db}_relational_drift"
    args.bins = args.bins or profile.default_bins
    args.artifact_dir = args.artifact_dir or Path("artifacts") / args.target_db
    args.manifest_path = args.artifact_dir / "manifest.json"
    args.comparison_dbs = tuple(name.strip() for name in args.comparison_dbs.split(",") if name.strip())
    if not 0 < args.sample_scale <= 1:
        parser.error("--sample-scale must be in (0, 1]")
    if args.root_count is not None and args.root_count <= 0:
        parser.error("--root-count must be positive")
    if args.statement_timeout_ms <= 0:
        parser.error("--statement-timeout-ms must be positive")
    if args.query_dir is not None and not args.query_dir.is_dir():
        parser.error(f"--query-dir does not exist: {args.query_dir}")
    return args


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.profile]
    if shutil.disk_usage(Path.cwd()).free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free space is required")
    if args.phase in {"preflight", "materialize", "all"}:
        manifest = preflight(args, profile)
        write_manifest(args.manifest_path, manifest)
    else:
        if not args.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {args.manifest_path}")
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("profile") != profile.name or manifest.get("target_db") != args.target_db:
            raise ValueError("Manifest does not match --profile/--target-db")
    if args.phase in {"materialize", "all"}:
        manifest = materialize(args, profile, manifest)
        write_manifest(args.manifest_path, manifest)
    if args.phase in {"validate", "all"}:
        if not database_exists(args, args.target_db):
            raise FileNotFoundError(f"Target database does not exist: {args.target_db}")
        manifest = validate_target(args, profile, manifest)
        write_manifest(args.manifest_path, manifest)
    print(f"Wrote manifest to {args.manifest_path}")


if __name__ == "__main__":
    main()
