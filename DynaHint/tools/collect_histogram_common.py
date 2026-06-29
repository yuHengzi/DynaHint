from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import psycopg2
from psycopg2 import sql


NUMERIC_TYPES = {
    "smallint",
    "integer",
    "bigint",
    "decimal",
    "numeric",
    "real",
    "double precision",
}
TEMPORAL_TYPES = {
    "date",
    "timestamp without time zone",
    "timestamp with time zone",
}
SUPPORTED_TYPES = NUMERIC_TYPES | TEMPORAL_TYPES
JOIN_KEYWORDS = {
    "on",
    "where",
    "join",
    "inner",
    "left",
    "right",
    "full",
    "cross",
    "group",
    "order",
    "limit",
    "using",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}")


def normalize_identifier(identifier: str) -> str:
    identifier = identifier.strip().strip('"').strip()
    if "." in identifier:
        identifier = identifier.split(".")[-1]
    return identifier.lower()


def strip_sql_comments(sql_text: str) -> str:
    sql_text = re.sub(r"/\*.*?\*/", " ", sql_text, flags=re.S)
    sql_text = re.sub(r"--.*?$", " ", sql_text, flags=re.M)
    return sql_text


def extract_alias_map(sql_text: str) -> Dict[str, str]:
    text = strip_sql_comments(sql_text).lower()
    pattern = re.compile(
        r"(?:\bfrom|\bjoin|,)\s+"
        r"(?P<table>\"?[a-z_][\w$]*\"?(?:\.\"?[a-z_][\w$]*\"?)?)"
        r"(?:\s+(?:as\s+)?(?P<alias>\"?[a-z_][\w$]*\"?))?",
        flags=re.I,
    )
    aliases: Dict[str, str] = {}
    for match in pattern.finditer(text):
        table = normalize_identifier(match.group("table"))
        alias = match.group("alias")
        aliases[table] = table
        if alias:
            alias = normalize_identifier(alias)
            if alias not in JOIN_KEYWORDS:
                aliases[alias] = table
    return aliases


def _right_is_column_reference(raw_right: str) -> bool:
    right = raw_right.strip().strip("()").lower()
    if re.match(r"^[a-z_][\w$]*\.[a-z_][\w$]*$", right):
        return True
    if re.match(r"^[a-z_][\w$]*$", right) and right not in {"true", "false", "null"}:
        return True
    return False


def extract_predicate_columns(sql_text: str) -> List[Tuple[str, str]]:
    text = strip_sql_comments(sql_text).lower()
    aliases = extract_alias_map(text)
    pattern = re.compile(
        r"(?P<left>(?:\"?[a-z_][\w$]*\"?\.)?\"?[a-z_][\w$]*\"?)\s*"
        r"(?P<op>>=|<=|<>|!=|=|>|<|~~\*|~~|!~~|like|ilike)\s*"
        r"(?P<right>date\s+'[^']*'|timestamp\s+'[^']*'|'[^']*'|[+-]?\d+(?:\.\d+)?|\"?[a-z_][\w$]*\"?\.[a-z_][\w$]*|[a-z_][\w$]*)",
        flags=re.I,
    )
    columns: Set[Tuple[str, str]] = set()
    for match in pattern.finditer(text):
        right = match.group("right")
        if _right_is_column_reference(right):
            continue
        left = match.group("left").replace('"', "")
        if "." not in left:
            continue
        alias, column = left.split(".", 1)
        table = aliases.get(alias, alias)
        if table:
            columns.add((normalize_identifier(table), normalize_identifier(column)))
    return sorted(columns)


def discover_workload_columns(workload_dirs: Sequence[str]) -> List[Tuple[str, str]]:
    columns: Set[Tuple[str, str]] = set()
    for workload_dir in workload_dirs:
        if not os.path.isdir(workload_dir):
            log(f"[Warn] workload dir not found: {workload_dir}")
            continue
        for filename in sorted(os.listdir(workload_dir)):
            if not filename.endswith(".sql"):
                continue
            path = os.path.join(workload_dir, filename)
            with open(path, "r", encoding="utf-8") as file:
                columns.update(extract_predicate_columns(file.read()))
    return sorted(columns)


def make_db_config(database: str, args: argparse.Namespace) -> Dict[str, str]:
    config = {
        "dbname": database,
        "host": args.host,
        "port": str(args.port),
        "user": args.user,
    }
    if args.password:
        config["password"] = args.password
    return config


def load_column_types(conn, schema: str) -> Dict[Tuple[str, str], str]:
    query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s
    """
    with conn.cursor() as cursor:
        cursor.execute(query, (schema,))
        return {
            (str(table).lower(), str(column).lower()): str(data_type).lower()
            for table, column, data_type in cursor.fetchall()
        }


def histogram_expression(column_name: str, data_type: str):
    column = sql.Identifier(column_name)
    if data_type in TEMPORAL_TYPES:
        return sql.SQL("EXTRACT(EPOCH FROM {}::timestamp)").format(column)
    return sql.SQL("{}::double precision").format(column)


def fetch_histogram_bins(
    conn,
    schema: str,
    table_name: str,
    column_name: str,
    data_type: str,
    bins: int,
) -> Optional[List[float]]:
    if bins < 2:
        raise ValueError("bins must be >= 2")
    fractions = [float(i) / float(bins - 1) for i in range(bins)]
    expr = histogram_expression(column_name, data_type)
    query = sql.SQL(
        "SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY {expr}) "
        "FROM {schema}.{table} WHERE {column} IS NOT NULL"
    ).format(
        expr=expr,
        schema=sql.Identifier(schema),
        table=sql.Identifier(table_name),
        column=sql.Identifier(column_name),
    )
    with conn.cursor() as cursor:
        cursor.execute(query, (fractions,))
        row = cursor.fetchone()
    if not row or row[0] is None:
        return None
    return [float(value) for value in row[0]]


def collect_database_histogram(
    database: str,
    columns: Sequence[Tuple[str, str]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    conn = psycopg2.connect(**make_db_config(database, args))
    try:
        column_types = load_column_types(conn, args.schema)
        records: List[Dict[str, object]] = []
        selected_columns = list(columns)
        if args.limit_columns and args.limit_columns > 0:
            selected_columns = selected_columns[: args.limit_columns]
        for index, (table_name, column_name) in enumerate(selected_columns, 1):
            data_type = column_types.get((table_name, column_name))
            if data_type not in SUPPORTED_TYPES:
                log(
                    f"[{database}] skip unsupported/missing column "
                    f"{index}/{len(selected_columns)} {table_name}.{column_name} type={data_type}"
                )
                continue
            try:
                bins = fetch_histogram_bins(
                    conn,
                    args.schema,
                    table_name,
                    column_name,
                    data_type,
                    args.bins,
                )
            except Exception as exc:
                log(f"[{database}] failed {table_name}.{column_name}: {exc}")
                continue
            if bins is None:
                log(f"[{database}] skip empty column {table_name}.{column_name}")
                continue
            records.append(
                {
                    "table": table_name,
                    "column": column_name,
                    "table_column": f"{table_name}.{column_name}",
                    "bins": bins,
                }
            )
            log(f"[{database}] collected {index}/{len(selected_columns)} {table_name}.{column_name}")
        return records
    finally:
        conn.close()


def write_histogram(database: str, records: Sequence[Dict[str, object]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{database}_histogram_string.json")
    with open(path, "w", encoding="utf-8") as file:
        json.dump(list(records), file, ensure_ascii=False, indent=2)
    return path


def build_arg_parser(default_databases: Sequence[str], default_workload_dirs: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect DynaHint histogram JSON files.")
    parser.add_argument("--databases", nargs="+", default=list(default_databases))
    parser.add_argument("--workload-dirs", nargs="+", default=list(default_workload_dirs))
    parser.add_argument("--output-dir", default="./experiment/histogram")
    parser.add_argument("--host", default=os.getenv("DYNAHINT_PG_HOST", "localhost"))
    parser.add_argument("--port", default=os.getenv("DYNAHINT_PG_PORT", "5433"))
    parser.add_argument("--user", default=os.getenv("DYNAHINT_PG_USER", ""))
    parser.add_argument("--password", default=os.getenv("DYNAHINT_PG_PASSWORD") or None)
    parser.add_argument("--schema", default="public")
    parser.add_argument("--bins", type=int, default=51)
    parser.add_argument("--limit-columns", type=int, default=0)
    return parser


def run_collection(default_databases: Sequence[str], default_workload_dirs: Sequence[str]) -> None:
    parser = build_arg_parser(default_databases, default_workload_dirs)
    args = parser.parse_args()
    columns = discover_workload_columns(args.workload_dirs)
    log(f"discovered {len(columns)} predicate columns from workloads")
    for database in args.databases:
        log(f"collecting histogram for database={database}")
        records = collect_database_histogram(database, columns, args)
        path = write_histogram(database, records, args.output_dir)
        log(f"written {len(records)} histogram records to {path}")
