from __future__ import annotations

from collections import Counter
import time

import psycopg2
from psycopg2 import sql

try:
    from .collect_histogram_common import (
        build_arg_parser, collect_database_histogram, discover_workload_columns,
        log, make_db_config, write_histogram,
    )
except ImportError:
    from collect_histogram_common import (
        build_arg_parser, collect_database_histogram, discover_workload_columns,
        log, make_db_config, write_histogram,
    )


FACT_DATE_KEYS = {
    "store_sales": "ss_sold_date_sk",
    "catalog_sales": "cs_sold_date_sk",
    "web_sales": "ws_sold_date_sk",
    "store_returns": "sr_returned_date_sk",
    "catalog_returns": "cr_returned_date_sk",
    "web_returns": "wr_returned_date_sk",
    "inventory": "inv_date_sk",
}
DRIFT_YEARS = (1998, 1999, 2000, 2001, 2002)
FACT_SAMPLE_ROWS = 20000


def sample_fact_date_histograms(database, args):
    conn = psycopg2.connect(**make_db_config(database, args))
    try:
        records = []
        with conn.cursor() as cursor:
            for table, date_key in FACT_DATE_KEYS.items():
                cursor.execute(
                    "SELECT c.reltuples FROM pg_class c JOIN pg_namespace n "
                    "ON n.oid = c.relnamespace WHERE n.nspname = %s AND c.relname = %s",
                    (args.schema, table),
                )
                row = cursor.fetchone()
                if row is None or row[0] <= 0:
                    raise RuntimeError("Missing ANALYZE statistics for {}.{}".format(database, table))
                rate = min(100.0, max(0.0001, 100.0 * FACT_SAMPLE_ROWS / float(row[0])))
                query = sql.SQL(
                    "SELECT f.{date_key}, d.d_year FROM {schema}.{table} AS f "
                    "TABLESAMPLE SYSTEM ({rate}) REPEATABLE (42) "
                    "JOIN {schema}.date_dim AS d ON d.d_date_sk = f.{date_key}"
                ).format(
                    date_key=sql.Identifier(date_key),
                    schema=sql.Identifier(args.schema),
                    table=sql.Identifier(table),
                    rate=sql.Literal(rate),
                )
                cursor.execute(query)
                samples = [(int(key), int(year)) for key, year in cursor.fetchall()]
                if not samples:
                    raise RuntimeError("Empty fact-date sample for {}.{}".format(database, table))
                keys = sorted(key for key, _ in samples)
                counts = Counter(year for _, year in samples)
                total = sum(counts.values())
                bins = [keys[round(i * (len(keys) - 1) / (args.bins - 1))] for i in range(args.bins)]
                records.append({
                    "table": table,
                    "column": date_key,
                    "table_column": "{}.{}".format(table, date_key),
                    "bins": bins,
                    "year_fractions": [counts[year] / total for year in DRIFT_YEARS],
                })
                log("[{}] sampled {}.{} rows={} rate={:.4f}%".format(
                    database, table, date_key, len(samples), rate,
                ))
        return records
    finally:
        conn.close()


DEFAULT_DATABASES = [
    "tpcds10g_199806",
    "tpcds10g_199812",
    "tpcds10g_199906",
    "tpcds10g_199912",
    "tpcds10g_200006",
    "tpcds10g_200012",
    "tpcds10g_200106",
    "tpcds10g_200112",
    "tpcds10g_200206",
    "tpcds10g",
]
DEFAULT_WORKLOAD_DIRS = [
    "./experiment/TPCDS/train",
    "./experiment/TPCDS/test",
]


if __name__ == "__main__":
    parser = build_arg_parser(DEFAULT_DATABASES, DEFAULT_WORKLOAD_DIRS)
    args = parser.parse_args()
    if args.bins < 2:
        parser.error("--bins must be at least 2")
    columns = discover_workload_columns(args.workload_dirs)
    log("discovered {} predicate columns from workloads".format(len(columns)))
    for database in args.databases:
        started = time.monotonic()
        records = collect_database_histogram(database, columns, args)
        records.extend(sample_fact_date_histograms(database, args))
        path = write_histogram(database, records, args.output_dir)
        log("written {} histogram records to {} elapsed_seconds={:.3f}".format(
            len(records), path, time.monotonic() - started,
        ))
