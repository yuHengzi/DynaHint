import argparse
import json
import os
import time

import pandas as pd
import psycopg2
from psycopg2 import sql


DEFAULT_DATABASE = "imdb2010cut"

IMDB_SCHEMA = {
    "title": ["t.id", "t.kind_id", "t.production_year"],
    "movie_companies": ["mc.id", "mc.company_id", "mc.movie_id", "mc.company_type_id"],
    "cast_info": ["ci.id", "ci.movie_id", "ci.person_id", "ci.role_id"],
    "movie_info_idx": ["mi_idx.id", "mi_idx.movie_id", "mi_idx.info_type_id"],
    "movie_info": ["mi.id", "mi.movie_id", "mi.info_type_id"],
    "movie_keyword": ["mk.id", "mk.movie_id", "mk.keyword_id"],
}

TABLE_ALIASES = {
    "title": "t",
    "movie_companies": "mc",
    "cast_info": "ci",
    "movie_info_idx": "mi_idx",
    "movie_info": "mi",
    "movie_keyword": "mk",
}


def make_arg_parser():
    parser = argparse.ArgumentParser(description="Collect IMDb histogram JSON for DynaHint.")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--databases", nargs="+")
    parser.add_argument("--host", default=os.getenv("DYNAHINT_PG_HOST", "localhost"))
    parser.add_argument("--port", default=os.getenv("DYNAHINT_PG_PORT", "5432"))
    parser.add_argument("--user", default=os.getenv("DYNAHINT_PG_USER", ""))
    parser.add_argument("--password", default=os.getenv("DYNAHINT_PG_PASSWORD") or None)
    parser.add_argument("--output-dir", default="./experiment/histogram")
    parser.add_argument("--bins", type=int, default=51)
    parser.add_argument("--timing-output")
    return parser


def make_db_config(args):
    config = {
        "database": args.database,
        "user": args.user,
        "host": args.host,
        "port": args.port,
    }
    if args.password:
        config["password"] = args.password
    return config


def collect_hist(database, args):
    args.database = database
    started = time.perf_counter()
    conn = psycopg2.connect(**make_db_config(args))
    conn.set_session(autocommit=True)
    cursor = conn.cursor()
    records = []
    try:
        for table, columns in IMDB_SCHEMA.items():
            for column in columns:
                column_name = column.split(".", 1)[1]
                fractions = [index / float(args.bins - 1) for index in range(args.bins)]
                command = sql.SQL(
                    "SELECT percentile_cont(%s) WITHIN GROUP (ORDER BY {}::double precision) "
                    "FROM {} WHERE {} IS NOT NULL"
                ).format(
                    sql.Identifier(column_name),
                    sql.Identifier(table),
                    sql.Identifier(column_name),
                )
                cursor.execute(command, (fractions,))
                bins = cursor.fetchone()[0]
                records.append(
                    {
                        "table": table,
                        "column": column,
                        "table_column": ".".join((table, column)),
                        "bins": bins,
                    }
                )
    finally:
        cursor.close()
        conn.close()

    histogram = pd.DataFrame(records, columns=["table", "column", "bins", "table_column"])
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{database}_histogram_string.json")
    histogram.to_json(output_path, orient="records", force_ascii=False, indent=2)
    elapsed = time.perf_counter() - started
    print(f"wrote histogram: {output_path} elapsed_seconds={elapsed:.3f}", flush=True)
    return {
        "database": database,
        "elapsed_seconds": elapsed,
        "output_path": output_path,
        "columns": len(records),
    }


if __name__ == "__main__":
    parsed_args = make_arg_parser().parse_args()
    if parsed_args.bins < 2:
        raise ValueError("--bins must be at least 2")
    databases = parsed_args.databases or [parsed_args.database]
    total_started = time.perf_counter()
    timings = [collect_hist(database, parsed_args) for database in databases]
    summary = {
        "databases": timings,
        "total_elapsed_seconds": time.perf_counter() - total_started,
    }
    if parsed_args.timing_output:
        timing_path = os.path.abspath(parsed_args.timing_output)
        os.makedirs(os.path.dirname(timing_path), exist_ok=True)
        with open(timing_path, "w", encoding="utf-8") as file_obj:
            json.dump(summary, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
        print(f"wrote timing summary: {timing_path}", flush=True)
