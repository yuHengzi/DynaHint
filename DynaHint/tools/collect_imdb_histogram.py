import argparse
import os

import numpy as np
import pandas as pd
import psycopg2


DEFAULT_DATABASE = "imdb"

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


def to_vals(data_list):
    val = None
    for datum in data_list:
        val = datum[0]
        if val is not None:
            break
    try:
        float(val)
        return np.array(data_list, dtype=float).squeeze()
    except Exception:
        values = []
        for datum in data_list:
            try:
                value = datum[0].timestamp()
            except Exception:
                value = 0
            values.append(value)
        return np.array(values)


def make_arg_parser():
    parser = argparse.ArgumentParser(description="Collect IMDb histogram JSON for DynaHint.")
    parser.add_argument("--database", default=DEFAULT_DATABASE)
    parser.add_argument("--host", default=os.getenv("DYNAHINT_PG_HOST", "localhost"))
    parser.add_argument("--port", default=os.getenv("DYNAHINT_PG_PORT", "5433"))
    parser.add_argument("--user", default=os.getenv("DYNAHINT_PG_USER", ""))
    parser.add_argument("--password", default=os.getenv("DYNAHINT_PG_PASSWORD") or None)
    parser.add_argument("--output-dir", default="./experiment/histogram")
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


def collect_hist(args):
    conn = psycopg2.connect(**make_db_config(args))
    conn.set_session(autocommit=True)
    cursor = conn.cursor()
    records = []
    try:
        for table, columns in IMDB_SCHEMA.items():
            for column in columns:
                command = "select {} from {} as {}".format(column, table, TABLE_ALIASES[table])
                cursor.execute(command)
                values = to_vals(cursor.fetchall())
                bins = np.nanpercentile(values, range(0, 101, 2), axis=0)
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
    output_path = os.path.join(args.output_dir, f"{args.database}_histogram_string.json")
    histogram.to_json(output_path, orient="records", force_ascii=False, indent=2)
    print(f"wrote histogram: {output_path}")


if __name__ == "__main__":
    collect_hist(make_arg_parser().parse_args())
