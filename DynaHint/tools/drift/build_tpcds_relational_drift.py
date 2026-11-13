#!/usr/bin/env python3
"""Build TPC-DS training cuts with real source facts and shifted date keys."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path
from typing import Mapping

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

import build_relational_drift as base


PROFILE_VERSION = "tpcds"
YEARS = tuple(range(1998, 2003))
PROFILE = base.DriftProfile(
    name="tpcds",
    default_source_db="tpcds_200012",
    tables=base.TPCDS_TABLES,
    relationships=base.TPCDS_RELATIONSHIPS,
    root_from_sql="tpcds_root_features AS root",
    root_id_sql="root.d_date_sk",
    root_value_sql="root.d_date",
    root_category_sql="root.category",
    default_bins="1998-12-31,1999-12-31",
    bin_type="date",
    root_table_is_filtered=False,
)
FACTOR_DIMENSIONS = ("year", "density", "inventory", "channel", "returns")
WINDOW_RE = re.compile(
    r"date_dim\.d_date\s+between\s+cast\('([0-9-]+)'\s+as\s+date\)"
    r"\s+and\s+\(cast\('[0-9-]+'\s+as\s+date\)\s*\+\s*'([0-9]+) days'",
    re.IGNORECASE,
)

BASE_FEATURE_TABLE_SQL = """
CREATE TEMP TABLE tpcds_root_features_base ON COMMIT PRESERVE ROWS AS
WITH sales_by_date AS (
    SELECT ss_sold_date_sk AS date_sk, COUNT(*) AS sales_count
    FROM public.store_sales
    GROUP BY ss_sold_date_sk
), dated AS (
    SELECT d.d_date_sk, d.d_date,
           EXTRACT(QUARTER FROM d.d_date)::int AS quarter,
           COALESCE(s.sales_count, 0) AS sales_count,
           EXISTS (
               SELECT 1 FROM public.inventory AS i
               WHERE i.inv_date_sk = d.d_date_sk
           ) AS has_inventory
    FROM public.date_dim AS d
    LEFT JOIN sales_by_date AS s ON s.date_sk = d.d_date_sk
    WHERE d.d_date BETWEEN DATE '1998-01-01' AND DATE '2000-12-31'
), ranked AS (
    SELECT *, NTILE(3) OVER (ORDER BY sales_count, d_date_sk) AS density_band
    FROM dated
)
SELECT d_date_sk, d_date, sales_count, has_inventory,
       'q' || quarter || '|' ||
       CASE density_band WHEN 1 THEN 'low' WHEN 2 THEN 'mid' ELSE 'high' END || '|' ||
       CASE WHEN has_inventory THEN 'inventory' ELSE 'other' END AS category
FROM ranked
"""


FEATURE_TABLE_SQL = """
CREATE TEMP TABLE tpcds_root_features ON COMMIT PRESERVE ROWS AS
WITH catalog AS (
    SELECT cs_sold_date_sk AS date_sk, COUNT(*) AS count FROM public.catalog_sales GROUP BY 1
), web AS (
    SELECT ws_sold_date_sk AS date_sk, COUNT(*) AS count FROM public.web_sales GROUP BY 1
), store_return AS (
    SELECT sr_returned_date_sk AS date_sk, COUNT(*) AS count FROM public.store_returns GROUP BY 1
), catalog_return AS (
    SELECT cr_returned_date_sk AS date_sk, COUNT(*) AS count FROM public.catalog_returns GROUP BY 1
), web_return AS (
    SELECT wr_returned_date_sk AS date_sk, COUNT(*) AS count FROM public.web_returns GROUP BY 1
), counts AS (
    SELECT root.d_date_sk, root.d_date, root.has_inventory, root.sales_count,
           split_part(root.category, '|', 2) AS density,
           COALESCE(catalog.count, 0) AS catalog_count,
           COALESCE(web.count, 0) AS web_count,
           COALESCE(store_return.count, 0) + COALESCE(catalog_return.count, 0)
             + COALESCE(web_return.count, 0) AS return_count
    FROM tpcds_root_features_base AS root
    LEFT JOIN catalog ON catalog.date_sk=root.d_date_sk
    LEFT JOIN web ON web.date_sk=root.d_date_sk
    LEFT JOIN store_return ON store_return.date_sk=root.d_date_sk
    LEFT JOIN catalog_return ON catalog_return.date_sk=root.d_date_sk
    LEFT JOIN web_return ON web_return.date_sk=root.d_date_sk
), ranked AS (
    SELECT *,
           NTILE(3) OVER (ORDER BY (catalog_count + 1.0) / (sales_count + 1.0), d_date_sk) AS catalog_band,
           NTILE(3) OVER (ORDER BY (web_count + 1.0) / (sales_count + 1.0), d_date_sk) AS web_band,
           NTILE(3) OVER (ORDER BY (return_count + 1.0) /
               (sales_count + catalog_count + web_count + 1.0), d_date_sk) AS return_band
    FROM counts
)
SELECT d_date_sk, d_date, has_inventory, sales_count,
       density || '|' ||
       CASE WHEN has_inventory THEN 'inventory' ELSE 'other' END || '|' ||
       CASE WHEN catalog_band=3 AND web_band<3 THEN 'catalog'
            WHEN web_band=3 THEN 'web' ELSE 'store' END || '|' ||
       CASE return_band WHEN 1 THEN 'low' WHEN 2 THEN 'mid' ELSE 'high' END AS category
FROM ranked
"""


def ensure_base_root_features(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('pg_temp.tpcds_root_features_base')")
        if cursor.fetchone()[0] is not None:
            return
        cursor.execute(BASE_FEATURE_TABLE_SQL)
        cursor.execute("CREATE UNIQUE INDEX ON tpcds_root_features_base (d_date_sk)")
        cursor.execute("ANALYZE tpcds_root_features_base")


def select_dates(connection, profile: base.DriftProfile, boundaries,
                 root_count: int, weights: Mapping[str, float], seed: int):
    expression = base.stratum_expression(profile, boundaries)
    with connection.cursor() as cursor:
        cursor.execute(
            f"SELECT root.d_date_sk, to_char(root.d_date, 'YYYY-MM'), "
            f"root.has_inventory, {expression} FROM {profile.root_from_sql}"
        )
        rows = cursor.fetchall()
    by_stratum: dict[str, list] = {}
    for row in rows:
        by_stratum.setdefault(row[3], []).append(row)

    def rank(row):
        return hashlib.md5((str(row[0]) + str(seed)).encode("ascii")).hexdigest(), row[0]

    inventory_by_month: dict[str, list] = {}
    for row in rows:
        if row[2]:
            inventory_by_month.setdefault(row[1], []).append(row)
    if len(inventory_by_month) != 36:
        raise ValueError("Source must contain inventory dates in all 36 months")

    reserved = {min(month_rows, key=rank)[0] for month_rows in inventory_by_month.values()}
    for stratum, stratum_rows in sorted(by_stratum.items()):
        if not any(row[0] in reserved for row in stratum_rows):
            reserved.add(min(stratum_rows, key=rank)[0])
    if len(reserved) > root_count:
        raise ValueError("Root count is too small for monthly inventory and stratum coverage")

    remaining = {
        stratum: sum(row[0] not in reserved for row in stratum_rows)
        for stratum, stratum_rows in by_stratum.items()
    }
    extras = base.allocate_stratum_quotas(remaining, root_count - len(reserved), weights)
    selected = set(reserved)
    for stratum, count in extras.items():
        candidates = (row for row in by_stratum[stratum] if row[0] not in reserved)
        selected.update(row[0] for row in sorted(candidates, key=rank)[:count])
    if len(selected) != root_count:
        raise RuntimeError("Date selection did not reach the requested count")
    quotas = {
        stratum: sum(row[0] in selected for row in stratum_rows)
        for stratum, stratum_rows in by_stratum.items()
    }
    return sorted(selected), quotas


def ensure_root_features(connection) -> None:
    ensure_base_root_features(connection)
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('pg_temp.tpcds_root_features')")
        if cursor.fetchone()[0] is not None:
            return
        cursor.execute(FEATURE_TABLE_SQL)
        cursor.execute("CREATE UNIQUE INDEX ON tpcds_root_features (d_date_sk)")
        cursor.execute("ANALYZE tpcds_root_features")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    mix = float(config.get("source_mix", 0.35))
    if not math.isfinite(mix) or not 0 < mix <= 1:
        raise ValueError("source_mix must be finite and in (0, 1]")
    factors = config.get("factors")
    if not isinstance(factors, dict) or set(factors) != set(FACTOR_DIMENSIONS):
        raise ValueError(f"factors must contain {FACTOR_DIMENSIONS}")
    normalized = {}
    for dimension in FACTOR_DIMENSIONS:
        values = factors[dimension]
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{dimension} weights must be a non-empty object")
        normalized[dimension] = {str(key): float(value) for key, value in values.items()}
        if any(not math.isfinite(value) or value < 0 for value in normalized[dimension].values()):
            raise ValueError(f"{dimension} weights must be finite and non-negative")
    year_weights = config.get("target_year_weights")
    if not isinstance(year_weights, dict) or set(year_weights) != {str(y) for y in YEARS}:
        raise ValueError("target_year_weights must specify 1998 through 2002")
    weights = {int(year): float(weight) for year, weight in year_weights.items()}
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("Every target year weight must be finite and positive")
    return {"factor_config": {"source_mix": mix, "factors": normalized}, "target_year_weights": weights}


def expand_factor_weights(strata: dict[str, int], config: dict) -> dict[str, float]:
    raw = {}
    for stratum, count in strata.items():
        labels = stratum.split("|")
        if len(labels) != len(FACTOR_DIMENSIONS):
            raise ValueError(f"Unexpected stratum: {stratum}")
        weight = 1.0
        for dimension, label in zip(FACTOR_DIMENSIONS, labels):
            factor = config["factors"][dimension]
            weight *= factor.get(label, factor.get("*", 0.0))
        raw[stratum] = count * weight
    raw_total, source_total = sum(raw.values()), sum(strata.values())
    if raw_total <= 0 or source_total <= 0:
        raise ValueError("No positive source strata weights")
    mix = config["source_mix"]
    return {
        key: mix * count / source_total + (1 - mix) * raw[key] / raw_total
        for key, count in strata.items()
    }


def year_quotas(count: int, weights: dict[int, float]) -> dict[int, int]:
    total = sum(weights.values())
    exact = {year: count * weight / total for year, weight in weights.items()}
    quotas = {year: int(value) for year, value in exact.items()}
    for year in sorted(weights, key=lambda y: (exact[y] - quotas[y], -y), reverse=True)[:count - sum(quotas.values())]:
        quotas[year] += 1
    return quotas


def inventory_windows(query_dir: Path | None) -> list[tuple[str, dt.date, dt.date]]:
    if query_dir is None:
        return []
    windows = []
    for path in sorted(query_dir.glob("*.sql")):
        query = path.read_text(encoding="utf-8")
        if not re.search(r"\binventory\b", query, re.IGNORECASE):
            continue
        match = WINDOW_RE.search(query)
        if match:
            start = dt.date.fromisoformat(match.group(1))
            windows.append((path.stem, start, start + dt.timedelta(days=int(match.group(2)))))
    return windows


def build_date_map(rows, destinations: dict[dt.date, int], weights: dict[int, float],
                   windows: list[tuple[str, dt.date, dt.date]]):
    """Stretch source dates monotonically across the requested target years."""
    quotas = year_quotas(len(rows), weights)
    ordered = sorted(rows, key=lambda row: (row[1], row[0]))
    assigned = {}
    mapped_inventory = []
    cursor = 0
    for year in YEARS:
        count = quotas[year]
        if count <= 0:
            raise ValueError("Every target year needs at least one selected date")
        first = dt.date(year, 1, 1)
        days = (dt.date(year + 1, 1, 1) - first).days
        if count > days:
            raise ValueError(f"Too many dates requested for target year {year}")
        for index, (source_id, source_date, has_inventory) in enumerate(ordered[cursor:cursor + count]):
            target_date = first + dt.timedelta(days=(2 * index + 1) * days // (2 * count))
            if target_date not in destinations:
                raise ValueError(f"Missing destination date {target_date}")
            assigned[source_id] = (destinations[target_date], (target_date - source_date).days)
            if has_inventory:
                mapped_inventory.append(target_date)
        cursor += count
    missing = [
        query_id for query_id, start, end in windows
        if not any(start <= date <= end for date in mapped_inventory)
    ]
    if missing:
        raise ValueError(f"Selected inventory dates miss workload windows: {', '.join(missing)}")
    if len(assigned) != len(rows) or len({value[0] for value in assigned.values()}) != len(rows):
        raise RuntimeError("Date map is incomplete or non-unique")
    return assigned, quotas


def create_date_map_table(connection, date_map: dict[int, tuple[int, int]]) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "CREATE TEMP TABLE selected_date_map (old_id integer PRIMARY KEY, "
            "new_id integer UNIQUE NOT NULL, offset_days integer NOT NULL) ON COMMIT PRESERVE ROWS"
        )
        execute_values(
            cursor,
            "INSERT INTO selected_date_map (old_id,new_id,offset_days) VALUES %s",
            [(old, new, offset) for old, (new, offset) in sorted(date_map.items())],
        )
        cursor.execute("ANALYZE selected_date_map")


def fact_select(table: str, columns: tuple[str, ...]) -> str:
    primary = base.TPCDS_FACT_DATE_KEYS[table]
    date_columns = [column for column in columns if column.endswith("_date_sk") and column != primary]
    joins = []
    projections = []
    for index, column in enumerate(date_columns):
        old_alias, new_alias = f"old_date_{index}", f"new_date_{index}"
        joins.append(
            f"LEFT JOIN public.date_dim AS {old_alias} ON {old_alias}.d_date_sk = fact.{column} "
            f"LEFT JOIN public.date_dim AS {new_alias} ON {new_alias}.d_date = "
            f"({old_alias}.d_date + dates.offset_days)"
        )
    for column in columns:
        if column == primary:
            expression = "dates.new_id"
        elif column in date_columns:
            index = date_columns.index(column)
            expression = f"COALESCE(new_date_{index}.d_date_sk, fact.{column})"
        else:
            expression = f"fact.{column}"
        projections.append(f"{expression} AS {column}")
    return (
        f"SELECT {', '.join(projections)} FROM public.{table} AS fact "
        f"JOIN selected_date_map AS dates ON dates.old_id = fact.{primary} "
        + " ".join(joins)
    )


def copy_fact(source, target, table: str, columns: tuple[str, ...]) -> int:
    query = fact_select(table, columns)
    copy_in = sql.SQL("COPY public.{} FROM STDIN WITH CSV HEADER").format(sql.Identifier(table)).as_string(target)
    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b") as handle:
        with source.cursor() as cursor:
            cursor.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", handle)
        handle.seek(0)
        with target.cursor() as cursor:
            cursor.copy_expert(copy_in, handle)
            cursor.execute(sql.SQL("SELECT COUNT(*) FROM public.{}").format(sql.Identifier(table)))
            count = cursor.fetchone()[0]
        target.commit()
    return count


def preflight(args) -> dict:
    profile = PROFILE
    boundaries = base.parse_bins(profile, args.bins)
    config = load_config(args.target_weights)
    with psycopg2.connect(**base.connection_kwargs(args, args.source_db)) as source:
        base.validate_source_schema(source, profile)
        ensure_root_features(source)
        strata = base.fetch_stratum_counts(source, profile, boundaries)
        count = args.root_count if args.root_count is not None else math.ceil(sum(strata.values()) * args.sample_scale)
        weights = expand_factor_weights(strata, config["factor_config"])
        with source.cursor() as cursor:
            cursor.execute(
                "SELECT d_date,d_date_sk FROM public.date_dim WHERE d_date BETWEEN DATE '1998-01-01' AND DATE '2002-12-31'"
            )
            destinations = dict(cursor.fetchall())
        windows = inventory_windows(args.query_dir)
        for attempt in range(50):
            selection_seed = args.seed + attempt * 1009
            selected, quotas = select_dates(source, profile, boundaries, count, weights, selection_seed)
            with source.cursor() as cursor:
                cursor.execute(
                    "SELECT d_date_sk,d_date,has_inventory FROM tpcds_root_features WHERE d_date_sk=ANY(%s)",
                    (selected,),
                )
                rows = cursor.fetchall()
            try:
                date_map, target_quotas = build_date_map(
                    rows, destinations, config["target_year_weights"], windows,
                )
                break
            except ValueError as error:
                if not str(error).startswith("Selected inventory dates miss workload windows"):
                    raise
        else:
            raise ValueError("Could not cover inventory query windows after 50 date selections")
        create_date_map_table(source, date_map)
        with source.cursor() as cursor:
            fact_date_counts = {}
            for table, column in base.TPCDS_FACT_DATE_KEYS.items():
                cursor.execute(
                    f"SELECT EXTRACT(YEAR FROM d.d_date)::int, COUNT(*) FROM public.{table} AS fact "
                    f"JOIN selected_date_map AS m ON m.old_id=fact.{column} "
                    "JOIN public.date_dim AS d ON d.d_date_sk=m.new_id GROUP BY 1 ORDER BY 1"
                )
                fact_date_counts[table] = dict(cursor.fetchall())
    return {
        "tool": "build_tpcds_relational_drift", "phase": "preflight",
        "profile_version": PROFILE_VERSION, "source_db": args.source_db, "target_db": args.target_db,
        "seed": args.seed, "selection_seed": selection_seed, "requested_root_count": count,
        "factor_config": config["factor_config"], "target_year_weights": config["target_year_weights"],
        "source_strata": strata, "requested_quotas": quotas,
        "selected_date_map": [[old, new, offset] for old, (new, offset) in sorted(date_map.items())],
        "target_year_quotas": target_quotas, "covered_inventory_windows": [name for name, _, _ in windows],
        "fact_date_counts": fact_date_counts,
        "date_selection_jsd": base.jensen_shannon_divergence(strata, quotas),
        "target_already_exists": base.database_exists(args, args.target_db),
    }


def materialize(args, manifest: dict) -> dict:
    profile = PROFILE
    with psycopg2.connect(**base.connection_kwargs(args, args.source_db)) as source:
        columns = base.validate_source_schema(source, profile)
        date_map = {old: (new, offset) for old, new, offset in manifest["selected_date_map"]}
        create_date_map_table(source, date_map)
        base.create_target_database(args)
        pre_data, post_data = base.dump_schema_sections(args)
        base.apply_sql_file(args, pre_data)
        with psycopg2.connect(**base.connection_kwargs(args, args.target_db)) as target:
            counts = {}
            for table in profile.tables:
                print(f"Copying {table}", flush=True)
                if table in base.TPCDS_FACT_DATE_KEYS:
                    counts[table] = copy_fact(source, target, table, columns[table])
                else:
                    counts[table] = base.copy_table(source, target, profile, table, columns[table])
        base.apply_sql_file(args, post_data)
        with psycopg2.connect(**base.connection_kwargs(args, args.target_db)) as target:
            with target.cursor() as cursor:
                cursor.execute("ANALYZE")
    manifest.update({"phase": "materialized", "table_counts": counts})
    return manifest


def validate_target(args, manifest: dict) -> dict:
    with psycopg2.connect(**base.connection_kwargs(args, args.target_db)) as target:
        orphans = base.relation_orphan_counts(target, PROFILE)
        with target.cursor() as cursor:
            actual_year_counts = {}
            for table, column in base.TPCDS_FACT_DATE_KEYS.items():
                cursor.execute(
                    f"SELECT EXTRACT(YEAR FROM d.d_date)::int, COUNT(*) FROM public.{table} AS fact "
                    f"JOIN public.date_dim AS d ON d.d_date_sk=fact.{column} GROUP BY 1 ORDER BY 1"
                )
                actual_year_counts[table] = dict(cursor.fetchall())
    expected_counts = {
        table: {int(year): count for year, count in counts.items()}
        for table, counts in manifest["fact_date_counts"].items()
    }
    if actual_year_counts != expected_counts or any(orphans.values()):
        raise ValueError("target fact/year counts or date relationships differ from preflight")
    manifest.update({
        "phase": "validated", "relation_orphans": orphans,
        "all_relationships_valid": True,
        "actual_fact_date_counts": actual_year_counts,
    })
    return manifest


def main() -> None:
    base.PROFILES["tpcds"] = PROFILE
    args = base.parse_args()
    if args.profile != "tpcds" or args.scenario != "distribution":
        raise ValueError("Use --profile tpcds --scenario distribution")
    if args.replace_target:
        raise ValueError("Drop old target databases explicitly before materializing ")
    if args.phase in {"materialize", "all"}:
        with psycopg2.connect(**base.connection_kwargs(args, args.source_db)) as source:
            with source.cursor() as cursor:
                cursor.execute("SHOW data_directory")
                data_directory = cursor.fetchone()[0]
        if shutil.disk_usage(data_directory).free < 12 * 1024**3:
            raise RuntimeError("At least 12 GiB free space is required on the PostgreSQL filesystem")
    if args.phase in {"preflight", "all"}:
        manifest = preflight(args)
        base.write_manifest(args.manifest_path, manifest)
    else:
        if not args.manifest_path.exists():
            raise FileNotFoundError(f"Run preflight first: {args.manifest_path}")
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("profile_version") != PROFILE_VERSION or manifest.get("target_db") != args.target_db:
            raise ValueError("Manifest does not match TPC-DS target")
    if args.phase in {"materialize", "all"}:
        manifest = materialize(args, manifest)
        base.write_manifest(args.manifest_path, manifest)
    if args.phase in {"validate", "all"}:
        if not base.database_exists(args, args.target_db):
            raise FileNotFoundError(args.target_db)
        manifest = validate_target(args, manifest)
        base.write_manifest(args.manifest_path, manifest)
    print(f"Wrote manifest to {args.manifest_path}")


if __name__ == "__main__":
    main()
