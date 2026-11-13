#!/usr/bin/env python3
"""Build equal-root-count STATS drift databases from posts in a source cut."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import psycopg2

import build_relational_drift as base


PROFILE = base.DriftProfile(
    name="stats",
    default_source_db="stats_201306",
    tables=base.STATS_TABLES,
    relationships=base.STATS_RELATIONSHIPS,
    root_from_sql="stats_root_features AS root",
    root_id_sql="root.id",
    root_value_sql="root.creationdate::date",
    root_category_sql="root.category",
    default_bins="2010-12-31,2011-12-31,2012-12-31",
    bin_type="date",
)

FEATURE_TABLE_SQL = """
CREATE TEMP TABLE stats_root_features ON COMMIT PRESERVE ROWS AS
WITH comments AS (
    SELECT postid, COUNT(*) AS n FROM public.comments GROUP BY postid
), votes AS (
    SELECT postid, COUNT(*) AS n FROM public.votes GROUP BY postid
), history AS (
    SELECT postid, COUNT(*) AS n FROM public.posthistory GROUP BY postid
)
SELECT p.id, p.creationdate,
    (CASE WHEN p.posttypeid = 1 THEN 'question'
          WHEN p.posttypeid = 2 THEN 'answer' ELSE 'other' END) || '|' ||
    (CASE WHEN COALESCE(comments.n, 0) + COALESCE(votes.n, 0) +
               COALESCE(history.n, 0) <= 5 THEN 'low'
          WHEN COALESCE(comments.n, 0) + COALESCE(votes.n, 0) +
               COALESCE(history.n, 0) <= 20 THEN 'medium'
          ELSE 'high' END) AS category
FROM public.posts AS p
LEFT JOIN comments ON comments.postid = p.id
LEFT JOIN votes ON votes.postid = p.id
LEFT JOIN history ON history.postid = p.id
"""


def ensure_root_features(connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT to_regclass('pg_temp.stats_root_features')")
        if cursor.fetchone()[0] is not None:
            return
        cursor.execute(FEATURE_TABLE_SQL)
        cursor.execute("CREATE UNIQUE INDEX ON stats_root_features (id)")
        cursor.execute("ANALYZE stats_root_features")


def load_factor_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        raise ValueError("--target-weights must name a STATS factor config")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Factor config must be a JSON object")
    source_mix = float(payload.get("source_mix", 0.3))
    if not 0 < source_mix <= 1 or not math.isfinite(source_mix):
        raise ValueError("source_mix must be finite and in (0, 1]")
    factors = payload.get("factors")
    if not isinstance(factors, dict) or set(factors) != {"period", "kind", "activity"}:
        raise ValueError("factors must contain exactly: period, kind, activity")
    normalized = {}
    for dimension, values in factors.items():
        if not isinstance(values, dict) or not values:
            raise ValueError(f"{dimension} must be a non-empty object")
        converted = {str(key): float(value) for key, value in values.items()}
        if any(not math.isfinite(value) or value < 0 for value in converted.values()):
            raise ValueError(f"{dimension} weights must be finite and non-negative")
        if not any(converted.values()):
            raise ValueError(f"{dimension} needs a positive weight")
        normalized[dimension] = converted
    return {"source_mix": source_mix, "factors": normalized}


def expand_factor_weights(source_strata: Mapping[str, int], config: Mapping[str, Any]) -> dict[str, float]:
    factors = config["factors"]
    raw = {}
    for stratum in source_strata:
        parts = stratum.split("|")
        if len(parts) != 3:
            raise ValueError(f"Unexpected STATS stratum: {stratum}")
        weight = 1.0
        for dimension, label in zip(("period", "kind", "activity"), parts):
            weight *= factors[dimension].get(label, factors[dimension].get("*", 0.0))
        raw[stratum] = weight
    source_total, target_total = sum(source_strata.values()), sum(raw.values())
    if not source_total or not target_total:
        raise ValueError("Observed strata and expanded factor weights must be non-empty")
    mix = config["source_mix"]
    return {
        stratum: mix * count / source_total + (1 - mix) * raw[stratum] / target_total
        for stratum, count in source_strata.items()
    }


def allocate_covered_quotas(capacities: Mapping[str, int], root_count: int,
                            weights: Mapping[str, float]) -> dict[str, int]:
    """Reserve one root per observed stratum, then allocate the remainder."""
    if root_count < len(capacities):
        raise ValueError("Root count is too small to cover every observed stratum")
    remaining = {key: count - 1 for key, count in capacities.items()}
    extra = base.allocate_stratum_quotas(remaining, root_count - len(capacities), weights)
    return {key: 1 + extra[key] for key in capacities}


def preflight(args, profile: base.DriftProfile) -> dict[str, Any]:
    boundaries = base.parse_bins(profile, args.bins)
    config = load_factor_config(args.target_weights)
    with psycopg2.connect(**base.connection_kwargs(args, args.source_db)) as source:
        base.validate_source_schema(source, profile)
        ensure_root_features(source)
        strata = base.fetch_stratum_counts(source, profile, boundaries)
    available = sum(strata.values())
    root_count = args.root_count if args.root_count is not None else math.ceil(available * args.sample_scale)
    weights = expand_factor_weights(strata, config)
    quotas = allocate_covered_quotas(strata, root_count, weights)
    return {
        "tool": "build_stats_relational_drift", "phase": "preflight",
        "profile": profile.name, "profile_version": "stats",
        "source_db": args.source_db, "target_db": args.target_db,
        "seed": args.seed, "requested_root_count": root_count,
        "bins": [str(value) for value in boundaries], "factor_config": config,
        "target_weights": weights, "source_strata": strata,
        "requested_quotas": quotas,
        "target_already_exists": base.database_exists(args, args.target_db),
    }


def materialize(args, profile: base.DriftProfile, manifest: dict[str, Any]) -> dict[str, Any]:
    boundaries = base.parse_bins(profile, ",".join(manifest["bins"]))
    with psycopg2.connect(**base.connection_kwargs(args, args.source_db)) as source:
        columns = base.validate_source_schema(source, profile)
        ensure_root_features(source)
        selected = base.select_seed_ids(source, profile, boundaries, manifest["requested_quotas"], args.seed)
        base.create_selection_tables(source, profile, selected)
        base.create_target_database(args)
        pre_data, post_data = base.dump_schema_sections(args)
        base.apply_sql_file(args, pre_data)
        with psycopg2.connect(**base.connection_kwargs(args, args.target_db)) as target:
            counts = {}
            for table in profile.tables:
                print(f"Copying {table}", flush=True)
                counts[table] = base.copy_table(source, target, profile, table, columns[table])
        base.apply_sql_file(args, post_data)
        with psycopg2.connect(**base.connection_kwargs(args, args.target_db)) as target:
            with target.cursor() as cursor:
                cursor.execute("ANALYZE")
    manifest.update({"phase": "materialized", "selected_root_count": len(selected), "table_counts": counts})
    return manifest


def validate_target(args, profile: base.DriftProfile, manifest: dict[str, Any]) -> dict[str, Any]:
    boundaries = base.parse_bins(profile, ",".join(manifest["bins"]))
    with psycopg2.connect(**base.connection_kwargs(args, args.target_db)) as target:
        ensure_root_features(target)
        orphans = base.relation_orphan_counts(target, profile)
        target_strata = base.fetch_stratum_counts(target, profile, boundaries)
    validation = {
        "phase": "validated", "relation_orphans": orphans,
        "all_relationships_valid": not any(orphans.values()),
        "target_strata": target_strata,
        "requested_target_jsd": base.jensen_shannon_divergence(manifest["requested_quotas"], target_strata),
        "source_target_jsd": base.jensen_shannon_divergence(manifest["source_strata"], target_strata),
    }
    if args.query_dir is not None:
        databases = list(dict.fromkeys([args.target_db, *args.comparison_dbs]))
        results = base.benchmark_queries(args, databases)
        path = args.artifact_dir / "query_results.json"
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        validation["query_results_path"] = str(path)
    manifest.update(validation)
    return manifest


def main() -> None:
    base.PROFILES["stats"] = PROFILE
    args = base.parse_args()
    profile = base.PROFILES[args.profile]
    if args.profile != "stats":
        raise ValueError("This entry point requires --profile stats")
    if args.scenario != "distribution":
        raise ValueError("STATS requires --scenario distribution")
    if shutil.disk_usage(Path.cwd()).free < 4 * 1024**3:
        raise RuntimeError("At least 4 GiB free space is required")
    if args.phase in {"preflight", "materialize", "all"}:
        manifest = preflight(args, profile)
        base.write_manifest(args.manifest_path, manifest)
    else:
        if not args.manifest_path.exists():
            raise FileNotFoundError(f"Missing manifest: {args.manifest_path}")
        manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("profile_version") != "stats" or manifest.get("target_db") != args.target_db:
            raise ValueError("Manifest does not match STATS target")
    if args.phase in {"materialize", "all"}:
        manifest = materialize(args, profile, manifest)
        base.write_manifest(args.manifest_path, manifest)
    if args.phase in {"validate", "all"}:
        if not base.database_exists(args, args.target_db):
            raise FileNotFoundError(f"Target database does not exist: {args.target_db}")
        manifest = validate_target(args, profile, manifest)
        base.write_manifest(args.manifest_path, manifest)
    print(f"Wrote manifest to {args.manifest_path}")


if __name__ == "__main__":
    main()
