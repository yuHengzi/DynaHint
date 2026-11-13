"""Prepare histogram JSON files before starting Ray workers."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory


COLLECTORS = {
    'JOB': 'collect_imdb_histogram.py',
    'JOBRand': 'collect_imdb_histogram.py',
    'STATS': 'collect_stats_histogram.py',
    'TPCDS': 'collect_tpcds_histogram.py',
}

TPCDS_FACT_DATE_KEYS = (
    'store_sales.ss_sold_date_sk',
    'catalog_sales.cs_sold_date_sk',
    'web_sales.ws_sold_date_sk',
    'store_returns.sr_returned_date_sk',
    'catalog_returns.cr_returned_date_sk',
    'web_returns.wr_returned_date_sk',
    'inventory.inv_date_sk',
)


def _valid_histogram(path, bins_per_col, mode=None):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            records = json.load(file)
    except (OSError, ValueError):
        return False
    valid = bool(records) and isinstance(records, list) and all(
        isinstance(record, dict)
        and isinstance(record.get('table_column'), str)
        and isinstance(record.get('bins'), list)
        and len(record['bins']) == bins_per_col
        and all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in record['bins']
        )
        for record in records
    )
    if not valid or mode != 'TPCDS':
        return valid
    fact_records = {record['table_column']: record for record in records}
    return all(
        key in fact_records
        and isinstance(fact_records[key].get('year_fractions'), list)
        and len(fact_records[key]['year_fractions']) == 5
        and all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in fact_records[key]['year_fractions'])
        for key in TPCDS_FACT_DATE_KEYS
    )


def ensure_histograms(config):
    """Collect only missing/invalid histograms for configured PostgreSQL databases."""
    if getattr(config, 'DBMS', None) != 'postgres':
        return
    if not (getattr(config, 'node_hist_dim', 0) > 0 or (
        getattr(config, 'use_db_features', False)
        and getattr(config, 'db_feature_source', None) == 'DynaHint'
        and getattr(config, 'db_hist_dim', 0) > 0
    )):
        return

    databases = list(dict.fromkeys(getattr(config, 'databases', [])))
    if getattr(config, 'train_mode', None) == 'query_drift':
        databases = databases[:1]
    if not databases:
        raise ValueError('Histogram preflight has no target databases in config.databases')

    output_dir = Path(config.hist_json_dir)
    mode = getattr(config, 'mode', None)
    bins_per_col = int(config.db_hist_bins_per_col)
    if bins_per_col < 2:
        raise ValueError('db_hist_bins_per_col must be at least 2')
    missing = [
        database for database in databases
        if not _valid_histogram(output_dir / '{}_histogram_string.json'.format(database), bins_per_col, mode)
    ]
    if not missing:
        print('Histogram preflight: all {} database files are ready.'.format(len(databases)), flush=True)
        return

    script_name = COLLECTORS.get(mode)
    if script_name is None:
        raise RuntimeError('No histogram collector for mode={!r}; missing databases: {}'.format(mode, missing))
    script = Path(__file__).resolve().parent / script_name
    if not script.is_file():
        raise FileNotFoundError('Histogram collector not found: {}'.format(script))

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.histogram-preflight-', dir=str(output_dir)) as temp_dir:
        command = [
            sys.executable, str(script), '--databases', *missing,
            '--output-dir', temp_dir,
            '--host', str(config.ip), '--port', str(config.port),
            '--user', str(config.user or ''), '--bins', str(bins_per_col),
        ]
        environment = os.environ.copy()
        if getattr(config, 'password', None):
            environment['PGPASSWORD'] = str(config.password)
        if mode in ('STATS', 'TPCDS'):
            command.extend([
                '--workload-dirs', config.train_workload_path, config.test_workload_path,
            ])
        print('Histogram preflight: collecting {}: {}'.format(len(missing), ', '.join(missing)), flush=True)
        try:
            subprocess.run(command, check=True, env=environment)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Histogram collection failed for: {}'.format(', '.join(missing))) from exc

        invalid = [
            database for database in missing
            if not _valid_histogram(Path(temp_dir) / '{}_histogram_string.json'.format(database), bins_per_col, mode)
        ]
        if invalid:
            raise RuntimeError('Histogram collection produced missing/invalid files for: {}'.format(', '.join(invalid)))
        for database in missing:
            filename = '{}_histogram_string.json'.format(database)
            os.replace(str(Path(temp_dir) / filename), str(output_dir / filename))
    print('Histogram preflight: ready for {} databases.'.format(len(databases)), flush=True)
