#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/houyuheng/miniconda3/envs/foss/bin/python}"
PG_BIN_DIR="${PG_BIN_DIR:-/home/houyuheng/.local/opt/postgresql-18.6/bin}"
PHASE="${PHASE:-preflight}"
PORT="${PORT:-5432}"
SOURCE_DB="${SOURCE_DB:-stats_201306}"
ROOT_COUNT="${ROOT_COUNT:-20000}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/stats_201306_pg18_diverse}"

for case_id in d1 d2 d3 d4 d5; do
    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_stats_relational_drift.py" \
        --profile stats \
        --phase "${PHASE}" \
        --scenario distribution \
        --host localhost \
        --port "${PORT}" \
        --pg-bin-dir "${PG_BIN_DIR}" \
        --source-db "${SOURCE_DB}" \
        --target-db "stats_201306_rel_${case_id}" \
        --root-count "${ROOT_COUNT}" \
        --target-weights "${SCRIPT_DIR}/drift_configs/stats_201306_pg18/${case_id}.json" \
        --seed "$((26100900 + 10#${case_id#d}))" \
        --artifact-dir "${ARTIFACT_ROOT}/${case_id}"
done
