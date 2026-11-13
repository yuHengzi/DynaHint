#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/houyuheng/miniconda3/envs/foss/bin/python}"
PG_BIN_DIR="${PG_BIN_DIR:-/home/houyuheng/.local/opt/postgresql-18.6/bin}"
PHASE="${PHASE:-preflight}"
PORT="${PORT:-5432}"
SOURCE_DB="${SOURCE_DB:-imdb1980cut}"
ROOT_COUNT="${ROOT_COUNT:-20000}"
BINS="1940,1960,1965,1970,1975"
CONFIG_DIR="${SCRIPT_DIR}/drift_configs/imdb1980cut_pg18"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/imdb1980cut_pg18_diverse}"

run_case() {
    local case_id="$1"
    local seed="$2"
    local target_db="imdb1980cut_rel_${case_id}"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/build_relational_drift.py" \
        --profile imdb \
        --phase "${PHASE}" \
        --scenario distribution \
        --host localhost \
        --port "${PORT}" \
        --pg-bin-dir "${PG_BIN_DIR}" \
        --source-db "${SOURCE_DB}" \
        --target-db "${target_db}" \
        --root-count "${ROOT_COUNT}" \
        --bins "${BINS}" \
        --target-weights "${CONFIG_DIR}/${case_id}.json" \
        --seed "${seed}" \
        --artifact-dir "${ARTIFACT_ROOT}/${case_id}"
}

run_case d1 26091401
run_case d2 26091402
run_case d3 26091403
run_case d4 26091404
run_case d5 26091405
