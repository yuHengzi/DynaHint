#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
TOOL_DIR="$ROOT_DIR/DynaHint/tools/drift"
PHASE=${PHASE:-preflight}
ROOT_COUNT=${ROOT_COUNT:-500}
START=${START:-1}
END=${END:-5}
PYTHON=${PYTHON:-/home/houyuheng/miniconda3/envs/foss/bin/python}
PG_BIN_DIR=${PG_BIN_DIR:-/home/houyuheng/.local/opt/postgresql-18.6/bin}

for ((n=START; n<=END; n++)); do
    db="tpcds_200012_rel_d${n}"
    if [[ "$PHASE" == "build-validate" ]]; then
        phases=(materialize validate)
    else
        phases=("$PHASE")
    fi
    for phase in "${phases[@]}"; do
        "$PYTHON" "$TOOL_DIR/build_tpcds_relational_drift.py" \
            --profile tpcds --scenario distribution --phase "$phase" \
            --host localhost --port 5432 --source-db tpcds_200012 \
            --target-db "$db" --root-count "$ROOT_COUNT" --seed "$((20261011 + n))" \
            --target-weights "$TOOL_DIR/drift_configs/tpcds_200012_pg18/d${n}.json" \
            --query-dir "$ROOT_DIR/experiment/TPCDS/train" \
            --pg-bin-dir "$PG_BIN_DIR" \
            --artifact-dir "$TOOL_DIR/artifacts/tpcds_200012_pg18/$db"
    done
done
