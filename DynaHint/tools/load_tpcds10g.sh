#!/usr/bin/env bash
set -euo pipefail

DB_NAME="${DB_NAME:-tpcds10g}"
PG_HOST="${PG_HOST:-localhost}"
PG_PORT="${PG_PORT:-5433}"
PG_USER="${PG_USER:-houyuheng}"
DATA_DIR="${DATA_DIR:-/home/houyuheng/TPCDS}"
TOOLS_DIR="${TOOLS_DIR:-/home/houyuheng/download/DSGen-software-code-4.0.0/tools}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_SQL="${SCHEMA_SQL:-$TOOLS_DIR/tpcds.sql}"
RI_SQL="${RI_SQL:-$SCRIPT_DIR/tpcds_ri_patched.sql}"
FORCE_LOAD="${FORCE_LOAD:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash DynaHint/tools/load_tpcds10g.sh

Optional environment overrides:
  DB_NAME=tpcds10g
  PG_HOST=localhost
  PG_PORT=5433
  PG_USER=houyuheng
  DATA_DIR=/home/houyuheng/TPCDS
  TOOLS_DIR=/home/houyuheng/download/DSGen-software-code-4.0.0/tools
  SCHEMA_SQL=/path/to/tpcds.sql
  RI_SQL=/path/to/tpcds_ri_patched.sql
  FORCE_LOAD=1

Examples:
  bash DynaHint/tools/load_tpcds10g.sh
  DB_NAME=tpcds10g DATA_DIR=/data/TPCDS bash DynaHint/tools/load_tpcds10g.sh
  FORCE_LOAD=1 bash DynaHint/tools/load_tpcds10g.sh
EOF
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
}

run_psql() {
  psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 "$@"
}

load_table() {
  local table="$1"
  local file="$DATA_DIR/$table.dat"

  require_file "$file"
  log "Loading $table from $file"
  sed 's/|$//' "$file" | run_psql -c "\copy $table FROM STDIN WITH (FORMAT text, DELIMITER '|', NULL '')"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

require_file "$SCHEMA_SQL"
require_file "$RI_SQL"

tables=(
  dbgen_version
  customer_address
  customer_demographics
  date_dim
  warehouse
  ship_mode
  time_dim
  reason
  income_band
  item
  store
  call_center
  customer
  web_site
  store_returns
  household_demographics
  web_page
  promotion
  catalog_page
  inventory
  catalog_returns
  web_returns
  web_sales
  catalog_sales
  store_sales
)

for table in "${tables[@]}"; do
  require_file "$DATA_DIR/$table.dat"
done

table_count="$(run_psql -Atc "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")"
if [[ "$table_count" != "0" && "$FORCE_LOAD" != "1" ]]; then
  echo "Database $DB_NAME is not empty (public tables=$table_count). Set FORCE_LOAD=1 to continue." >&2
  exit 1
fi

if [[ "$FORCE_LOAD" == "1" && "$table_count" != "0" ]]; then
  log "FORCE_LOAD=1 set; existing public tables will be dropped by reloading schema table-by-table."
fi

log "Loading TPC-DS schema into $DB_NAME"
run_psql -f "$SCHEMA_SQL"

for table in "${tables[@]}"; do
  load_table "$table"
done

log "Applying patched referential integrity constraints"
run_psql -f "$RI_SQL"

log "Running ANALYZE"
run_psql -c "ANALYZE;"

log "Validating key row counts"
run_psql -Atc "
SELECT 'table_count' || E'\t' || count(*)
FROM information_schema.tables
WHERE table_schema='public'
UNION ALL
SELECT 'store_sales' || E'\t' || count(*) FROM public.store_sales
UNION ALL
SELECT 'catalog_sales' || E'\t' || count(*) FROM public.catalog_sales
UNION ALL
SELECT 'web_sales' || E'\t' || count(*) FROM public.web_sales
UNION ALL
SELECT 'inventory' || E'\t' || count(*) FROM public.inventory
UNION ALL
SELECT 'customer' || E'\t' || count(*) FROM public.customer
ORDER BY 1;
"

log "TPC-DS load completed successfully"
