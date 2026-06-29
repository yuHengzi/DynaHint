# DynaHint

DynaHint is a learned query hint recommendation system for database query optimization. It starts from PostgreSQL's original optimizer plan, explores candidate plans by applying hint actions, and selects a final hinted SQL using a learned scorer. The current implementation targets PostgreSQL.

## Requirements

- Python 3.7
- PyTorch 1.12.x
- Ray RLlib 2.4.0
- PostgreSQL v12.1 with hint support
  - PostgreSQL experiments usually require [`pg_hint_plan`](https://github.com/yxfish13/PostgreSQL12.1_hint).

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

## Repository Layout

```text
DynaHint/                         Core implementation
  config.py                       Database, workload, model, and training configuration
  DynaHintEnv.py                  RL environment for plan exploration
  database_util.py                Plan parsing and local node feature utilities
  datacollector.py                Scorer training data collection
  encoding.py                     Table, column, type, and operator encoding cache
  learner.py                      Scorer training logic
  manager.py                      Result, metric, and diagnostic management
  model.py                        Planner/scorer models and Local-Global context fusion layers
  pointestimator.py               Pointwise scorer and candidate selection
  pghelper.py                     PostgreSQL execution helper
  planhelper.py                   Plan feature construction and Local-Global context assembly
  run_parallel.py                 Main training and inference entry point
  tools/                          Dataset cutting and histogram collection utilities
experiment/                       Workload SQL files and histogram files
latency/                          Baseline latency and cardinality caches
latencybuffer/                    Candidate-plan execution buffers
model/                            Saved encodings, checkpoints, and model metadata
result/                           Query traces, diagnostics, and exported reports
runstate/                         TensorBoard event files
```

## Database and Workload Preparation

### 1. Configure Database Connection

DynaHint reads database connection settings from environment variables by default. Set them before running training or inference:

```bash
export DYNAHINT_PG_HOST=localhost
export DYNAHINT_PG_PORT=5432
export DYNAHINT_PG_USER=<your_user>
export DYNAHINT_PG_PASSWORD=<your_password>
```

Then edit `DynaHint/config.py` for the experiment-specific settings:

- `self.DBMS`: `postgres`
- `self.mode`: workload name, for example `JOB`, `STATS`, or `TPCDS`
- `self.databases`: ordered source/target databases for drift experiments
- `self.train_mode`: `data_drift`, `query_drift`, `mix`, or `mix+`
- `self.expname`: unique experiment name

### 2. Collect Histograms

DynaHint uses histogram files in `experiment/histogram/`. Collect them after the databases and workload SQL files are ready.

IMDb/JOB:

```bash
python DynaHint/tools/collect_imdb_histogram.py --database imdb
```

STATS:

```bash
python DynaHint/tools/collect_stats_histogram.py
```

TPC-DS:

```bash
python DynaHint/tools/collect_tpcds_histogram.py
```

The generated files are named like:

```text
experiment/histogram/<database>_histogram_string.json
```

## Training

After editing `DynaHint/config.py`, start training with:

```bash
python -u DynaHint/run_parallel.py
```

## Results

Training metrics are written to TensorBoard event files under `runstate/`:

```bash
tensorboard --logdir ./runstate
```

Detailed traces and diagnostics are written under `result/<DBMS>_<WORKLOAD>/`, including query-time traces, planning breakdowns, and generalization diagnostics when enabled.

Latency and cardinality caches are stored under `latency/`; candidate execution buffers are stored under `latencybuffer/`.

## Notes

- `WRL` reports execution-time weighted runtime ratio by default.
- `GMRL` reports geometric mean runtime ratio using execution time.
- `Speedup` baseline total time divided by DynaHint total time.
- Lower WRL/GMRL is better; higher Speedup is better.
