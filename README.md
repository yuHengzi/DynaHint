# DynaHint: Drift-Aware Learned Query Optimization via Hint-Based Plan Repair

DynaHint is a learned PostgreSQL query optimizer. Starting from PostgreSQL's
original plan, it explores candidate plans using hint actions and selects plans
with a learned scorer. It supports data-drift and workload-shift experiments.

Repository: <https://github.com/yuHengzi/DynaHint> · [中文说明](README_zh.md)

## Installation

Reference environment: Linux, Python 3.7, PyTorch 1.12.1, Ray RLlib 2.4.0,
and PostgreSQL 18.

```bash
git clone https://github.com/yuHengzi/DynaHint.git
cd DynaHint
conda create -n dynahint python=3.7 -y
conda activate dynahint
python -m pip install -r requirements.txt
```

For the CUDA 11.3 PyTorch build:

```bash
python -m pip install 'torch==1.12.1+cu113' \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

Install PostgreSQL and a compatible `pg_hint_plan` separately. The database role
must be allowed to execute `LOAD 'pg_hint_plan'`. Database construction requires
database-creation privileges and matching `pg_dump` and `psql` binaries.

## Data Preparation

Import benchmark data, schemas and indexes into PostgreSQL. Name the source
databases `imdb`, `stats`, and `tpcds`.

Edit connection settings and run the slicing script for your workload:

```bash
python DynaHint/tools/database/create_imdb_DB.py
python DynaHint/tools/database/create_stats_halfyear_DB.py
python DynaHint/tools/database/create_tpcds_halfyear_DB.py
```

Generate training databases with the corresponding launcher. For STATS:

```bash
export PG_BIN_DIR=/path/to/postgresql/bin
PYTHON_BIN="$(command -v python)" PHASE=preflight \
  bash DynaHint/tools/drift/run_stats_201306_pg18_diverse.sh
PYTHON_BIN="$(command -v python)" PHASE=all \
  bash DynaHint/tools/drift/run_stats_201306_pg18_diverse.sh
```

For IMDB, use `run_imdb1980cut_pg18_diverse.sh` with `PHASE=preflight` followed
by `PHASE=all`. For TPC-DS:

```bash
PYTHON="$(command -v python)" PHASE=preflight \
  bash DynaHint/tools/drift/run_tpcds_200012_pg18.sh
PYTHON="$(command -v python)" PHASE=build-validate \
  bash DynaHint/tools/drift/run_tpcds_200012_pg18.sh
```

## Configuration and Training

Edit `DynaHint/shared/config.py` to set database connections, workload,
database lists and compute resources. The default is STATS Full:

```python
self.mode = 'STATS'
self.port = 5432
self.train_databases = 6
self.train_mode = 'mix+'
self.total_iter = 2000
```

The first six entries in `self.databases` are the five generated databases and
the source database; remaining entries are evaluation databases. When switching
to JOB or TPCDS, update both `mode` and the database list. Check
`CUDA_VISIBLE_DEVICES` before running; the default GPU index is `1`.

```bash
PYTHONPATH=DynaHint python -u DynaHint/run_parallel.py --ExpName stats_full
```

The workflow prepares histograms, encoding, metadata, training cardinalities
and baselines before training and validation. Set `adaptive_gate_mode` to
`nearest` (default), `baseline`, or `off` to configure TAG. Repeated measurements
are controlled by `latency_repeat_phases`, `latency_warmup_runs`,
`latency_measure_runs`, and `latency_aggregate`.

View TensorBoard metrics:

```bash
tensorboard --logdir runstate --port 6006
```

## Resume and Evaluate

Resume training:

```bash
PYTHONPATH=DynaHint python DynaHint/run_parallel.py \
  --ContinueTrain model/postgres_STATS/stats_full_mix_plus
```

Evaluate a saved model:

```bash
python DynaHint/evaluation/evaluate_saved_model.py \
  --model-dir model/postgres_STATS/stats_full_mix_plus \
  --databases stats_201206 stats_201212 stats_201312 stats \
  --exp-name stats_full_eval \
  --checkpoint-metric train_wrl \
  --host localhost --port 5432
```

Replace the model path and database list with your own values. Use
`--checkpoint-metric train_gmrl` for the best Train GMRL checkpoint.
Additional arguments are listed by each entry point's `--help`.

## Layout and Outputs

| Path | Contents |
| --- | --- |
| `DynaHint/shared/` | Configuration, managers and TAG |
| `DynaHint/collection/` | Data collection and feature encoding |
| `DynaHint/training/` | Planner, scorer and training workflow |
| `DynaHint/evaluation/` | Validation and saved-model evaluation |
| `DynaHint/tools/` | Data slicing and training-database generators |
| `experiment/` | SQL workloads and collected histograms |
| `model/` | Models, encoding, metadata and training information |
| `latency/`, `latencybuffer/` | Baselines, cardinalities and execution experience |
| `runstate/` | TensorBoard events |
| `timely_result/`, `result/` | Runtime records, evaluation results and diagnostics |

WRL and GMRL use execution time; lower is better. Speedup compares total time
including optimization overhead; higher is better. Raw data and pretrained
models are not bundled.
