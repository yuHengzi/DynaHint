# DynaHint: Drift-Aware Learned Query Optimization via Hint-Based Plan Repair

DynaHint 是面向 PostgreSQL 的学习型查询优化系统。系统从 PostgreSQL 原始计划出发，通过 hint 动作探索候选计划，再由 Scorer 选择计划，支持数据偏移和工作负载偏移实验。

项目地址：<https://github.com/yuHengzi/DynaHint>

## 环境安装

参考环境：Linux、Python 3.7、PyTorch 1.12.1、Ray RLlib 2.4.0、PostgreSQL 18。

```bash
git clone https://github.com/yuHengzi/DynaHint.git
cd DynaHint
conda create -n dynahint python=3.7 -y
conda activate dynahint
python -m pip install -r requirements.txt
```

如需使用 CUDA 11.3 对应的 PyTorch：

```bash
python -m pip install 'torch==1.12.1+cu113' \
  --extra-index-url https://download.pytorch.org/whl/cu113
```

另行安装 PostgreSQL 和匹配版本的 `pg_hint_plan`，确保数据库角色可以执行 `LOAD 'pg_hint_plan'`。数据构造需要建库权限及匹配版本的 `pg_dump`、`psql`。

## 数据准备

将基准数据、表结构和索引导入 PostgreSQL，源库分别命名为 `imdb`、`stats`、`tpcds`。

按需修改连接参数并运行时间切分脚本：

```bash
python DynaHint/tools/database/create_imdb_DB.py
python DynaHint/tools/database/create_stats_halfyear_DB.py
python DynaHint/tools/database/create_tpcds_halfyear_DB.py
```

使用对应负载的脚本生成训练库。以 STATS 为例：

```bash
export PG_BIN_DIR=/path/to/postgresql/bin
PYTHON_BIN="$(command -v python)" PHASE=preflight \
  bash DynaHint/tools/drift/run_stats_201306_pg18_diverse.sh
PYTHON_BIN="$(command -v python)" PHASE=all \
  bash DynaHint/tools/drift/run_stats_201306_pg18_diverse.sh
```

IMDB 使用 `run_imdb1980cut_pg18_diverse.sh`，同样依次设置 `PHASE=preflight` 和 `PHASE=all`。TPC-DS 使用：

```bash
PYTHON="$(command -v python)" PHASE=preflight \
  bash DynaHint/tools/drift/run_tpcds_200012_pg18.sh
PYTHON="$(command -v python)" PHASE=build-validate \
  bash DynaHint/tools/drift/run_tpcds_200012_pg18.sh
```

## 配置与训练

编辑 `DynaHint/shared/config.py`，设置数据库连接、负载、数据库列表和计算资源。默认使用 STATS Full：

```python
self.mode = 'STATS'
self.port = 5432
self.train_databases = 6
self.train_mode = 'mix+'
self.total_iter = 2000
```

`self.databases` 前六项为五个生成库和源库，其余为测试库。切换 JOB 或 TPCDS 时，同时修改 `mode` 和数据库列表。运行前检查 `CUDA_VISIBLE_DEVICES`，默认 GPU 索引为 `1`。

```bash
PYTHONPATH=DynaHint python -u DynaHint/run_parallel.py --ExpName stats_full
```

程序自动准备直方图、编码、metadata、训练基数和 Baseline，然后训练并验证模型。TAG 开关为 `adaptive_gate_mode`，支持 `nearest`（默认）、`baseline`、`off`。重复执行测量通过 `latency_repeat_phases`、`latency_warmup_runs`、`latency_measure_runs` 和 `latency_aggregate` 设置。

查看 TensorBoard：

```bash
tensorboard --logdir runstate --port 6006
```

## 恢复与评估

恢复训练：

```bash
PYTHONPATH=DynaHint python DynaHint/run_parallel.py \
  --ContinueTrain model/postgres_STATS/stats_full_mix_plus
```

评估保存的模型：

```bash
python DynaHint/evaluation/evaluate_saved_model.py \
  --model-dir model/postgres_STATS/stats_full_mix_plus \
  --databases stats_201206 stats_201212 stats_201312 stats \
  --exp-name stats_full_eval \
  --checkpoint-metric train_wrl \
  --host localhost --port 5432
```

将模型目录和数据库列表替换为实际值。选择 Train GMRL 最优模型时，使用 `--checkpoint-metric train_gmrl`。更多参数见各入口的 `--help`。

## 目录与输出

| 路径 | 内容 |
| --- | --- |
| `DynaHint/shared/` | 配置、管理器和 TAG |
| `DynaHint/collection/` | 数据采集和特征编码 |
| `DynaHint/training/` | Planner、Scorer 和训练流程 |
| `DynaHint/evaluation/` | 验证和保存模型评估 |
| `DynaHint/tools/` | 数据切分和训练库生成工具 |
| `experiment/` | SQL 负载及采集的直方图 |
| `model/` | 模型、编码、metadata 和训练信息 |
| `latency/`、`latencybuffer/` | Baseline、基数和执行经验缓存 |
| `runstate/` | TensorBoard 事件 |
| `timely_result/`、`result/` | 运行记录、评估结果和诊断 |

WRL 和 GMRL 使用执行时间，越小越好；Speedup 比较包含优化开销的总时间，越大越好。原始数据和预训练模型不随代码提供。
