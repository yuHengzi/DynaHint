#!/usr/bin/env python3
"""Evaluate a saved DynaHint model on test databases without starting training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import ray


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DYNAHINT_ROOT = PROJECT_ROOT / "DynaHint"
if str(DYNAHINT_ROOT) not in sys.path:
    sys.path.insert(0, str(DYNAHINT_ROOT))
existing_pythonpath = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    part for part in (str(DYNAHINT_ROOT), existing_pythonpath) if part
)

from shared.config import Config  # noqa: E402
import run_parallel  # noqa: E402


DynaHint = run_parallel.DynaHint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--databases", nargs="+", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--val-iter", type=int)
    parser.add_argument("--max-steps", type=int, help="Override the saved model's rollout step limit for evaluation.")
    parser.add_argument("--test-stop-min-steps", type=int)
    parser.add_argument("--checkpoint-metric", default="train_wrl")
    parser.add_argument("--test-only", action="store_true", help="Skip TestDataDrift evaluation.")
    parser.add_argument('--adaptive-gate-mode', choices=['off', 'baseline', 'nearest'])
    parser.add_argument('--nearest-tag-model', type=Path, help='Override the nearest TAG model JSON.')
    parser.add_argument(
        "--scorer-path", type=Path,
        help="Override the checkpoint scorer for an isolated scorer-only evaluation.",
    )
    diagnostics_group = parser.add_mutually_exclusive_group()
    diagnostics_group.add_argument(
        "--selection-strategy-diagnostics",
        dest="selection_strategy_diagnostics",
        action="store_true",
        help="Execute every generated candidate and compare oracle, scorer, and safe-select.",
    )
    diagnostics_group.add_argument(
        "--no-selection-strategy-diagnostics",
        dest="selection_strategy_diagnostics",
        action="store_false",
        help="Disable candidate strategy diagnostics for a faster evaluation.",
    )
    parser.set_defaults(selection_strategy_diagnostics=None)
    return parser.parse_args()


def configure(args: argparse.Namespace) -> tuple[Config, int]:
    model_dir = args.model_dir.resolve()
    train_info_path = model_dir / "trainfo.json"
    if not train_info_path.is_file():
        raise FileNotFoundError(f"missing train info: {train_info_path}")

    train_info = json.loads(train_info_path.read_text(encoding="utf-8"))
    checkpoint_metric = {
        "wrl": "train_wrl",
        "gmrl": "train_gmrl",
        "train_wrl_noPlanningTime": "train_wrl",
    }.get(args.checkpoint_metric, args.checkpoint_metric)
    checkpoint = train_info.get("best_checkpoints", {}).get(checkpoint_metric)
    val_iter = args.val_iter or int(
        checkpoint.get("iteration", 0) if checkpoint else train_info.get("model_saved_iter", 0)
    )
    if val_iter <= 0:
        raise ValueError("--val-iter is required when the selected checkpoint has no iteration")

    config = Config()
    for part in model_dir.parts:
        if part in ('postgres_JOB', 'postgres_STATS', 'postgres_TPCDS'):
            config.mode = part.split('_', 1)[1]
    saved_gate_mode = train_info.get('adaptive_gate_mode')
    if saved_gate_mode is None and 'adaptive_gate_enable' in train_info:
        saved_gate_mode = 'baseline' if train_info['adaptive_gate_enable'] else 'off'
    config.adaptive_gate_mode = getattr(args, 'adaptive_gate_mode', None) or saved_gate_mode or config.adaptive_gate_mode
    if getattr(args, 'nearest_tag_model', None) is not None:
        config.nearest_tag_model_path = str(args.nearest_tag_model.resolve())
    config.expname = args.exp_name
    config.ip = args.host
    config.port = args.port
    config.databases = list(args.databases)
    # A negative full-length slice gives an empty train set and keeps every DB in test.
    config.train_databases = -len(config.databases)
    config.train_mode = "mix+"
    config.action_mode = train_info.get("action_mode", config.action_mode)
    config.cost_plan_timeoutcoeff = train_info.get(
        "cost_plan_timeoutcoeff", config.cost_plan_timeoutcoeff
    )
    config.cost_plan_timeout_min_ms = train_info.get(
        "cost_plan_timeout_min_ms", config.cost_plan_timeout_min_ms
    )
    config.data_drift_workload_source = "train"
    config.update_scorer = False
    if args.selection_strategy_diagnostics is not None:
        config.selection_strategy_diagnostics = args.selection_strategy_diagnostics
    if args.test_stop_min_steps is not None:
        config.test_stop_min_steps = args.test_stop_min_steps
    config.ConfirmPath()
    config.eval_output_path = f"./timely_result/{config.mode}_{args.exp_name}_metrics.json"
    planner_path, scorer_path = config.ResolveBestCheckpointPaths(
        str(model_dir), metric=args.checkpoint_metric
    )
    config.agent_checkpoint = planner_path
    config.scorer_path = scorer_path
    if args.scorer_path is not None:
        if not args.scorer_path.is_file():
            raise FileNotFoundError(f"missing scorer: {args.scorer_path}")
        config.scorer_path = str(args.scorer_path.resolve())
    config.auto_config = str(model_dir / "auto_config.json")
    auto_config = json.loads(
        (model_dir / "auto_config.json").read_text(encoding="utf-8")
    )
    for key, value in auto_config.items():
        setattr(config, key, value)
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be at least 1")
        config.maxsteps = args.max_steps
    # Keep the saved model directory read-only during evaluation.
    config.AutoGetParam = False
    config.CpuDeployment(type="test")
    return config, val_iter


def run_test_baselines(dynahint: DynaHint, val_iter: int, test_only: bool = False) -> None:
    dynahint.baselineTrain = 0.0
    dynahint.baselineTest = 0.0
    dynahint.baseline = {}
    dynahint.baseline_info = {}
    dynahint.baseline_info_by_phase = {"train": {}, "test": {}}
    dynahint.baselineTrainByDb = {}
    dynahint._reset_skipped_eval_queries()

    cache_hit = 0
    executed = 0
    while True:
        sql, db, query_id, oneloop = dynahint.queryManager.get2eval()
        latency_timeout, iscollect, _ = ray.get(
            dynahint.planhelper.GetLatency.remote(
                "", sql, db, query_id, collect_cardinality=False, measurement_phase='baseline'
            )
        )
        if latency_timeout[1]:
            dynahint._register_baseline_timeout("test", db, query_id)
            if oneloop:
                break
            continue

        _, _, _, plan_json = ray.get(
            dynahint.planhelper.GetFeature.remote(
                "", sql, False, db, query_id=query_id, need_card_label=False
            )
        )
        planning_time = float(plan_json["Planning Time"]) * 1000.0
        key = f"{db}|{query_id}"
        dynahint.anaManger.recordRuning(
            f"base_{db}_{query_id}",
            f"{latency_timeout[0]:.4f}|{planning_time:.4f}",
        )
        dynahint.anaManger.recordeval(
            db, query_id, latency_timeout[0], planning_time, val_iter=0, phase="test"
        )
        dynahint.anaManger.recordQuery(db, query_id, True)
        dynahint.baselineTest += latency_timeout[0] + planning_time
        dynahint.baseline[key] = float(latency_timeout[0])
        info = {
            "execution_time": float(latency_timeout[0]),
            "planning_time": planning_time,
            "total_time": float(latency_timeout[0] + planning_time),
        }
        dynahint.baseline_info[key] = info
        dynahint.baseline_info_by_phase["test"][key] = info
        if iscollect:
            executed += 1
        else:
            cache_hit += 1
        if oneloop:
            break

    print(
        f"[Baseline test] latency_cache_hit={cache_hit} executed={executed} "
        f"skipped_timeout={len(dynahint.skipped_eval_queries.get('test', set()))}",
        flush=True,
    )
    if not test_only:
        dynahint._run_extra_test_baseline("test_data_drift")
    dynahint.anaManger.recordMetric(0)


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    config, val_iter = configure(args)
    started = time.perf_counter()
    dynahint = DynaHint(config, config.AutoGetParam, istrain=False)
    try:
        run_test_baselines(dynahint, val_iter, test_only=args.test_only)
        dynahint.BuildPlanner(
            train_batch_size=config.planner_batch_size,
            workersNum=config.PlannerworkersNum,
            planner_path=config.agent_checkpoint,
            override_checkpoint_env_config=True,
        )
        dynahint.Validate(
            val_iter,
            isTest=True,
            predict_phase="test",
            metric_phase="test",
            collect_samples=False,
            update_train_state=False,
            eval_phase="test",
        )
        if not args.test_only:
            dynahint.Validate(
                val_iter,
                isTest=True,
                predict_phase="test",
                metric_phase="test_data_drift",
                collect_samples=False,
                update_train_state=False,
                eval_phase="test_data_drift",
            )
        dynahint.anaManger.recordMetric(val_iter)
        dynahint.anaManger.writeout()

        summary = {
            "model_dir": str(args.model_dir.resolve()),
            "model_iter": val_iter,
            "scorer_path": config.scorer_path,
            "databases": list(args.databases),
            "test_only": args.test_only,
            "test_stop_min_steps": config.test_stop_min_steps,
            "max_steps": config.maxsteps,
            "hint_sets_switch": list(config.hint_sets_switch),
            "metrics": dynahint.anaManger.getLatestModelMetrics(),
            "selection_strategy_summary": dynahint.anaManger.getSelectionStrategySummary(),
            "selection_strategy_diagnostics": (
                dynahint.anaManger.selectionStrategyDiagnosticsPath
                if config.selection_strategy_diagnostics else None
            ),
            "skipped_baseline_timeouts": {
                phase: sorted(values)
                for phase, values in dynahint.skipped_eval_queries.items()
            },
            "elapsed_seconds": time.perf_counter() - started,
            "trace_test": dynahint.anaManger.queryTimeTraceTestPath,
            "trace_test_data_drift": dynahint.anaManger.queryTimeTraceTestDataDriftPath,
        }
        output = Path(config.opti_result_path) / f"saved_model_eval_{args.exp_name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        print(f"Wrote evaluation summary: {output.resolve()}", flush=True)
    finally:
        dynahint.Close()


if __name__ == "__main__":
    main()
