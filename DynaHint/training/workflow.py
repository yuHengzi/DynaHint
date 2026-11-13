"""Planner training, scorer coordination, and checkpoint handling."""

import json
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
from types import ModuleType

import numpy as np
import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.checkpoints import get_checkpoint_info

from training.DynaHintEnv import DynaHintEnvTrain
from collection.encoding import write_json_atomic
from collection.datacollector import cached_hint_context
from shared.config import Config
from shared.latency_gate import LatencyGate


@contextmanager
def legacy_planner_checkpoint_modules():
    """Resolve class names embedded in planner checkpoints saved before the rename."""
    aliases = {
        'config': {'Config': Config},
        'OptiQueryEnv': {'OptiQueryEnvTrain': DynaHintEnvTrain},
    }
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        for name, classes in aliases.items():
            module = ModuleType(name)
            for class_name, class_type in classes.items():
                setattr(module, class_name, class_type)
            sys.modules[name] = module
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def checkpoint_record(model_dir, metric, train_iter, value, checkpoint_dir, model_metrics):
    model_dir = os.path.abspath(model_dir)
    return {
        'metric': metric,
        'iteration': int(train_iter),
        'value': float(value),
        'checkpoint_dir': os.path.relpath(checkpoint_dir, model_dir),
        'planner_path': os.path.relpath(os.path.join(checkpoint_dir, 'planner'), model_dir),
        'scorer_path': os.path.relpath(os.path.join(checkpoint_dir, 'scorer.pt'), model_dir),
        'model_metrics': model_metrics,
    }


def absolute_checkpoint_path(model_dir, relative_path):
    return os.path.abspath(os.path.normpath(os.path.join(model_dir, relative_path)))


def cleanup_unreferenced_checkpoints(model_dir, best_checkpoints, old_records):
    referenced = {
        absolute_checkpoint_path(model_dir, record['checkpoint_dir'])
        for record in best_checkpoints.values()
        if isinstance(record, dict) and record.get('checkpoint_dir')
    }
    checkpoint_root = os.path.abspath(os.path.join(model_dir, 'checkpoints'))
    for record in old_records:
        if not isinstance(record, dict) or not record.get('checkpoint_dir'):
            continue
        old_dir = absolute_checkpoint_path(model_dir, record['checkpoint_dir'])
        if old_dir in referenced or not os.path.isdir(old_dir):
            continue
        if os.path.commonpath([checkpoint_root, old_dir]) == checkpoint_root:
            shutil.rmtree(old_dir)


class PlannerRewardCallbacks(DefaultCallbacks):
    def on_episode_start(self, *, episode, **kwargs):
        episode.user_data['planner_reward_source_counts'] = {}
        episode.user_data['scorer_sign_correct'] = []
        episode.user_data['scorer_abs_error'] = []

    def on_episode_step(self, *, episode, **kwargs):
        for agent_id in episode.get_agents():
            if hasattr(episode, 'last_info_for'):
                info = episode.last_info_for(agent_id)
            else:
                info = episode._last_infos.get(agent_id)
            if not isinstance(info, dict):
                continue
            source = info.get('reward_source')
            if source:
                counts = episode.user_data['planner_reward_source_counts']
                counts[source] = counts.get(source, 0) + 1
            if 'scorer_sign_correct' in info:
                episode.user_data['scorer_sign_correct'].append(float(info['scorer_sign_correct']))
            if 'scorer_abs_error' in info:
                episode.user_data['scorer_abs_error'].append(float(info['scorer_abs_error']))

    def on_episode_end(self, *, episode, **kwargs):
        counts = episode.user_data.get('planner_reward_source_counts', {})
        for source, count in counts.items():
            episode.custom_metrics[f'planner_reward_{source}_steps'] = float(count)
        sign_values = episode.user_data.get('scorer_sign_correct', [])
        error_values = episode.user_data.get('scorer_abs_error', [])
        if sign_values:
            episode.custom_metrics['planner_scorer_sign_accuracy'] = float(np.mean(sign_values))
        if error_values:
            episode.custom_metrics['planner_scorer_abs_error'] = float(np.mean(error_values))


def gen_policy(i,genConfig):
    """Build a policy configuration for policy index i."""
    if i % 2 == 0:
        gamma_ = genConfig.planner_config_even['gamma_']
        lr = genConfig.planner_config_even['lr']
    else:
        gamma_ = genConfig.planner_config_odd['gamma_']
        lr = genConfig.planner_config_odd['lr']
    config = PPOConfig.overrides(
        model={
            "custom_model": 'gen_model',
            "vf_share_layers": genConfig.vf_share_layers,
            "fcnet_hiddens": genConfig.fcnet_hiddens,
            "fcnet_activation": genConfig.fcnet_activation,
        },
        entropy_coeff = genConfig.entropy_coeff,
        kl_coeff = genConfig.kl_coeff,
        lambda_ = genConfig.lambda_,
        gamma = gamma_,
        lr = lr)
    return PolicySpec(config=config)
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map an agent ID to its policy ID."""
    pol_id = "policy_{}".format(agent_id)
    return pol_id


class TrainingWorkflow:
    def _update_latency_gate(self, iteration):
        mode = getattr(self.genConfig, 'adaptive_gate_mode', None)
        if mode is None:
            mode = 'baseline' if getattr(self.genConfig, 'adaptive_gate_enable', False) else 'off'
        if mode not in ('off', 'baseline', 'nearest'):
            raise ValueError('adaptive_gate_mode must be off, baseline or nearest')
        if mode == 'off':
            return
        databases = list(self.queryManager.train_database)
        rows = []
        for key, records in self.anaManger.QueryTimeTrace.items():
            db, query = key.split('|', 1)
            if db not in databases:
                continue
            training = [r for r in records if r.get('phase') == 'train']
            base = next((r for r in training if r.get('is_baseline')), None)
            selected = next((r for r in reversed(training) if not r.get('is_baseline')
                             and r.get('val_iter') == iteration), None)
            if base is not None and selected is not None:
                rows.append(dict(
                    phase='train', database=db, query=query,
                    baseline_exec=base['execution_time'], baseline_plan=base['planning_time'],
                    optimized_exec=selected['execution_time'], optimized_plan=selected['planning_time'],
                ))
        if rows:
            gate = LatencyGate(getattr(self.genConfig, 'adaptive_gate_min_records', 6))
            gate.fit(rows, databases, iteration)
            self.adaptive_gate = gate
            if mode == 'nearest':
                if getattr(self.genConfig, 'nearest_tag_model_path', ''):
                    nearest = self._load_nearest_tag()
                    nearest.gate = gate
                else:
                    from shared.nearest_latency_gate import NearestLatencyTAG
                    k, exponent = {'JOB': (3, .5), 'STATS': (3, 1.), 'TPCDS': (1, .5)}[self.genConfig.mode.upper()]
                    nearest = NearestLatencyTAG(self.genConfig.mode, k, exponent)
                    states = getattr(self, '_nearest_train_signatures', {})
                    nearest_rows = []
                    for row in rows:
                        sql = self.queryManager.get2all(row['database'], row['query'])
                        if sql is None:
                            raise ValueError('Missing Train SQL for nearest TAG: '+row['query'])
                        key = (row['database'], row['query'], sql)
                        if key not in states:
                            state, _ = ray.get(self.planhelper.GetNearestTAGInput.remote(
                                sql, row['database'], row['query'], timeout=self.genConfig.max_time_out))
                            states[key] = state
                        nearest_rows.append(dict(row, signature=states[key]))
                    self._nearest_train_signatures = states
                    nearest.fit(nearest_rows, databases, iteration)
                    nearest.gate = gate
                    self.nearest_tag = nearest
                nearest.save(os.path.join(self.genConfig.model_dir, 'nearest_tag.json'))
            write_json_atomic(os.path.join(self.genConfig.model_dir, 'adaptive_gate.json'), gate.to_dict())
            print('AdaptiveGate calibrated | iteration={} | train_records={} | global_ms={}'.format(
                iteration, len(rows), gate.global_threshold), flush=True)

    def _checkpoint_metric_values(self, wrl_test_withPlanning, wrl_train_withPlanning,
                                  gmrl_test, gmrl_train, wrl_test, wrl_train):
        metric = getattr(self.genConfig, 'checkpoint_metric', 'train_wrl')
        if metric == 'train_wrl':
            return wrl_train, gmrl_train
        if metric == 'train_wrl_withPlanning':
            return wrl_train_withPlanning, gmrl_train
        if metric == 'test_wrl_withPlanning':
            return wrl_test_withPlanning, gmrl_test
        return wrl_test, gmrl_test

    def _should_save_model(self, wrl_test_withPlanning, wrl_train_withPlanning,
                           gmrl_test, gmrl_train, wrl_test, wrl_train):
        # Use checkpoint_metric to prioritize checkpoint selection; default to Train execution-only WRL.
        metric = getattr(self.genConfig, 'checkpoint_metric', 'train_wrl')
        checkpoint_wrl, checkpoint_gmrl = self._checkpoint_metric_values(
            wrl_test_withPlanning,
            wrl_train_withPlanning,
            gmrl_test,
            gmrl_train,
            wrl_test,
            wrl_train,
        )
        if metric not in ('train_wrl', 'train_wrl_withPlanning', 'test_wrl_withPlanning'):
            if checkpoint_gmrl < self.best_model_gmrl:
                return True
            if abs(checkpoint_gmrl - self.best_model_gmrl) <= 1e-9 and checkpoint_wrl < self.best_model_wrl:
                return True
            return False
        if checkpoint_wrl < self.best_model_wrl:
            return True
        if abs(checkpoint_wrl - self.best_model_wrl) <= 1e-9 and checkpoint_gmrl < self.best_model_gmrl:
            return True
        return False

    def _best_train_metric_improvements(self, wrl, gmrl):
        improved = []
        first_checkpoint = not self.best_checkpoints and not getattr(self, 'has_legacy_checkpoint', True)
        if np.isfinite(wrl) and (first_checkpoint or wrl <= self.best_model_wrl):
            improved.append('train_wrl')
        if np.isfinite(gmrl) and (first_checkpoint or gmrl <= self.best_model_gmrl):
            improved.append('train_gmrl')
        return improved

    def _should_validate_test_phases(self, improved_metrics, train_iter):
        if not getattr(self.genConfig, 'validate_test_only_on_train_improvement', False):
            return True
        full_validation_after_iter = int(
            getattr(self.genConfig, 'full_validation_after_iter', 0) or 0
        )
        if full_validation_after_iter > 0 and train_iter >= full_validation_after_iter:
            return True
        return bool(improved_metrics)

    def _checkpoint_record(self, metric, train_iter, value, checkpoint_dir, model_metrics):
        if hasattr(self.genConfig, '_normalize_model_metrics'):
            model_metrics = self.genConfig._normalize_model_metrics(model_metrics)
        return checkpoint_record(
            self.genConfig.model_dir, metric, train_iter, value, checkpoint_dir, model_metrics,
        )

    def _absolute_checkpoint_path(self, relative_path):
        return absolute_checkpoint_path(self.genConfig.model_dir, relative_path)

    def _cleanup_unreferenced_checkpoints(self, old_records):
        cleanup_unreferenced_checkpoints(
            self.genConfig.model_dir, self.best_checkpoints, old_records,
        )

    def _save_best_train_checkpoint(self, train_iter, improved_metrics, wrl, gmrl, model_metrics):
        checkpoint_dir = os.path.join(self.genConfig.model_dir, 'checkpoints', f'iter_{train_iter:06d}')
        planner_path = os.path.join(checkpoint_dir, 'planner')
        scorer_path = os.path.join(checkpoint_dir, 'scorer.pt')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.SavePlanner(checkpoint_path=planner_path)
        ray.get(self.scorer_actor.SaveModel.remote(model_path=scorer_path))
        if getattr(self.genConfig, 'adaptive_gate_enable', False) and hasattr(self, 'adaptive_gate'):
            write_json_atomic(os.path.join(checkpoint_dir, 'adaptive_gate.json'), self.adaptive_gate.to_dict())
            if getattr(self.genConfig, 'adaptive_gate_mode', '') == 'nearest' and hasattr(self, 'nearest_tag'):
                self.nearest_tag.save(os.path.join(checkpoint_dir, 'nearest_tag.json'))

        old_records = [deepcopy(self.best_checkpoints.get(metric)) for metric in improved_metrics]
        values = {
            'train_wrl': wrl,
            'train_gmrl': gmrl,
        }
        for metric in improved_metrics:
            record = self._checkpoint_record(
                metric,
                train_iter,
                values[metric],
                checkpoint_dir,
                model_metrics,
            )
            self.best_checkpoints[metric] = record

        if 'train_wrl' in improved_metrics:
            self.best_model_wrl = float(wrl)
        if 'train_gmrl' in improved_metrics:
            self.best_model_gmrl = float(gmrl)

        # Keep legacy resume paths pointed at the WRL best model by default.
        primary = self.best_checkpoints.get('train_wrl') or self.best_checkpoints.get('train_gmrl')
        if primary:
            self.genConfig.agent_checkpoint = self._absolute_checkpoint_path(primary['planner_path'])
            self.genConfig.scorer_path = self._absolute_checkpoint_path(primary['scorer_path'])
        self.genConfig.best_checkpoints = deepcopy(self.best_checkpoints)
        self._cleanup_unreferenced_checkpoints(old_records)
        return checkpoint_dir

    def _log_train_db_best_plan_wrl(self, totals_by_db, step):
        # Write per-database best-plan snapshots with the same semantics as global Train best-plan WRL.
        for db, best_exec_total in totals_by_db.items():
            baseline_total = self.baselineTrainByDb.get(db, 0.0)
            if baseline_total <= 0:
                continue
            self.writer.add_scalar(
                f'Others/BestPlanWRL_withPlanning/{db}',
                best_exec_total / baseline_total,
                step,
            )

    @staticmethod
    def _nested_get(mapping, path, default=None):
        cur = mapping
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    def _log_planner_reward_curves(self, result, train_iter):
        # Normalize RLlib reward-field layouts before logging under Others.
        reward_mean = (
            self._nested_get(result, ('episode_reward_mean',))
            if self._nested_get(result, ('episode_reward_mean',)) is not None
            else self._nested_get(result, ('sampler_results', 'episode_reward_mean'))
        )
        if reward_mean is None:
            reward_mean = self._nested_get(result, ('env_runners', 'episode_reward_mean'))
        reward_min = (
            self._nested_get(result, ('episode_reward_min',))
            if self._nested_get(result, ('episode_reward_min',)) is not None
            else self._nested_get(result, ('sampler_results', 'episode_reward_min'))
        )
        if reward_min is None:
            reward_min = self._nested_get(result, ('env_runners', 'episode_reward_min'))
        reward_max = (
            self._nested_get(result, ('episode_reward_max',))
            if self._nested_get(result, ('episode_reward_max',)) is not None
            else self._nested_get(result, ('sampler_results', 'episode_reward_max'))
        )
        if reward_max is None:
            reward_max = self._nested_get(result, ('env_runners', 'episode_reward_max'))
        episode_len_mean = (
            self._nested_get(result, ('episode_len_mean',))
            if self._nested_get(result, ('episode_len_mean',)) is not None
            else self._nested_get(result, ('sampler_results', 'episode_len_mean'))
        )
        if episode_len_mean is None:
            episode_len_mean = self._nested_get(result, ('env_runners', 'episode_len_mean'))
        if reward_mean is not None:
            self.writer.add_scalar('Others/PlannerRewardMean', reward_mean, train_iter)
        if reward_min is not None:
            self.writer.add_scalar('Others/PlannerRewardMin', reward_min, train_iter)
        if reward_max is not None:
            self.writer.add_scalar('Others/PlannerRewardMax', reward_max, train_iter)
        if episode_len_mean is not None:
            self.writer.add_scalar('Others/PlannerStepLenMean', episode_len_mean, train_iter)
        custom_metrics = (
            self._nested_get(result, ('custom_metrics',))
            or self._nested_get(result, ('sampler_results', 'custom_metrics'))
            or self._nested_get(result, ('env_runners', 'custom_metrics'))
            or {}
        )
        metric_tags = {
            'planner_reward_real_latency_steps_mean': 'PlannerRewardRealLatencySteps',
            'planner_reward_scorer_surrogate_steps_mean': 'PlannerRewardSurrogateSteps',
            'planner_reward_duplicate_steps_mean': 'PlannerRewardDuplicateSteps',
            'planner_scorer_sign_accuracy_mean': 'PlannerScorerSignAccuracy',
            'planner_scorer_abs_error_mean': 'PlannerScorerAbsError',
        }
        for metric_name, tag_name in metric_tags.items():
            if metric_name in custom_metrics:
                self.writer.add_scalar(f'Others/{tag_name}', custom_metrics[metric_name], train_iter)

    def _flush_scorer_metrics(self):
        # Write scorer loss and reward curves into one TensorBoard stream from the main process.
        if not self.learner:
            return
        metric_batches = ray.get(self.learner.pop_train_metrics.remote())
        for history in metric_batches:
            if not history:
                continue
            global_step_base = int(history.get('global_step_base', 0))
            for epoch, value in enumerate(history.get('total_loss', [])):
                self.writer.add_scalar('Others/Scorer/Total_Loss', value, global_step_base + epoch)
            for epoch, value in enumerate(history.get('score_loss', [])):
                self.writer.add_scalar('Others/Scorer/Score_Loss', value, global_step_base + epoch)
            for epoch, value in enumerate(history.get('card_loss', [])):
                self.writer.add_scalar('Others/Scorer/Card_Loss', value, global_step_base + epoch)
            for epoch, value in enumerate(history.get('risk_loss', [])):
                self.writer.add_scalar('Others/Scorer/Risk_Loss', value, global_step_base + epoch)
            for epoch, value in enumerate(history.get('val_reward', [])):
                self.writer.add_scalar('Others/Scorer/Reward_Val', value, global_step_base + epoch)
            for epoch, value in enumerate(history.get('val_mse', [])):
                self.writer.add_scalar('Others/Scorer/MSE_Val', value, global_step_base + epoch)
            for epoch, value in enumerate(history.get('test_reward', [])):
                self.writer.add_scalar('Others/Scorer/Reward_Test', value, global_step_base + epoch)
            for epoch, value in enumerate(history.get('test_mse', [])):
                self.writer.add_scalar('Others/Scorer/MSE_Test', value, global_step_base + epoch)
            if history.get('elapsed_seconds') is not None:
                self.writer.add_scalar(
                    'Timing/Training/ScorerSeconds',
                    float(history['elapsed_seconds']),
                    int(history.get('train_round', 0)) + 1,
                )

    def BuildPlanner(self, train_batch_size, workersNum, planner_path = None,
                     override_checkpoint_env_config = False):
        """Build the planner with the requested batch size, workers and optional checkpoint."""
        if planner_path is None:
            policies = {"policy_{}".format(i): gen_policy(i,self.genConfig) for i in range(self.genConfig.num_policies)}
            planner_config = (
                PPOConfig()
                .environment(DynaHintEnvTrain, env_config = {'genConfig':self.genConfig}, disable_env_checking = True)
                .training(
                    train_batch_size=train_batch_size,
                    num_sgd_iter=self.genConfig.planner_num_sgd_iter,
                )
                .multi_agent(policies = policies, policy_mapping_fn = policy_mapping_fn)
                .callbacks(PlannerRewardCallbacks)
                .resources(num_gpus=1, num_gpus_per_learner_worker=1)
                .rollouts(num_rollout_workers = workersNum, num_envs_per_worker = 1)#,sample_async=True)
                # .rollouts(num_envs_per_worker = 1)#,sample_async=True)
            )
            planner_config['seed'] = self.genConfig.seed
            self.planner           = planner_config.build(logger_creator = None)
        else:
            if self.istrain:
                self.LoadPlanner(planner_path,load_tmp=True)
            else:
                self.LoadPlanner(
                    planner_path,
                    override_env_config=override_checkpoint_env_config,
                )
                            
        # Optional resource-usage diagnostics.
        # resources = ray.available_resources()
        # total_resources = ray.cluster_resources()
        # Optional resource-usage heading.
        # Optional CPU-usage diagnostic.
        # Optional total-CPU diagnostic.

    def InitScorer(self, initialPro = 0.25):
        """Initialize scorer training from the initial exploration fraction."""
        estotal,_ = self.Validate(0, Explore = True, initialPro = initialPro)
        train_db_estotal, _ = ray.get(self.bpm.get_best_executed_by_db.remote())
        self.writer.add_scalar('Others/Accumulated Excecuted Plans Num', 0, 0)
        self.writer.add_scalar('Others/Train Set Best Plan WRL_withPlanning', estotal / self.baselineTrain, 0)
        self._log_train_db_best_plan_wrl(train_db_estotal, 0)
        runner_ref = self.learner.Runing.remote()
        time.sleep(5)
        ray.get(self.bpm.update_schedule.remote(True))
        ray.get(runner_ref)

    def Run(self, startIter = 0, totalIter = 300, valFreq = 5):
        """Train the planner and scorer over the requested iterations and validation frequency."""
        ray.get(self.bpm.update_schedule.remote(False))
        runner_ref = self.learner.Runing.remote()   # Run the plan-selection learner.
        startVal = self.genConfig.start_val
        wrl_test_withPlanning, wrl_train_withPlanning = 1.0, 1.0
        wrl_test, wrl_train, gmrl_test, gmrl_train = 1.0, 1.0, 1.0, 1.0
        for trainIter in range(startIter, totalIter):
            iteration_started = time.perf_counter()
            timing = {
                'LatencySyncSeconds': 0.0,
                'PlannerSeconds': 0.0,
                'LearnerBarrierSeconds': 0.0,
                'ValidationSeconds': 0.0,
                'CheckpointSeconds': 0.0,
            }
            model_metrics = None
            did_validate = False
            improved_metrics = []
            print(f'---------- TrainIter {trainIter+1} ----------')
            phase_started = time.perf_counter()
            latencybuffer = ray.get(self.planhelper.GetPGLatencyBuffer.remote())
            ray.get(self.bpm.update_latencyBuffer.remote(latencybuffer))
            timing['LatencySyncSeconds'] = time.perf_counter() - phase_started
            phase_started = time.perf_counter()
            result = self.planner.train()    # Train the planner.
            ray.get(self.bpm.advance_planner_iteration.remote())
            timing['PlannerSeconds'] = time.perf_counter() - phase_started
            self._log_planner_reward_curves(result, trainIter + 1)  # Log planner reward curves to track PPO learning.
            save_info = True
            if trainIter <= 4 or (trainIter + 1) % valFreq == 0 or trainIter == totalIter - 1:
                phase_started = time.perf_counter()
                ray.get(self.bpm.update_schedule.remote(True))
                accumulatedPlans = ray.get(runner_ref)
                timing['LearnerBarrierSeconds'] = time.perf_counter() - phase_started
                self._flush_scorer_metrics()
                self.writer.add_scalar('Others/Accumulated Excecuted Plans Num', accumulatedPlans, trainIter + 1)
                if trainIter == totalIter - 1 or ((trainIter + 1) % valFreq == 0 and trainIter+1 >= startVal):
                    did_validate = True
                    validation_all_started = time.perf_counter()
                    print('---------- Start Validate ----------')
                    validation_times = {}
                    validation_started = time.perf_counter()
                    estotal, best_steps = self.Validate(trainIter + 1)  # Validate the model on the training workload.
                    validation_times['Train'] = time.perf_counter() - validation_started
                    self._update_latency_gate(trainIter + 1)
                    train_db_estotal, _ = ray.get(self.bpm.get_best_executed_by_db.remote())
                    self.writer.add_scalar('Others/Train Set Best Plan WRL_withPlanning', estotal / self.baselineTrain, trainIter + 1)
                    self._log_train_db_best_plan_wrl(train_db_estotal, trainIter + 1)
                    wrl_train_withPlanning, gmrl_train, wrl_train = self.anaManger.getTrainMetrics()
                    improved_metrics = self._best_train_metric_improvements(wrl_train, gmrl_train)
                    run_test_validation = self._should_validate_test_phases(
                        improved_metrics, trainIter + 1
                    )
                    if run_test_validation:
                        validation_started = time.perf_counter()
                        self.Validate(trainIter + 1, isTest=True)
                        validation_times['Test'] = time.perf_counter() - validation_started
                        validation_started = time.perf_counter()
                        if self._validate_mix_plus_data_drift_if_enabled(trainIter + 1):
                            validation_times['TestDataDrift'] = time.perf_counter() - validation_started
                    else:
                        self.anaManger.getTrainMetrics(trainIter + 1, record=True)
                        print(
                            '---------- Skip Test/TestDataDrift: Train WRL/GMRL did not improve ----------',
                            flush=True,
                        )
                    self._record_validation_times(trainIter + 1, validation_times)
                    if run_test_validation:
                        wrl_test_withPlanning, wrl_train_withPlanning, gmrl_test, gmrl_train, wrl_test, wrl_train = self.anaManger.recordMetric(trainIter + 1)
                        model_metrics = self.anaManger.getLatestModelMetrics()
                    self.anaManger.writeout()
                    timing['ValidationSeconds'] = time.perf_counter() - validation_all_started
                    print('---------- Validate End ----------')
                if improved_metrics:
                    phase_started = time.perf_counter()
                    self._save_best_train_checkpoint(
                        trainIter + 1,
                        improved_metrics,
                        wrl_train,
                        gmrl_train,
                        model_metrics,
                    )
                    timing['CheckpointSeconds'] += time.perf_counter() - phase_started
                    first_start_time,end_time = self.genConfig.SaveTrainInfo(
                        trainIter+1,
                        start_time=self.start_time,
                        first_save=(trainIter==startIter),
                        best_checkpoints=self.best_checkpoints,
                    )
                    save_info = False
                    print(f'---------- Best checkpoints updated: {", ".join(improved_metrics)} ----------')
                ray.get(self.bpm.update_schedule.remote(False))
                runner_ref = self.learner.Runing.remote()
            tmp_checkpoint_freq = max(int(getattr(self.genConfig, 'tmp_checkpoint_freq', 20)), 1)
            if (trainIter + 1) % tmp_checkpoint_freq == 0 or trainIter == totalIter - 1:
                phase_started = time.perf_counter()
                self.SavePlanner(save_tmp=True)
                timing['CheckpointSeconds'] += time.perf_counter() - phase_started
            if save_info:
                first_start_time,end_time = self.genConfig.SaveTrainInfo(
                    trainIter+1,
                    start_time=self.start_time,
                    first_save=(trainIter==startIter),
                    best_checkpoints=self.best_checkpoints,
                )
            timing['IterationSeconds'] = time.perf_counter() - iteration_started
            for timing_name, elapsed in timing.items():
                self.writer.add_scalar(f'Timing/Training/{timing_name}', elapsed, trainIter + 1)
        print(f'---------- {totalIter} training iterations completed; first start: {first_start_time}; session start: {self.start_time.strftime("%Y/%m/%d %H.%M.%S")}; end: {end_time.strftime("%Y/%m/%d %H.%M.%S")} ----------')
        print(f'---------- Session training duration: {self.genConfig.GetTotalTime()[1]}; total training duration: {self.genConfig.GetTotalTime()[0]} ----------')
        ray.get(self.bpm.update_schedule.remote(True))
        self._flush_scorer_metrics()

    def LoadExperiencePool(self, buffer_path, databases=None, need_card_label=True,
                           update_best_plan=True):
        """Load training experience from a latency buffer."""
        print('Loading experience pool.')
        database_filter = None if databases is None else set(databases)
        warmup = getattr(self.genConfig, 'latency_warmup_runs', 0)
        runs = getattr(self.genConfig, 'latency_measure_runs', 1)
        repeat = ('experience' in getattr(self.genConfig, 'latency_repeat_phases', [])
                  and (warmup != 0 or runs != 1))
        measurement_path = buffer_path + '.measurements.jsonl'
        if repeat and not os.path.exists(measurement_path):
            print('No records for the current measurement protocol; skipping the legacy single-execution experience pool.', flush=True)
            return 0
        if os.path.exists(buffer_path) or (repeat and os.path.exists(measurement_path)):
            if repeat:
                expected = [warmup, runs, getattr(self.genConfig, 'latency_aggregate', 'median')]
                with open(measurement_path) as handle:
                    records = [json.loads(line) for line in handle]
                lines = [json.dumps([r['database'], r['query'], r.get('cardinality', -1),
                                     r['hint'], r['latency'], r.get('step', 0.)])
                         for r in records if r['recipe'] == expected
                         and r['phase'] in ('baseline', 'experience')]
            else:
                with open(buffer_path) as handle:
                    lines = handle.readlines()
            experience_num = 0
            for line in lines:
                data = json.loads(line)
                db = data[0]
                if database_filter is not None and db not in database_filter:
                    continue
                query_id = data[1]
                cardinality = data[2] if need_card_label else -1
                opti_hint = data[3]
                latency = data[4]
                step = data[5]
                sql = self.queryManager.get2all(db,query_id)
                if sql is not None:
                    feature_dict,_,_,_ = ray.get(self.planhelper.GetFeature.remote(
                        opti_hint, sql, True, db, query_id=query_id,
                        need_card_label=need_card_label,
                    ))
                    feature_dict['steps'], feature_dict['set_hint_bits'] = cached_hint_context(
                        opti_hint, step, self.genConfig
                    )
                    if self.learner:
                        baseline_plan = self.baseline_info.get(db+'|'+query_id, {}).get('planning_time', 0.0) if hasattr(self, 'baseline_info') else 0.0
                        self.learner.CollectSample.remote(
                            db,
                            query_id,
                            feature_dict,
                            latency[0],
                            cardinality,
                            latency[1],
                            planning_time_cum=0.0 if opti_hint != '' else baseline_plan,
                            baseline_planning_time=baseline_plan,
                            is_baseline=not bool(opti_hint),
                        )
                        if update_best_plan:
                            bestfeature, bestexec = ray.get(self.bpm.get_best_executed_plan.remote(db,query_id))
                            Advbybest = (bestexec - latency[0]) / bestexec
                            if Advbybest >= self.genConfig.splitpoint[-1]:
                                self.bpm.update_best_executed_plan.remote(db,query_id, deepcopy(feature_dict),latency[0])
                    experience_num += 1
                    if experience_num % 1000 == 0:
                        print(f'Loaded {experience_num} experience records...', flush=True)
            print(f'Loaded {experience_num} experience records; experience pool loaded......')
            return experience_num
        else:
            raise FileNotFoundError(f"经验池文件不存在，请检查路径是否正确: {buffer_path}")

    def SavePlanner(self,save_tmp = False, checkpoint_path=None):
        """Save the planner; temporary checkpoints support recovery after interruptions."""
        path = checkpoint_path or (self.tmp_planner_path if save_tmp else self.genConfig.agent_checkpoint)
        os.makedirs(path, exist_ok=True)
        self.planner.save_checkpoint(path)
        if not self.genConfig.AutoGetParam:
            config_data = {
                "types":self.genConfig.types,
                "columns":self.genConfig.columns,
                "tablenum":self.genConfig.tablenum,
                "opsnum":self.genConfig.opsnum,
                "heightsize":self.genConfig.heightsize,
                "maxnode":self.genConfig.maxnode,
                "maxjoins":self.genConfig.maxjoins,
                "max_logical_edges":self.genConfig.max_logical_edges,
                "filtmaxnum":self.genConfig.filtmaxnum,
                "maxpos":self.genConfig.maxpos,
                "num_node_feature":self.genConfig.num_node_feature,
                "db_hist_dim":self.genConfig.db_hist_dim,
                "db_bitmap_dim":self.genConfig.db_bitmap_dim,
                "query_feat_dim":self.genConfig.query_feat_dim,
                "use_db_bitmap_features": bool(getattr(self.genConfig, "use_db_bitmap_features", True)),
                "AutoGetParam":0
            }
            write_json_atomic(self.genConfig.auto_config, config_data, ensure_ascii=False, indent=4)

    def LoadPlanner(self,planner_path,load_tmp = False, override_env_config = False):
        """Load the planner from a saved or temporary checkpoint."""
        checkpoint_path = self.tmp_planner_path if load_tmp else planner_path
        with legacy_planner_checkpoint_modules():
            if not override_env_config:
                self.planner = Algorithm.from_checkpoint(checkpoint_path)
                return

            # Preserve the saved policy while replacing its training DB environment.
            checkpoint_info = get_checkpoint_info(checkpoint_path)
            state = Algorithm._checkpoint_info_to_algorithm_state(checkpoint_info)
            planner_config = state.get('config')
            if planner_config is None:
                raise ValueError('Planner checkpoint does not contain a config.')
            if isinstance(planner_config, dict):
                env_config = dict(planner_config.get('env_config') or {})
                env_config['genConfig'] = self.genConfig
                planner_config['env_config'] = env_config
            else:
                env_config = dict(getattr(planner_config, 'env_config', {}) or {})
                env_config['genConfig'] = self.genConfig
                planner_config.env_config = env_config
            self.planner = Algorithm.from_state(state)
