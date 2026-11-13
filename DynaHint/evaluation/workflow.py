"""Baseline, validation, and inference workflows."""

import json
import math
import os
import random
import shutil
import time
from copy import deepcopy
from datetime import datetime

import numpy as np
import pandas as pd
import ray
from shared.latency_gate import LatencyGate


def collect_candidates(env, planner, num_policies, maxsteps, obs, info, breakdown):
    candidates = [(
        info['hint'],
        deepcopy(obs[0]),
        {'planning_time_cum_ms': float(info.get('planning_time_cum_ms', 0.0))},
    )]
    breakdown['candidate_count'] = len(candidates)
    if not info['can_optimize']:
        return candidates

    done = False
    while not done:
        action_start = time.time()
        actions = {
            i: planner.compute_single_action(
                obs[i], policy_id=f'policy_{i}', explore=False,
            )
            for i in range(num_policies)
        }
        breakdown['action_ms'] += (time.time() - action_start) * 1000.0

        env_step_start = time.time()
        obs, _, terminated, _, info_all = env.step(actions)
        breakdown['env_step_ms'] += (time.time() - env_step_start) * 1000.0

        copy_start = time.time()
        for agent_id, candidate_info in info_all.items():
            if candidate_info.get('record_candidate', True):
                candidates.append((
                    candidate_info['hint'],
                    deepcopy(obs[agent_id]),
                    {
                        'planning_time_cum_ms': float(
                            candidate_info.get('planning_time_cum_ms', 0.0)
                        ),
                    },
                ))
            if candidate_info.get('stop_taken', False):
                breakdown['stop_taken'] = True
                breakdown['stop_step'] = int(
                    candidate_info.get('stop_step', maxsteps)
                )
            terminate_reason = candidate_info.get('terminate_reason', '')
            if terminate_reason:
                breakdown['terminate_reason'] = terminate_reason
        breakdown['obs_copy_ms'] += (time.time() - copy_start) * 1000.0
        breakdown['loop_count'] += 1
        breakdown['candidate_count'] = len(candidates)
        done = bool(terminated['__all__'])

    if not breakdown['terminate_reason']:
        breakdown['terminate_reason'] = 'maxsteps'
    return candidates


class EvaluationWorkflow:
    def _record_validation_times(self, val_iter, phase_times):
        normalized = {
            phase: max(float(phase_times.get(phase, 0.0)), 0.0)
            for phase in ('Train', 'Test', 'TestDataDrift')
        }
        totals = None
        for phase in phase_times:
            elapsed_seconds = normalized[phase]
            totals = self.genConfig.RecordValidationTime(phase, elapsed_seconds)
            self.writer.add_scalar(f'Timing/Validation/{phase}Seconds', elapsed_seconds, val_iter)
            self.writer.add_scalar(
                f'Timing/Validation/Cumulative{phase}Seconds',
                totals[phase],
                val_iter,
            )
            print(
                'ValidationTime | iter={} | phase={} | elapsed_s={:.3f} | cumulative_total_s={:.3f}'.format(
                    val_iter,
                    phase,
                    elapsed_seconds,
                    totals['Total'],
                )
            )
        if totals is not None:
            self.writer.add_scalar(
                'Timing/Validation/CumulativeTotalSeconds',
                totals['Total'],
                val_iter,
            )
        for phase, elapsed_seconds in normalized.items():
            self.writer.add_scalar(f'time/{phase}', elapsed_seconds, val_iter)
        self.writer.add_scalar('time/Total', sum(normalized.values()), val_iter)

    def _validate_mix_plus_data_drift_if_enabled(self, val_iter):
        if self.genConfig.train_mode != 'mix+':
            return False
        if 'test_data_drift' not in getattr(self.queryManager, 'extraEvalSets', {}):
            return False
        self.Validate(
            val_iter,
            isTest=True,
            predict_phase='test',
            metric_phase='test_data_drift',
            eval_phase='test_data_drift',
            collect_samples=False,
            update_train_state=False,
        )
        return True

    def _get_baseline_info(self, phase, db, query_id):
        db_and_queryid = db + '|' + query_id
        phase_info = getattr(self, 'baseline_info_by_phase', {})
        if phase in phase_info and db_and_queryid in phase_info[phase]:
            return phase_info[phase][db_and_queryid]
        return self.baseline_info.get(db_and_queryid, {
            'execution_time': self.baseline.get(db_and_queryid, self.genConfig.max_time_out),
            'planning_time': 0.0,
            'total_time': self.baseline.get(db_and_queryid, self.genConfig.max_time_out),
        })

    def _load_nearest_tag(self):
        from shared.nearest_latency_gate import NearestLatencyTAG
        if not hasattr(self, 'nearest_tag'):
            path = getattr(self.genConfig, 'nearest_tag_model_path', '') or os.path.join(
                os.path.dirname(self.genConfig.scorer_path), 'nearest_tag.json')
            if not os.path.isfile(path):
                raise FileNotFoundError('nearest TAG requires a trained model.json: '+path)
            model = NearestLatencyTAG.load(path)
            if model.dataset != self.genConfig.mode.upper():
                raise ValueError('Nearest TAG dataset does not match the experiment')
            databases = getattr(getattr(self, 'queryManager', None), 'train_database', None)
            if databases and not {r['database'] for r in model.samples}.issubset(set(databases)):
                raise ValueError('Nearest TAG contains databases outside this experiment training set')
            self.nearest_tag = model
        return self.nearest_tag

    def _latency_gate_skip_execution(self, is_test, explore, phase, db, query_id, sql=None):
        self._tag_initial_plan_feature = None
        mode = getattr(self.genConfig, 'adaptive_gate_mode', None)
        if mode is None:
            mode = 'baseline' if getattr(self.genConfig, 'adaptive_gate_enable', False) else 'off'
        if mode not in ('off', 'baseline', 'nearest'):
            raise ValueError('adaptive_gate_mode must be off, baseline or nearest')
        if not is_test or explore or phase not in ('test', 'test_data_drift') or mode == 'off':
            return None
        started = time.perf_counter()
        if mode == 'nearest':
            model = self._load_nearest_tag()
            if sql is None:
                raise ValueError('Nearest TAG requires query SQL')
            try:
                signature, prepared = ray.get(self.planhelper.GetNearestTAGInput.remote(
                    sql, db, query_id, timeout=self._validation_cost_plan_timeout(phase, db, query_id)))
            except Exception as exc:
                if 'statement timeout' not in str(exc).lower():
                    raise
                return None  # Existing reset/timeout path records the failed initial plan.
            self._tag_initial_plan_feature = prepared
            decision = model.predict(query_id, signature)
            if decision['optimize']:
                return None
            key = db+'|'+query_id
            known = (key in getattr(self, 'baseline_info_by_phase', {}).get(phase, {})
                     or key in getattr(self, 'baseline_info', {}) or key in getattr(self, 'baseline', {}))
            if not known:
                return None  # An evaluation skip needs an observed result for metric accounting.
            baseline = self._get_baseline_info(phase, db, query_id)
            return dict(optimal_feature=None, latency_timeout=[baseline['execution_time'], False],
                        planning_time=time.perf_counter()-started)
        if not hasattr(self, 'adaptive_gate'):
            path = os.path.join(os.path.dirname(self.genConfig.scorer_path), 'adaptive_gate.json')
            if os.path.isfile(path):
                with open(path) as handle:
                    self.adaptive_gate = LatencyGate.from_dict(json.load(handle))
        gate = getattr(self, 'adaptive_gate', None)
        key = db + '|' + query_id
        known = (key in getattr(self, 'baseline_info_by_phase', {}).get(phase, {})
                 or key in getattr(self, 'baseline_info', {})
                 or key in getattr(self, 'baseline', {}))
        if gate is None or not known:
            return None
        baseline = self._get_baseline_info(phase, db, query_id)
        if gate.allows(query_id, baseline.get('execution_time')):
            return None
        # Cached evaluation keeps skipped queries in every aggregate metric.
        return dict(optimal_feature=None,
                    latency_timeout=[baseline['execution_time'], False],
                    planning_time=time.perf_counter() - started)

    def _reset_skipped_eval_queries(self):
        self.skipped_eval_queries = {}

    def _register_baseline_timeout(self, phase, db, query_id):
        phase = str(phase)
        db_and_queryid = db + '|' + query_id
        self.skipped_eval_queries.setdefault(phase, set()).add(db_and_queryid)
        # print(f'[Baseline {phase}] skip timeout query: {db_and_queryid}', flush=True)

    def _is_skipped_eval_query(self, phase, db, query_id):
        return db + '|' + query_id in self.skipped_eval_queries.get(str(phase), set())

    def _print_cache_work(self, label, count, detail=''):
        message = f"[{label}] {count}"
        if detail:
            message += f" | {detail}"
        print(message, flush=True)

    def _load_cardinality_cache_keys(self):
        keys = set()
        cache_path = getattr(self.genConfig, 'cardinality_cache_path', None)
        if not cache_path or not os.path.exists(cache_path):
            return keys
        with open(cache_path, 'r') as cache_file:
            for line in cache_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if float(record.get('cardinality', -1)) >= 0:
                        keys.add('|'.join([str(record['database']), str(record['query_id'])]))
                except Exception:
                    continue
        return keys

    def _iter_query_phase(self):
        query_set = self.queryManager.validateSet
        for db_and_queryid in query_set.index:
            db, query_id = db_and_queryid.split('|')
            yield query_set.loc[db_and_queryid, 'sql'], db, query_id

    def _warmup_cardinality_phase(self):
        cache_keys = self._load_cardinality_cache_keys()
        cache_hit = 0
        miss = 0
        for sql, db, query_id in self._iter_query_phase():
            cache_key = '|'.join([str(db), str(query_id)])
            if cache_key in cache_keys:
                cache_hit += 1
                continue
            miss += 1
            cardinality = ray.get(self.planhelper.GetCardinality.remote(db, query_id, sql))
            cache_keys.add(cache_key)
            # self._print_cache_work(
            #     f"Cardinality {phase}",
            #     f"miss={miss}",
            #     f"db={db} query={query_id} card={cardinality}",
            # )
        print(f"[Cardinality train] cache_hit={cache_hit} miss={miss}", flush=True)

    def _run_extra_test_baseline(self, phase):
        phase_total = 0.0
        self.baseline_info_by_phase.setdefault(phase, {})
        cache_hit = 0
        executed = 0
        while True:
            sql, db, query_id, oneloop = self.queryManager.get2eval(phase=phase)
            latency_timeout,iscollect,_ = ray.get(
                self.planhelper.GetLatency.remote('', sql, db, query_id, collect_cardinality=False, measurement_phase='baseline')
            )
            if latency_timeout[1]:
                self._register_baseline_timeout(phase, db, query_id)
                if oneloop:
                    break
                continue
            _, _,_, plan_json = ray.get(
                self.planhelper.GetFeature.remote('', sql, False, db, query_id=query_id, need_card_label=False)
            )
            planning_time = plan_json['Planning Time'] * 1000
            key_out = '_'.join(['base', phase, db, query_id])
            value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(planning_time)])
            self.anaManger.recordRuning(key_out,value_out)
            self.anaManger.recordExtraEval(
                phase,
                db,
                query_id,
                latency_timeout[0],
                planning_time,
                val_iter=0,
                is_baseline=True,
            )
            self.baseline_info_by_phase[phase][db+'|'+query_id] = {
                'execution_time': float(latency_timeout[0]),
                'planning_time': float(planning_time),
                'total_time': float(latency_timeout[0]),
            }
            phase_total += latency_timeout[0]
            if iscollect:
                executed += 1
                self._print_cache_work(
                    f"Baseline {phase}",
                    f"executed={executed}",
                    f"db={db} query={query_id} exec_ms={latency_timeout[0]:.3f} plan_ms={planning_time:.3f}",
                )
            else:
                cache_hit += 1
            if oneloop:
                break
        print(f"[Baseline {phase}] latency_cache_hit={cache_hit} executed={executed}", flush=True)
        skipped = len(self.skipped_eval_queries.get(str(phase), {}))
        print(f'[Baseline {phase}] skipped_timeout={skipped}', flush=True)
        print('Baseline_{}:{:.4f}'.format(phase, phase_total / 1000))

    def RunBaseline(self, startIter = 0, train_only=False):
        """Collect baselines for training and testing queries, optionally resuming at startIter."""
        self.baselineTrain = 0
        self.baselineTest  = 0
        self.baseline      = {}
        self.baseline_info = {}
        self.baseline_info_by_phase = {'train': {}, 'test': {}}
        self.baselineTrainByDb = {}
        self._reset_skipped_eval_queries()
        self._warmup_cardinality_phase()
        train_cache_hit = 0
        train_executed = 0
        while True:
            sql, db,query_id, oneloop = self.queryManager.get2validate()
            latency_timeout,iscollect,cardinality = ray.get(self.planhelper.GetLatency.remote('',sql,db,query_id, measurement_phase='baseline'))
            feature_dict,hint,left_deep,plan_json = ray.get(self.planhelper.GetFeature.remote('', sql, True, db, query_id = query_id))
            feature_dict['steps'] = np.array([0], dtype=np.float32)
            planning_time = plan_json['Planning Time'] * 1000
            if iscollect:
                train_executed += 1
                self._print_cache_work(
                    "Baseline train",
                    f"executed={train_executed}",
                    f"db={db} query={query_id} exec_ms={latency_timeout[0]:.3f} plan_ms={planning_time:.3f}",
                )
            else:
                train_cache_hit += 1
            key_out = '_'.join(['base',db,query_id])
            value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(planning_time)])
            self.anaManger.recordRuning(key_out, value_out)
            self.anaManger.recordeval(db,query_id, latency_timeout[0], planning_time, val_iter = 0, phase='train')
            self.anaManger.recordQuery(db,query_id, False)
            # print('\t Train Set Database:{} Query:{} Cardinality:{} Execution Time:{:.4f}'.format(db,query_id,cardinality,latency_timeout[0]))
            self.baseline[db+'|'+query_id] = latency_timeout[0]
            self.baseline_info[db+'|'+query_id] = {
                'execution_time': float(latency_timeout[0]),
                'planning_time': float(planning_time),
                'total_time': float(latency_timeout[0]),
            }
            self.baseline_info_by_phase['train'][db+'|'+query_id] = self.baseline_info[db+'|'+query_id]
            self.bpm.update_best_executed_plan.remote(db,query_id,deepcopy(feature_dict),latency_timeout[0])
            self.bpm.update_RL_esbest.remote(db,query_id,deepcopy(feature_dict),latency_timeout[0])
            if self.learner and startIter == 0:
                self.learner.CollectSample.remote(
                    db,
                    query_id,
                    feature_dict,
                    latency_timeout[0],
                    cardinality,
                    latency_timeout[1],
                    planning_time_cum=planning_time,
                    baseline_planning_time=planning_time,
                )
            # if left_deep:
            self.bpm.updateMask.remote(toTrain = [db+'|'+query_id])
            # else:
            #     self.bpm.updateMask.remote(toMask = [query_id])
            self.baselineTrain = self.baselineTrain + latency_timeout[0]
            self.baselineTrainByDb[db] = self.baselineTrainByDb.get(db, 0.0) + latency_timeout[0]
            if latency_timeout[1]:
                print(f'{query_id}: baseline query timed out. Increase max_time_out and regenerate the relevant latency cache to avoid inaccurate training labels.')
            if oneloop:
                break
        print(f"[Baseline train] latency_cache_hit={train_cache_hit} executed={train_executed}", flush=True)
        print('Baseline_train:{:.4f}'.format(self.baselineTrain / 1000))
        if train_only:
            if self.learner:
                self.learner.updateBaseline.remote(self.baseline_info)
            return

        test_cache_hit = 0
        test_executed = 0
        while True:
            sql, db, query_id, oneloop = self.queryManager.get2eval()
            latency_timeout,iscollect,_ = ray.get(
                self.planhelper.GetLatency.remote('', sql, db, query_id, collect_cardinality=False, measurement_phase='baseline')
            )
            if latency_timeout[1]:
                self._register_baseline_timeout('test', db, query_id)
                if oneloop:
                    break
                continue
            _, _,_, plan_json = ray.get(
                self.planhelper.GetFeature.remote('', sql, False, db, query_id=query_id, need_card_label=False)
            )
            planning_time = plan_json['Planning Time'] * 1000
            if iscollect:
                test_executed += 1
                self._print_cache_work(
                    "Baseline test",
                    f"executed={test_executed}",
                    f"db={db} query={query_id} exec_ms={latency_timeout[0]:.3f} plan_ms={planning_time:.3f}",
                )
            else:
                test_cache_hit += 1
            key_out = '_'.join(['base',db,query_id])
            value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(planning_time)])
            self.anaManger.recordRuning(key_out,value_out)
            self.anaManger.recordeval(db,query_id, latency_timeout[0], planning_time, val_iter = 0, phase='test')
            self.anaManger.recordQuery(db,query_id,True)
            # print('\t Test Set Database:{} Query:{} Cardinality:{} Execution Time:{:.4f}'.format(db,query_id,cardinality,latency_timeout[0]))
            self.baselineTest = self.baselineTest + latency_timeout[0]
            self.baseline[db+'|'+query_id] = latency_timeout[0]
            self.baseline_info[db+'|'+query_id] = {
                'execution_time': float(latency_timeout[0]),
                'planning_time': float(planning_time),
                'total_time': float(latency_timeout[0]),
            }
            self.baseline_info_by_phase['test'][db+'|'+query_id] = self.baseline_info[db+'|'+query_id]
            if oneloop:
                break
        print(f"[Baseline test] latency_cache_hit={test_cache_hit} executed={test_executed}", flush=True)
        print(
            f"[Baseline test] skipped_timeout={len(self.skipped_eval_queries.get('test', {}))}",
            flush=True,
        )
        if self.genConfig.train_mode == 'mix+' and 'test_data_drift' in self.queryManager.extraEvalSets:
            self._run_extra_test_baseline('test_data_drift')
        if startIter == 0:
            self.anaManger.recordMetric(0)
        print('Baseline_test:{:.4f}'.format(self.baselineTest / 1000))
        if self.learner:
            self.learner.updateBaseline.remote(self.baseline_info)
        latencybuffer = ray.get(self.planhelper.GetPGLatencyBuffer.remote())
        ray.get(self.bpm.update_latencyBuffer.remote(latencybuffer))
        if not os.path.exists(self.genConfig.pg_latency):
            os.makedirs(os.path.dirname(self.genConfig.pg_latency), exist_ok=True)
            shutil.copy(self.genConfig.latency_buffer_path, self.genConfig.pg_latency)

    def _reset_validation_env(self, sqlinfo):
        try:
            return self.evalEnv.reset(options=sqlinfo)
        except Exception as exc:
            if "statement timeout" not in str(exc).lower():
                raise
            print(
                "[Warning] validation baseline cost plan timed out twice; "
                "recording query timeout: "
                f"{sqlinfo['database']}|{sqlinfo['query_id']} | "
                f"cost_plan_timeout_ms={sqlinfo.get('cost_plan_timeout')} | "
                f"execution_timeout_ms={sqlinfo.get('execution_timeout')}"
            )
            return None

    def _validation_cost_plan_timeout(self, metric_phase, db, query_id):
        baseline_record = self._get_baseline_info(metric_phase, db, query_id)
        baseline_planning = float(baseline_record.get('planning_time', 0.0))
        if baseline_planning <= 0:
            return int(self.genConfig.max_time_out)
        timeout = self.genConfig.cost_plan_timeoutcoeff * baseline_planning
        timeout = max(float(self.genConfig.cost_plan_timeout_min_ms), timeout)
        return max(1, min(int(math.ceil(timeout)), int(self.genConfig.max_time_out)))

    def _validation_execution_timeout(self, metric_phase, db, query_id):
        baseline_record = self._get_baseline_info(metric_phase, db, query_id)
        timeout = self.genConfig.timeoutcoeff * baseline_record['execution_time']
        return max(1, min(int(math.ceil(timeout)), int(self.genConfig.max_time_out)))

    @staticmethod
    def _is_statement_timeout(exc):
        return "statement timeout" in str(exc).lower()

    def _cost_plan_timeout_execution(self, execution_timeout, planning_start):
        return {
            'optimal_feature': None,
            'latency_timeout': [float(execution_timeout), True],
            'planning_time': time.time() - planning_start,
            'selection_diagnostic': None,
        }

    def _collect_candidates(self, obs, info, breakdown):
        return collect_candidates(
            self.evalEnv, self.planner, self.genConfig.num_policies,
            self.genConfig.maxsteps, obs, info, breakdown,
        )

    def _add_balance_candidates(self, db, query_id, sql, hint_feature):
        for hint, feature, feature_meta in hint_feature[1:]:
            feature.pop('action_mask', None)
            self.bpm.add_balances.remote(
                db,
                query_id,
                hint,
                feature,
                sql,
                planning_time_cum=float(feature_meta.get('planning_time_cum_ms', 0.0)),
            )

    def _execute_selection_strategy_diagnostic(
        self, hint_feature, optimal_hint, optimal_meta, sql, db, query_id, timeout
    ):
        score_by_hint = {
            candidate['hint']: candidate
            for candidate in optimal_meta.get('candidate_scores', [])
        }
        candidates = []
        seen_hints = set()
        latency_by_hint = {}
        for hint, feature, feature_meta in hint_feature:
            if hint in seen_hints:
                continue
            seen_hints.add(hint)
            latency_timeout, _, _ = ray.get(
                self.planhelper.GetLatency.remote(
                    hint,
                    sql,
                    db,
                    query_id,
                    timeout=timeout,
                    step=feature['steps'][0],
                    collect_cardinality=False,
                    measurement_phase='evaluation',
                )
            )
            latency_by_hint[hint] = latency_timeout
            score_record = score_by_hint.get(hint, {})
            candidates.append({
                'hint': hint,
                'step': int(round(float(feature.get('steps', [0])[0]) * self.genConfig.maxsteps)),
                'execution_ms': float(latency_timeout[0]),
                'timed_out': bool(latency_timeout[1]),
                'planning_time_cum_ms': float(feature_meta.get('planning_time_cum_ms', 0.0)),
                'score': score_record.get('score'),
                'adjusted_score': score_record.get('adjusted_score'),
                'risk': score_record.get('risk'),
            })

        by_hint = {candidate['hint']: candidate for candidate in candidates}
        baseline = candidates[0]
        raw_hint = optimal_meta.get('raw_selected_hint', optimal_hint)
        scorer = by_hint.get(raw_hint, by_hint[optimal_hint])
        safe_select = by_hint[optimal_hint]
        oracle = min(candidates, key=lambda candidate: candidate['execution_ms'])
        diagnostic = {
            'baseline': baseline,
            'oracle': oracle,
            'scorer': scorer,
            'safe_select': safe_select,
            'safe_fallback_to_baseline': bool(
                optimal_meta.get('safe_fallback_to_baseline', False)
            ),
            'candidates': candidates,
        }
        return latency_by_hint[optimal_hint], diagnostic

    def _execute_validation_candidate(
        self,
        hint_feature,
        sql,
        db,
        query_id,
        is_test,
        predict_phase,
        metric_phase,
        collect_samples,
        update_train_state,
        balance_keys,
        planning_start,
        breakdown,
    ):
        predict_start = time.time()
        optimal_hint, optimal_feature, optimal_meta = ray.get(
            self.scorer_actor.GetPrediction.remote(hint_feature, phase=predict_phase)
        )
        breakdown['predict_ms'] += (time.time() - predict_start) * 1000.0
        breakdown['selected_step'] = int(
            round(float(optimal_feature.get('steps', [0])[0]) * self.genConfig.maxsteps)
        )
        planning_time = time.time() - planning_start
        baseline_record = self._get_baseline_info(metric_phase, db, query_id)
        timeout = self.genConfig.timeoutcoeff * baseline_record['execution_time']
        selection_diagnostic = None

        if is_test and getattr(self.genConfig, 'selection_strategy_diagnostics', False):
            latency_timeout, selection_diagnostic = self._execute_selection_strategy_diagnostic(
                hint_feature,
                optimal_hint,
                optimal_meta,
                sql,
                db,
                query_id,
                timeout,
            )
        elif is_test:
            latency_timeout, _, _ = ray.get(
                self.planhelper.GetLatency.remote(
                    optimal_hint,
                    sql,
                    db,
                    query_id,
                    timeout=timeout,
                    step=optimal_feature['steps'][0],
                    collect_cardinality=False,
                    measurement_phase='evaluation',
                )
            )
        else:
            latency_timeout, iscollect, cardinality = ray.get(
                self.planhelper.GetLatency.remote(
                    optimal_hint,
                    sql,
                    db,
                    query_id,
                    timeout=timeout,
                    step=optimal_feature['steps'][0],
                    measurement_phase='evaluation',
                )
            )
            if collect_samples and iscollect and self.learner:
                ray.get(
                    self.learner.CollectSample.remote(
                        db,
                        query_id,
                        optimal_feature,
                        latency_timeout[0],
                        cardinality,
                        latency_timeout[1],
                        planning_time_cum=optimal_meta.get('selected_planning_time_cum_ms', 0.0),
                        baseline_planning_time=baseline_record.get('planning_time', 0.0),
                    )
                )
            if update_train_state and query_id in balance_keys:
                self._add_balance_candidates(db, query_id, sql, hint_feature)

        return {
            'optimal_feature': optimal_feature,
            'latency_timeout': latency_timeout,
            'planning_time': planning_time,
            'selection_diagnostic': selection_diagnostic,
        }

    def _record_validation_result(
        self,
        db,
        query_id,
        val_iter,
        is_test,
        metric_phase,
        can_optimize,
        update_train_state,
        execution,
        breakdown,
    ):
        latency_timeout = execution['latency_timeout']
        optimal_feature = execution['optimal_feature']
        if not is_test and can_optimize and update_train_state and optimal_feature is not None:
            _, bestexec = ray.get(self.bpm.get_best_executed_plan.remote(db, query_id))
            advantage = (bestexec - latency_timeout[0]) / bestexec
            if advantage >= self.genConfig.splitpoint[-1]:
                self.bpm.update_best_executed_plan.remote(
                    db,
                    query_id,
                    deepcopy(optimal_feature),
                    latency_timeout[0],
                )
            self.bpm.update_RL_esbest.remote(
                db,
                query_id,
                deepcopy(optimal_feature),
                latency_timeout[0],
            )

        planning_time = execution['planning_time'] * 1000.0
        breakdown['planning_total_ms'] = planning_time
        known_ms = sum(
            breakdown[key]
            for key in ('reset_ms', 'action_ms', 'env_step_ms', 'obs_copy_ms', 'predict_ms')
        )
        breakdown['other_ms'] = max(planning_time - known_ms, 0.0)
        selection_diagnostic = execution.get('selection_diagnostic')
        if selection_diagnostic is not None:
            selection_diagnostic['planning_time_ms'] = planning_time
            self.anaManger.recordSelectionStrategyDiagnostic(
                metric_phase,
                db,
                query_id,
                val_iter,
                selection_diagnostic,
            )
        if metric_phase in ('train', 'test'):
            self.anaManger.recordeval(
                db,
                query_id,
                latency_timeout[0],
                planning_time,
                val_iter=val_iter,
                phase=metric_phase,
            )
        else:
            self.anaManger.recordExtraEval(
                metric_phase,
                db,
                query_id,
                latency_timeout[0],
                planning_time,
                val_iter=val_iter,
            )
        self.anaManger.recordPlanningBreakdown(
            db,
            query_id,
            val_iter,
            breakdown,
            phase=metric_phase,
        )
        phase = 'test' if is_test else 'train'
        key_out = '_'.join([phase, str(val_iter), db, query_id])
        value_out = '|'.join([
            '{:.4f}'.format(latency_timeout[0]),
            '{:.4f}'.format(planning_time),
        ])
        self.anaManger.recordRuning(key_out, value_out)

    def Validate(self, valIter, loopPro = 0, initialPro = 0.25, isTest = False, Explore = False,
                 predict_phase = None, metric_phase = None, collect_samples = True,
                 update_train_state = True, eval_phase = None):
        """Evaluate the model for a workload phase. Control sample collection and training-state updates independently; return total execution time and selected search steps."""
        metric_phase = metric_phase or ('test' if isTest else 'train')
        predict_phase = predict_phase or ('test' if isTest else 'train')
        eval_phase = eval_phase or (metric_phase if isTest else None)
        balanceKeys = []
        skipped_timeout_count = 0
        if not isTest:
            sortedQueryID = ray.get(self.planhelper.GetSortedQueryID.remote())
            balanceKeys = sortedQueryID[:int(loopPro * len(sortedQueryID))]
        while True:
            if isTest:
                sql, db, query_id, oneloop = self.queryManager.get2eval(phase=eval_phase)
            else:
                sql, db, query_id, oneloop = self.queryManager.get2validate()
            if isTest and self._is_skipped_eval_query(eval_phase, db, query_id):
                skipped_timeout_count += 1
                if oneloop:
                    break
                continue
            cost_plan_timeout = self._validation_cost_plan_timeout(
                metric_phase, db, query_id
            )
            execution_timeout = self._validation_execution_timeout(
                metric_phase, db, query_id
            )
            sqlinfo = {
                'sql': sql,
                'database': db,
                'query_id': query_id,
                'cost_plan_timeout': cost_plan_timeout,
                'execution_timeout': execution_timeout,
            }
            planning_start = time.time()
            breakdown = {
                'can_optimize': False,
                'loop_count': 0,
                'candidate_count': 0,
                'selected_step': 0,
                'stop_taken': False,
                'stop_step': 0,
                'terminate_reason': 'no_opti',
                'reset_ms': 0.0,
                'action_ms': 0.0,
                'env_step_ms': 0.0,
                'obs_copy_ms': 0.0,
                'predict_ms': 0.0,
                'other_ms': 0.0,
                'planning_total_ms': 0.0,
            }
            execution = self._latency_gate_skip_execution(isTest, Explore, metric_phase, db, query_id, sql=sql)
            if execution is not None:
                breakdown['terminate_reason'] = 'adaptive_gate_skip'
                self._record_validation_result(db, query_id, valIter, True, metric_phase,
                                               False, False, execution, breakdown)
                if oneloop:
                    break
                continue
            if self._tag_initial_plan_feature is not None:
                sqlinfo['initial_plan_feature'] = self._tag_initial_plan_feature
            reset_start = time.time()
            reset_result = self._reset_validation_env(sqlinfo)
            if reset_result is None:
                if not Explore:
                    execution = self._cost_plan_timeout_execution(
                        execution_timeout, planning_start
                    )
                    self._record_validation_result(
                        db, query_id, valIter, isTest, metric_phase, False,
                        update_train_state, execution, breakdown,
                    )
                if oneloop:
                    break
                continue
            obs, info = reset_result
            breakdown['reset_ms'] = (time.time() - reset_start) * 1000.0
            breakdown['can_optimize'] = bool(info['can_optimize'])
            try:
                hint_feature = self._collect_candidates(obs, info, breakdown)
            except Exception as exc:
                if not self._is_statement_timeout(exc):
                    raise
                print(
                    "[Warning] validation candidate cost plan timed out twice; "
                    f"recording query timeout: {db}|{query_id} | "
                    f"cost_plan_timeout_ms={cost_plan_timeout} | "
                    f"execution_timeout_ms={execution_timeout}",
                    flush=True,
                )
                if not Explore:
                    execution = self._cost_plan_timeout_execution(
                        execution_timeout, planning_start
                    )
                    self._record_validation_result(
                        db, query_id, valIter, isTest, metric_phase, True,
                        update_train_state, execution, breakdown,
                    )
                if oneloop:
                    break
                continue
            execution = None
            if info['can_optimize']:
                if not Explore:
                    execution = self._execute_validation_candidate(
                        hint_feature,
                        sql,
                        db,
                        query_id,
                        isTest,
                        predict_phase,
                        metric_phase,
                        collect_samples,
                        update_train_state,
                        balanceKeys,
                        planning_start,
                        breakdown,
                    )
                elif update_train_state and random.random() < initialPro:
                    self._add_balance_candidates(db, query_id, sql, hint_feature)
            else:
                planning_time = time.time() - planning_start
                latency_timeout, _, _ = ray.get(
                    self.planhelper.GetLatency.remote(
                        '',
                        sql,
                        db,
                        query_id,
                        collect_cardinality=not isTest,
                        measurement_phase='evaluation',
                    )
                )
                execution = {
                    'optimal_feature': None,
                    'latency_timeout': latency_timeout,
                    'planning_time': planning_time,
                }
            if not Explore:
                self._record_validation_result(
                    db,
                    query_id,
                    valIter,
                    isTest,
                    metric_phase,
                    info['can_optimize'],
                    update_train_state,
                    execution,
                    breakdown,
                )
            if oneloop:
                break 
        if isTest:
            phase_prefix = 'TestDataDrift' if eval_phase == 'test_data_drift' else 'Test'
            self.writer.add_scalar(
                f'{phase_prefix}/SkippedBaselineTimeoutCount',
                skipped_timeout_count,
                valIter,
            )
            if skipped_timeout_count:
                print(
                    f'[Validate {eval_phase}] skipped_baseline_timeout={skipped_timeout_count}',
                    flush=True,
                )
        estotal, best_steps = ray.get(self.bpm.get_best_executed.remote())
        if isTest:
            self.anaManger.recordTime(f'{valIter}_iter_time')
        return estotal, best_steps
    
    def SelfCheck(self, totalIter=1):
        """Check that required model files exist and can be loaded."""
        print(f'---------- Starting model checks ----------')
        if not os.path.exists(self.genConfig.scorer_path):
            raise FileNotFoundError(f"自检失败，优势模型不存在: {self.genConfig.scorer_path}")
        agent_path = os.path.join(self.genConfig.agent_checkpoint,'rllib_checkpoint.json')
        if not os.path.exists(agent_path):
            raise FileNotFoundError(f"自检失败，动作模型不存在: {agent_path}")
        try:
            self.planner = None
            self.LoadPlanner(self.genConfig.agent_checkpoint)
            self.scorer_actor.LoadModel.remote(scorer_path = self.genConfig.scorer_path)
            self.genConfig.opti_result_path = os.path.join(self.genConfig.opti_result_path,self.genConfig.time)
            self.RunInference(self.queryManager.test_path, train_iter = totalIter)
        except Exception as e:
            raise Exception(f'自检失败，错误信息：{e}')
        print(f'---------- Model checks completed; directory: {os.path.abspath(os.path.dirname(self.genConfig.scorer_path))} ----------')

    def RunInference(self, sql_dir, exe = True, train_iter = None):
        """Optimize queries in sql_dir and optionally execute the original and hinted SQL."""
        time_now = datetime.now()
        plan_start_time = time_now.strftime('%Y/%m/%d %H.%M.%S')
        result_time = time_now.strftime('%Y%m%d%H%M%S')
        if not os.path.exists(self.genConfig.opti_result_path):
            os.makedirs(self.genConfig.opti_result_path)
        if train_iter is None:
            result_path = os.path.join(self.genConfig.opti_result_path, f'{result_time}.csv')
        else:
            result_path = os.path.join(self.genConfig.opti_result_path, f'train_{train_iter}_{result_time}.csv')
        sql_files = [f for f in os.listdir(sql_dir) if f.endswith('.sql')]
        sql_files.sort()  # Sort by filename.
        sql_num = len(sql_files)
        save_sql_dir = os.path.abspath(sql_dir)  # Resolve the original SQL directory to an absolute path.
        opti_num = 0    # Number of optimizable SQL queries.
        query_id_list = []    # Query IDs.
        opti_necessity_list = []    # Optimization decisions.
        sql_list = []    # Original SQL queries.
        opti_result_list = []    # Optimized SQL queries.
        plan_time_list = []    # Inference times.
        no_opti_latency_list = []    # Original execution times.
        opti_latency_list = []    # Optimized execution times.
        result_latency_list = []    # Optimized execution times.
        opti_rate_list = []    # Optimization ratios.
        for db in self.queryManager.test_database:
            for sql_file in sql_files:
                with open(os.path.join(sql_dir, sql_file), 'r') as f:
                    sql = f.read()
                query_id = os.path.splitext(sql_file)[0]  # Remove the file extension.
                sqlinfo = {'sql':sql, 'database':db, 'query_id':query_id}
                start_time = time.time()
                try:
                    obs, info = self.evalEnv.reset(options = sqlinfo)
                except:
                    raise Exception(f"推理失败，请检查SQL {query_id}.sql 是否适用于此模型！")
                breakdown = {
                    'can_optimize': bool(info.get('can_optimize', False)),
                    'loop_count': 0,
                    'candidate_count': 1,
                    'stop_taken': False,
                    'stop_step': 0,
                    'terminate_reason': 'no_opti',
                    'reset_ms': round((time.time() - start_time) * 1000.0, 3),
                    'action_ms': 0.0,
                    'env_step_ms': 0.0,
                    'obs_copy_ms': 0.0,
                    'predict_ms': 0.0,
                    'other_ms': 0.0,
                    'planning_total_ms': 0.0,
                }
                try:
                    hint_feature = self._collect_candidates(obs, info, breakdown)
                except Exception as exc:
                    raise RuntimeError(
                        f'推理失败，动作模型或环境执行出错: {db}|{query_id}'
                    ) from exc
                if info['can_optimize']:
                    predict_start = time.time()
                    optimal_hint,optimal_feature,_ = ray.get(self.scorer_actor.GetPrediction.remote(hint_feature, phase='infer'))  # Select the best execution plan.
                    breakdown['predict_ms'] += (time.time() - predict_start) * 1000.0
                else:
                    optimal_hint = ''
                plan_time = round((time.time() - start_time)*1000, 3)  # Optimization inference time.
                known_ms = (
                    breakdown['reset_ms']
                    + breakdown['action_ms']
                    + breakdown['env_step_ms']
                    + breakdown['obs_copy_ms']
                    + breakdown['predict_ms']
                )
                breakdown['planning_total_ms'] = plan_time
                breakdown['other_ms'] = max(plan_time - known_ms, 0.0)
                infer_iter = train_iter if train_iter is not None else -1
                self.anaManger.recordPlanningBreakdown(db, query_id, infer_iter, breakdown, phase='infer')
                # Optional scorer-inference timing diagnostic.
                if exe:
                    if info['can_optimize']:
                        print(f"Running on database {db}; inference query: {query_id}")
                    query_id_list.append(db+'|'+query_id)
                    sql_list.append(sql)
                    plan_time = 0   # Ignore plan_time in this calculation.
                    plan_time_list.append(plan_time)
                    no_opti_latency,_,_ = ray.get(
                        self.planhelper.GetLatency.remote(
                            '', sql, db, query_id, use_buffer=False, collect_cardinality=False,
                            measurement_phase='baseline',
                        )
                    )  # Get the original SQL execution time.
                    no_opti_latency_list.append(no_opti_latency[0])
                    if optimal_hint!='':
                        opti_latency,_,_ = ray.get(
                            self.planhelper.GetLatency.remote(
                                optimal_hint,
                                sql,
                                db,
                                query_id,
                                use_buffer=False,
                                timeout=(no_opti_latency[0] - plan_time),
                                collect_cardinality=False,
                                measurement_phase='evaluation',
                            )
                        )  # Get the optimized SQL execution time.
                        if (opti_latency[0]+plan_time) < no_opti_latency[0] * (1 - self.genConfig.recommend_threshold):    # Require the optimized query to improve execution by more than 5%.
                            result_latency = opti_latency
                            opti_num += 1
                            opti_necessity_list.append('Y')
                            opti_latency_list.append(opti_latency[0])
                            result_latency_list.append(result_latency[0])
                            # opti_rate_list.append(f"{round((1-(opti_latency[0]+plan_time) / no_opti_latency[0])*100, 2)}%")
                            opti_rate_list.append(f"{round((no_opti_latency[0] / (opti_latency[0]+plan_time))*100, 2)}%")
                            if self.genConfig.DBMS == 'postgres':
                                opti_result_list.append(optimal_hint + sql)  # PostgreSQL can execute the hinted SQL directly.
                            # elif self.genConfig.DBMS == 'opengauss' or self.genConfig.DBMS == 'gaussdb':
                            # Other database backends may require different hint formatting.
                            # elif self.genConfig.DBMS == 'hive':
                            #     from Hive.hive_helper import HiveHelper
                            # Hive requires rewriting the SQL with the selected hint.
                        else:
                            result_latency = no_opti_latency
                            opti_necessity_list.append('N')
                            opti_latency_list.append('-')
                            result_latency_list.append(result_latency[0])
                            opti_rate_list.append('-')
                            opti_result_list.append('-')
                    else:
                        result_latency = no_opti_latency
                        opti_necessity_list.append('N')
                        opti_latency_list.append('-')
                        result_latency_list.append(result_latency[0])
                        opti_rate_list.append('-')
                        opti_result_list.append('-')
                else:
                    no_opti_latency = [0, False]
                    result_latency = [0, False]
        if exe:
            opti_data = {
                '查询ID': query_id_list,
                '是否需要优化': opti_necessity_list,
                '优化前执行时间ms': no_opti_latency_list,
                '优化后执行时间ms': opti_latency_list,
                # Optional optimization-time output column.
                '性能提升率': opti_rate_list,
                '优化后的SQL': opti_result_list
            }
            df = pd.DataFrame(opti_data)
            # Sort optimization decisions with Y before N.
            df = df.sort_values(by='是否需要优化', key=lambda x: x.map({'Y': 0, 'N': 1}))

            # Compute the total optimization ratio.
            no_opti_latency_sum = sum(no_opti_latency_list) # Total original execution time.
            result_latency_sum = sum(result_latency_list) # Total optimized execution time.
            plan_time_sum = sum(plan_time_list) # Total optimization overhead.
            # total_opti_rate = round((no_opti_latency_sum - result_latency_sum - plan_time_sum) / no_opti_latency_sum * 100, 2)
            total_opti_rate = round((no_opti_latency_sum / (result_latency_sum + plan_time_sum))*100, 2)
            if total_opti_rate < 0:
                total_opti_rate = 0
            summary_data = pd.DataFrame({
                '总体性能提升': [f"{total_opti_rate}%"]
            })

            plan_end_time = datetime.now().strftime('%Y/%m/%d %H.%M.%S')
            start_data = pd.DataFrame({
                '测试开始时间': [plan_start_time],
                '测试结束时间': [plan_end_time],
                '测试SQL路径': [save_sql_dir],
                '可优化SQL数量|总数': [f"{opti_num} | {sql_num}"],
                '优化覆盖率': [f"{round(opti_num/sql_num*100, 2)}%"]
            })
            # Reset the DataFrame index to preserve row order when merging.
            df_sorted = pd.DataFrame(df).reset_index(drop=True)
            # Merge columns while preserving the sorted row order.
            final_df = pd.concat([start_data.reset_index(drop=True), df_sorted, summary_data.reset_index(drop=True)], axis=1)
            final_df.to_csv(result_path, index=False, encoding='utf-8-sig') 
            print(f"Inference completed; overall improvement: {total_opti_rate}%; results saved to: {os.path.abspath(result_path)}")
        self.anaManger.writeout()

    
