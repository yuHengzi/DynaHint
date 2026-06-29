# -*- coding: utf-8 -*-
from config import Config
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms import Algorithm
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import numpy as np
import time
from datetime import datetime
import argparse
import random
import os,shutil
import json
import pandas as pd
import hashlib
from model import create_custom_model
from manager import QueryManager, BestPlanManager, ResultManager
from learner import Learner, RemoteScorer
from encoding import is_encoding_cache_ready, write_json_atomic
import signal
import sys

def gen_policy(i,genConfig):
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
    pol_id = "policy_{}".format(agent_id)
    return pol_id

class DynaHint():
    @staticmethod
    def _ensure_ray_initialized(genConfig):
        if ray.is_initialized():
            return
        ray.init(
            num_cpus=int(getattr(genConfig, 'use_resources', 1)),
            num_gpus=1,
            include_dashboard=False,
            ignore_reinit_error=True,
        )

    def __init__(self, config, AutoGetParam, isFirst=False, istrain=True,best_model_wrl = 1.0,best_model_gmrl = 1.0):
        self.genConfig    = config
        self.istrain      = istrain
        self._ensure_ray_initialized(self.genConfig)
        self.planhelper   = RemotePlanHelper.remote(self.genConfig)
        self.bpm          = BestPlanManager.options(name="bpm").remote(self.genConfig)
        self.queryManager = QueryManager(self.genConfig, planhelper = self.planhelper, isfirst=isFirst)
        self._wait_for_encoding_cache()
        if AutoGetParam:
            self.AutoGetParam()
        self.predictor    = RemoteScorer.remote(self.genConfig, istrain=istrain)
        if self.genConfig.update_scorer:
            self.learner  = Learner.remote(self.bpm, self.planhelper, self.predictor, self.genConfig)
        else:
            self.learner  = None
        self.planner      = None
        self.writer       = SummaryWriter(log_dir = self.genConfig.runstate)
        self.evalEnv      = DynaHintEnvTest({'planhelper':self.planhelper, 'genConfig':self.genConfig})
        self.anaManger    = ResultManager(self.genConfig, self.writer)
        self.best_model_wrl = min(best_model_wrl, 1.0)
        self.best_model_gmrl = min(best_model_gmrl, 1.0)
        self.tmp_planner_path = os.path.join(os.path.dirname(os.path.dirname(self.genConfig.agent_checkpoint)), 'tmp_planner/')
        self.start_time = datetime.now()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        ModelCatalog.register_custom_model("gen_model", create_custom_model(self.genConfig))

    def _wait_for_encoding_cache(self, timeout_sec=None, poll_sec=1.0):
        timeout_sec = getattr(self.genConfig, 'encoding_bootstrap_timeout_sec', 300) if timeout_sec is None else timeout_sec
        start_time = time.time()
        while not is_encoding_cache_ready(self.genConfig.encoding_path):
            if time.time() - start_time >= timeout_sec:
                raise TimeoutError(
                    f"encoding cache missing or incomplete after {timeout_sec}s: {self.genConfig.encoding_path}"
                )
            time.sleep(poll_sec)

    def _should_save_model(self, wrl_test, wrl_train, gmrl_test, gmrl_train, speedup_test, speedup_train):
        metric = getattr(self.genConfig, 'checkpoint_metric', 'train_wrl')
        if metric == 'train_wrl':
            if wrl_train < self.best_model_wrl:
                return True
            if abs(wrl_train - self.best_model_wrl) <= 1e-9 and gmrl_train < self.best_model_gmrl:
                return True
            return False
        if metric == 'train_speedup':
            if speedup_train > self.best_model_wrl:
                return True
            if abs(speedup_train - self.best_model_wrl) <= 1e-9 and gmrl_train < self.best_model_gmrl:
                return True
            return False
        if metric == 'test_speedup':
            if speedup_test > self.best_model_wrl:
                return True
            if abs(speedup_test - self.best_model_wrl) <= 1e-9 and gmrl_test < self.best_model_gmrl:
                return True
            return False
        if gmrl_test < self.best_model_gmrl:
            return True
        if abs(gmrl_test - self.best_model_gmrl) <= 1e-9 and wrl_test < self.best_model_wrl:
            return True
        return False

    def _log_train_db_best_plan_wrl(self, totals_by_db, step):
        for db, best_exec_total in totals_by_db.items():
            baseline_total = self.baselineTrainByDb.get(db, 0.0)
            if baseline_total <= 0:
                continue
            self.writer.add_scalar(
                f'Others/BestPlanWRL/{db}',
                best_exec_total / baseline_total,
                step,
            )

    def _should_run_train_safe_eval(self):
        if not getattr(self.genConfig, 'enable_train_safe_eval', False):
            return False
        return bool(getattr(self.genConfig, 'safe_select_enable_train', False)) != bool(
            getattr(self.genConfig, 'safe_select_enable_test', False)
        )

    def _validate_train_with_test_phase_if_enabled(self, val_iter):
        if not self._should_run_train_safe_eval():
            return
        self.Validate(
            val_iter,
            predict_phase='test',
            metric_phase='train_safe',
            collect_samples=False,
            update_train_state=False,
        )

    def _validate_mix_plus_data_drift_if_enabled(self, val_iter):
        if self.genConfig.train_mode != 'mix+':
            return
        if 'test_data_drift' not in getattr(self.queryManager, 'extraEvalSets', {}):
            return
        self.Validate(
            val_iter,
            isTest=True,
            predict_phase='test',
            metric_phase='test_data_drift',
            eval_phase='test_data_drift',
            collect_samples=False,
            update_train_state=False,
            record_cross_db=False,
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

    def _flush_scorer_metrics(self):
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

    def _signal_handler(self, signum, frame):
        print('\n---------- Safely terminating the program... ----------')
        try:
            if hasattr(self, 'anaManger'):
                self.anaManger.close()
            ray.shutdown()
            print('---------- Program terminated safely ----------')
        except Exception as e:
            print(f'Error during termination: {e}')
        finally:
            sys.exit(0)

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

    def _print_cache_work(self, label, count, detail=''):
        message = f"[{label}] {count}"
        if detail:
            message += f" | {detail}"
        print(message, flush=True)

    def _sql_hash_for_cardinality_cache(self, sql):
        normalized_sql = ' '.join(str(sql).strip().rstrip(';').lower().split())
        return hashlib.md5(normalized_sql.encode('utf-8')).hexdigest()

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
                        keys.add('|'.join([str(record['database']), str(record['query_id']), str(record['sql_hash'])]))
                except Exception:
                    continue
        return keys

    def _iter_query_phase(self, phase):
        if phase == 'train':
            query_set = self.queryManager.validateSet
        elif phase == 'test':
            query_set = self.queryManager.testSet
        else:
            query_set = self.queryManager.extraEvalSets.get(phase, pd.DataFrame(columns=['sql']))
        for db_and_queryid in query_set.index:
            db, query_id = db_and_queryid.split('|')
            yield query_set.loc[db_and_queryid, 'sql'], db, query_id

    def _warmup_cardinality_phase(self, phase):
        cache_keys = self._load_cardinality_cache_keys()
        cache_hit = 0
        miss = 0
        for sql, db, query_id in self._iter_query_phase(phase):
            sql_hash = self._sql_hash_for_cardinality_cache(sql)
            cache_key = '|'.join([str(db), str(query_id), str(sql_hash)])
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
        print(f"[Cardinality {phase}] cache_hit={cache_hit} miss={miss}", flush=True)

    def _run_extra_test_baseline(self, phase):
        phase_total = 0.0
        self.baseline_info_by_phase.setdefault(phase, {})
        cache_hit = 0
        executed = 0
        while True:
            sql, db, query_id, oneloop = self.queryManager.get2eval(phase=phase)
            latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote('',sql,db,query_id))
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
                'total_time': float(latency_timeout[0] + planning_time),
            }
            phase_total += latency_timeout[0] + planning_time
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
        print('Baseline_{}:{:.4f}'.format(phase, phase_total / 1000))

    def RunBaseline(self, startIter = 0):
        self.baselineTrain = 0
        self.baselineTest  = 0
        self.baseline      = {}
        self.baseline_info = {}
        self.baseline_info_by_phase = {'train': {}, 'test': {}}
        self.baselineTrainByDb = {}
        self._warmup_cardinality_phase('train')
        self._warmup_cardinality_phase('test')
        if self.genConfig.train_mode == 'mix+' and 'test_data_drift' in self.queryManager.extraEvalSets:
            self._warmup_cardinality_phase('test_data_drift')
        train_cache_hit = 0
        train_executed = 0
        while True:
            sql, db,query_id, oneloop = self.queryManager.get2validate()
            latency_timeout,iscollect,cardinality = ray.get(self.planhelper.GetLatency.remote('',sql,db,query_id))
            feature_dict,hint,left_deep,plan_json = ray.get(self.planhelper.GetFeature.remote('', sql, True, db, query_id = query_id))
            feature_dict['steps'] = np.array([0])
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
                'total_time': float(latency_timeout[0] + planning_time),
            }
            self.baseline_info_by_phase['train'][db+'|'+query_id] = self.baseline_info[db+'|'+query_id]
            self.bpm.update_scorer_esbest.remote(db,query_id,deepcopy(feature_dict),latency_timeout[0])
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
            self.baselineTrain = self.baselineTrain + latency_timeout[0] + planning_time
            self.baselineTrainByDb[db] = self.baselineTrainByDb.get(db, 0.0) + latency_timeout[0] + planning_time
            if latency_timeout[1]:
                print(f'{query_id}Initial query timed out. Increase max_time_out and remove cached latency files before retrying, or the model error may increase.')
            if oneloop:
                break
        print(f"[Baseline train] latency_cache_hit={train_cache_hit} executed={train_executed}", flush=True)
        print('Baseline_train:{:.4f}'.format(self.baselineTrain / 1000))

        test_cache_hit = 0
        test_executed = 0
        while True:
            sql, db, query_id, oneloop = self.queryManager.get2eval()
            latency_timeout,iscollect,_ = ray.get(self.planhelper.GetLatency.remote('',sql,db,query_id))
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
            self.baselineTest = self.baselineTest + latency_timeout[0] + planning_time
            self.baseline[db+'|'+query_id] = latency_timeout[0]
            self.baseline_info[db+'|'+query_id] = {
                'execution_time': float(latency_timeout[0]),
                'planning_time': float(planning_time),
                'total_time': float(latency_timeout[0] + planning_time),
            }
            self.baseline_info_by_phase['test'][db+'|'+query_id] = self.baseline_info[db+'|'+query_id]
            if oneloop:
                break
        print(f"[Baseline test] latency_cache_hit={test_cache_hit} executed={test_executed}", flush=True)
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
            shutil.copy(self.genConfig.latency_buffer_path, self.genConfig.pg_latency)

    def BuildPlanner(self, train_batch_size, workersNum, planner_path = None):
        if planner_path is None:
            policies = {"policy_{}".format(i): gen_policy(i,self.genConfig) for i in range(self.genConfig.num_policies)}
            planner_config = (
                PPOConfig()
                .environment(DynaHintEnvTrain, env_config = {'genConfig':self.genConfig}, disable_env_checking = True)
                .training(train_batch_size = train_batch_size)
                .multi_agent(policies = policies, policy_mapping_fn = policy_mapping_fn)
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
                self.LoadPlanner(planner_path)
                            
        # resources = ray.available_resources()
        # total_resources = ray.cluster_resources()

    def InitScorer(self, initialPro = 0.25):
        estotal,_ = self.Validate(0, Explore = True, initialPro = initialPro)
        train_db_estotal, _ = ray.get(self.bpm.get_scorer_best_by_db.remote())
        self.writer.add_scalar('Others/Accumulated Excecuted Plans Num', 0, 0)
        self.writer.add_scalar('Others/Train Set Best Plan WRL', estotal / self.baselineTrain, 0)
        self._log_train_db_best_plan_wrl(train_db_estotal, 0)
        runner_ref = self.learner.Runing.remote()
        time.sleep(5)
        ray.get(self.bpm.update_schedule.remote(True))
        ray.get(runner_ref)

    def Validate(self, valIter, loopPro = 0, initialPro = 0.25, isTest = False, Explore = False,
                 record_cross_db = False, predict_phase = None, metric_phase = None,
                 collect_samples = True, update_train_state = True, eval_phase = None):
        metric_phase = metric_phase or ('test' if isTest else 'train')
        predict_phase = predict_phase or ('test' if isTest else 'train')
        eval_phase = eval_phase or (metric_phase if isTest else None)
        actions = {}
        balanceKeys = None
        if not isTest:
            sortedQueryID = ray.get(self.planhelper.GetSortedQueryID.remote())
            balanceKeys = sortedQueryID[:int(loopPro * len(sortedQueryID))]
        while True:
            if isTest:
                sql, db, query_id, oneloop = self.queryManager.get2eval(phase=eval_phase)
            else:
                sql, db, query_id, oneloop = self.queryManager.get2validate()
            sqlinfo = {'sql':sql, 'database':db, 'query_id':query_id}
            planning_start = time.time()
            breakdown = {
                'use_dynahint': False,
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
            reset_start = time.time()
            obs, info = self.evalEnv.reset(options = sqlinfo)
            breakdown['reset_ms'] = (time.time() - reset_start) * 1000.0
            breakdown['use_dynahint'] = bool(info['useDynaHint'])
            hint_feature = [(
                info['hint'],
                deepcopy(obs[0]),
                {'planning_time_cum_ms': float(info.get('planning_time_cum_ms', 0.0))}
            )]
            breakdown['candidate_count'] = len(hint_feature)
            optimal_hint = ''
            optimal_meta = {
                'selected_score': None,
                'score_margin_top1_top2': 0.0,
                'selected_planning_time_cum_ms': float(info.get('planning_time_cum_ms', 0.0)),
            }
            steps = 1
            if info['useDynaHint']:
                done = False
                add_bpmCandidate = False
                while not done:
                    actions.clear()
                    action_start = time.time()
                    for i in range(self.genConfig.num_policies):
                        actions[i] = self.planner.compute_single_action(obs[i], policy_id = 'policy_{}'.format(i), explore = False)
                    breakdown['action_ms'] += (time.time() - action_start) * 1000.0
                    env_step_start = time.time()
                    obs, reward, terminated, _, info_all = self.evalEnv.step(actions)
                    breakdown['env_step_ms'] += (time.time() - env_step_start) * 1000.0
                    copy_start = time.time()
                    for k in info_all:
                        # if isTest:
                        #     self.anaManger.recordExp(query_id, info_all[k]['hint'], k, steps)
                        if info_all[k].get('record_candidate', True):
                            hint_feature.append((
                                info_all[k]['hint'],
                                deepcopy(obs[k]),
                                {'planning_time_cum_ms': float(info_all[k].get('planning_time_cum_ms', 0.0))}
                            ))
                        if info_all[k].get('stop_taken', False):
                            breakdown['stop_taken'] = True
                            breakdown['stop_step'] = int(info_all[k].get('stop_step', self.genConfig.maxsteps))
                        terminate_reason = info_all[k].get('terminate_reason', '')
                        if terminate_reason:
                            breakdown['terminate_reason'] = terminate_reason
                    breakdown['obs_copy_ms'] += (time.time() - copy_start) * 1000.0
                    breakdown['loop_count'] += 1
                    breakdown['candidate_count'] = len(hint_feature)
                    steps += 1
                    if terminated['__all__'] == True:
                        if not breakdown['terminate_reason']:
                            breakdown['terminate_reason'] = 'maxsteps'
                        done = True
                if not Explore:
                    predict_start = time.time()
                    optimal_hint,optimal_feature,optimal_meta = ray.get(self.predictor.GetPrediction.remote(hint_feature, phase=predict_phase))
                    breakdown['predict_ms'] += (time.time() - predict_start) * 1000.0
                    breakdown['selected_step'] = int(round(float(optimal_feature.get('steps', [0])[0]) * self.genConfig.maxsteps))
                    planning_time = time.time() - planning_start
                    baseline_record = self._get_baseline_info(metric_phase, db, query_id)
                    baseline_exec_for_timeout = baseline_record['execution_time']
                    if isTest:
                        latency_timeout,_,_ = ray.get(self.planhelper.GetLatency.remote(optimal_hint, sql, db, query_id, timeout = self.genConfig.timeoutcoeff * baseline_exec_for_timeout))
                    else:
                        latency_timeout,iscollect,cardinality = ray.get(self.planhelper.GetLatency.remote(optimal_hint, sql, db, query_id, timeout = self.genConfig.timeoutcoeff * baseline_exec_for_timeout, step=optimal_feature['steps'][0]))
                        if update_train_state and query_id in balanceKeys:
                            add_bpmCandidate = True  # if not balance
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
                else:
                    if random.random() < initialPro:
                        add_bpmCandidate = True
                if update_train_state and add_bpmCandidate:
                    for hint,feature,feature_meta in hint_feature[1:]:
                        del feature['action_mask']
                        self.bpm.add_balances.remote(
                            db,
                            query_id,
                            hint,
                            feature,
                            sql,
                            planning_time_cum=float(feature_meta.get('planning_time_cum_ms', 0.0)),
                        )
            else:
                planning_time =  time.time() - planning_start 
                latency_timeout,_,_ = ray.get(self.planhelper.GetLatency.remote('',sql,db,query_id))
            if not Explore:
                if not isTest and info['useDynaHint'] and update_train_state:
                    bestfeature, bestexec = ray.get(self.bpm.get_scorer_esbest.remote(db,query_id))
                    Advbybest = (bestexec - latency_timeout[0]) / bestexec
                    # self.anaManger.recordBestPlanSteps(query_id,int(self.genConfig.maxsteps * optimal_feature['steps'][0]), latency_timeout[0])# TO_Delete
                    if Advbybest >= self.genConfig.splitpoint[-1]:
                        self.bpm.update_scorer_esbest.remote(db,query_id, deepcopy(optimal_feature),latency_timeout[0])
                    self.bpm.update_RL_esbest.remote(db,query_id,deepcopy(optimal_feature), latency_timeout[0])
                planning_time = planning_time * 1000
                breakdown['planning_total_ms'] = planning_time
                known_ms = (
                    breakdown['reset_ms']
                    + breakdown['action_ms']
                    + breakdown['env_step_ms']
                    + breakdown['obs_copy_ms']
                    + breakdown['predict_ms']
                )
                breakdown['other_ms'] = max(planning_time - known_ms, 0.0)
                if metric_phase in ('train', 'test'):
                    self.anaManger.recordeval(db,query_id,latency_timeout[0], planning_time, val_iter = valIter, phase=metric_phase)
                else:
                    self.anaManger.recordExtraEval(metric_phase, db, query_id, latency_timeout[0], planning_time, val_iter=valIter)
                self.anaManger.recordPlanningBreakdown(db, query_id, valIter, breakdown, phase=metric_phase)
                if isTest and record_cross_db and metric_phase == 'test':
                    db_and_queryid = db + '|' + query_id
                    baseline_exec = self.baseline_info[db_and_queryid]['execution_time']
                    baseline_plan = self.baseline_info[db_and_queryid]['planning_time']
                    selected_total = latency_timeout[0] + planning_time
                    baseline_total = baseline_exec + baseline_plan
                    train_db_hint_info = ray.get(self.bpm.get_best_hints_by_query.remote(query_id, self.queryManager.train_database))
                    cross_hint_results = []
                    negative_cross_db = False
                    for train_hint, source_train_dbs in train_db_hint_info['by_hint'].items():
                        same_as_test_hint = (train_hint == optimal_hint)
                        if same_as_test_hint:
                            cross_exec = latency_timeout[0]
                            cross_plan = planning_time
                            cross_timeout = latency_timeout[1]
                        else:
                            _, _, _, cross_plan_json = ray.get(
                                self.planhelper.GetFeature.remote(train_hint, sql, False, db, query_id=query_id, need_card_label=False)
                            )
                            cross_plan = cross_plan_json['Planning Time'] * 1000
                            cross_latency_timeout, _, _ = ray.get(
                                self.planhelper.GetLatency.remote(
                                    train_hint,
                                    sql,
                                    db,
                                    query_id,
                                    timeout=self.genConfig.timeoutcoeff * self.baseline[db_and_queryid],
                                )
                            )
                            cross_exec = cross_latency_timeout[0]
                            cross_timeout = cross_latency_timeout[1]
                        cross_total = cross_exec + cross_plan
                        if cross_timeout:
                            vs_base = 'timeout'
                            vs_test = 'timeout'
                        else:
                            vs_base = cross_total / baseline_total if baseline_total != 0 else 'timeout'
                            vs_test = cross_total / selected_total if selected_total != 0 else 'timeout'
                            if cross_total > baseline_total:
                                negative_cross_db = True
                        cross_hint_results.append({
                            'hint': train_hint,
                            'source_train_dbs': source_train_dbs,
                            'same_as_test_hint': same_as_test_hint,
                            'exec_time': cross_exec,
                            'planning_time': cross_plan,
                            'timeout': cross_timeout,
                            'vs_test_baseline_total': vs_base,
                            'vs_test_selected_total': 1.0 if same_as_test_hint and not cross_timeout else vs_test,
                        })
                    self.anaManger.recordCrossDBHintEffect(
                        db,
                        query_id,
                        valIter,
                        optimal_hint,
                        baseline_exec,
                        baseline_plan,
                        latency_timeout[0],
                        planning_time,
                        cross_hint_results,
                    )
                    query_latency_records = ray.get(self.bpm.get_query_latency_records.remote(db, query_id))
                    oracle_best_exec = None
                    explored_best_exec = None
                    explored_hints = {hint for hint, _, _ in hint_feature}
                    for hint, latency_info in query_latency_records.items():
                        if hint in ('cardinality', '_cardinality_evidence'):
                            continue
                        if not isinstance(latency_info, list) or len(latency_info) == 0:
                            continue
                        latency_value = float(latency_info[0])
                        if oracle_best_exec is None or latency_value < oracle_best_exec:
                            oracle_best_exec = latency_value
                        if hint in explored_hints and (explored_best_exec is None or latency_value < explored_best_exec):
                            explored_best_exec = latency_value
                    self.anaManger.recordGeneralizationDiagnostic(
                        db,
                        query_id,
                        valIter,
                        {
                            'selected_hint': optimal_hint,
                            'selected_score': optimal_meta.get('selected_score', None),
                            'selected_adjusted_score': optimal_meta.get('selected_adjusted_score', None),
                            'selected_risk': optimal_meta.get('selected_risk', None),
                            'raw_selected_adjusted_score': optimal_meta.get('raw_selected_adjusted_score', None),
                            'raw_selected_risk': optimal_meta.get('raw_selected_risk', None),
                            'safe_fallback_to_baseline': optimal_meta.get('safe_fallback_to_baseline', False),
                            'score_margin_top1_top2': optimal_meta.get('score_margin_top1_top2', 0.0),
                            'baseline_total': baseline_total,
                            'selected_total': selected_total,
                            'oracle_best_exec': oracle_best_exec if oracle_best_exec is not None else 'unknown',
                            'oracle_best_total': 'unknown',
                            'same_as_train_best': any(item['same_as_test_hint'] for item in cross_hint_results),
                            'negative_cross_db': negative_cross_db,
                            'stop_taken': breakdown.get('stop_taken', False),
                            'stop_step': breakdown.get('stop_step', 0),
                            'terminate_reason': breakdown.get('terminate_reason', ''),
                            'planner_found_oracle': (
                                oracle_best_exec is not None
                                and explored_best_exec is not None
                                and explored_best_exec <= oracle_best_exec + 1e-6
                            ),
                            'scorer_ranked_oracle_top1': (
                                explored_best_exec is not None
                                and latency_timeout[0] <= explored_best_exec + 1e-6
                            ),
                        },
                    )
                if isTest:
                    key_out = '_'.join(['test', str(valIter), db, query_id])
                else:
                    key_out = '_'.join(['train',str(valIter), db, query_id])
                value_out = '|'.join(['{:.4f}'.format(latency_timeout[0]),'{:.4f}'.format(planning_time)])
                self.anaManger.recordRuning(key_out,value_out)
            if oneloop:
                break 
        estotal, best_steps = ray.get(self.bpm.get_scorer_best.remote())
        if isTest:
            self.anaManger.recordTime(f'{valIter}_iter_time')
        return estotal, best_steps
    
    def Run(self, startIter = 0, totalIter = 300, valFreq = 5):
        ray.get(self.bpm.update_schedule.remote(False))
        runner_ref = self.learner.Runing.remote()
        startVal = self.genConfig.start_val
        wrl_test, wrl_train, gmrl_test, gmrl_train = 1.0, 1.0, 1.0, 1.0
        speedup_test, speedup_train = 1.0, 1.0
        for trainIter in range(startIter, totalIter):
            print(f'---------- TrainIter {trainIter+1} ----------')
            latencybuffer = ray.get(self.planhelper.GetPGLatencyBuffer.remote())
            ray.get(self.bpm.update_latencyBuffer.remote(latencybuffer))
            result = self.planner.train()
            self._log_planner_reward_curves(result, trainIter + 1)
            save_info = True
            if trainIter <= 4 or (trainIter + 1) % valFreq == 0 or trainIter == totalIter - 1:
                ray.get(self.bpm.update_schedule.remote(True))
                accumulatedPlans = ray.get(runner_ref)
                self._flush_scorer_metrics()
                self.writer.add_scalar('Others/Accumulated Excecuted Plans Num', accumulatedPlans, trainIter + 1)
                if trainIter == totalIter - 1 or ((trainIter + 1) % valFreq == 0 and trainIter+1 >= startVal):
                    print('---------- Start Validate ----------')
                    estotal, best_steps = self.Validate(trainIter + 1)
                    train_db_estotal, _ = ray.get(self.bpm.get_scorer_best_by_db.remote())
                    self.writer.add_scalar('Others/Train Set Best Plan WRL', estotal / self.baselineTrain, trainIter + 1)
                    self._log_train_db_best_plan_wrl(train_db_estotal, trainIter + 1)
                    self._validate_train_with_test_phase_if_enabled(trainIter + 1)
                    self.Validate(trainIter + 1, isTest = True, record_cross_db = False)
                    self._validate_mix_plus_data_drift_if_enabled(trainIter + 1)
                    wrl_test, wrl_train, gmrl_test, gmrl_train, speedup_test, speedup_train = self.anaManger.recordMetric(trainIter + 1)
                    self.anaManger.writeout()
                    print('---------- Validate End ----------')
                if trainIter <= 4 or self._should_save_model(wrl_test, wrl_train, gmrl_test, gmrl_train, speedup_test, speedup_train):
                    wrl_train = round(wrl_train, 4)
                    wrl_test = round(wrl_test, 4)
                    gmrl_train = round(gmrl_train, 4)
                    gmrl_test = round(gmrl_test, 4)
                    self.SavePlanner()
                    self.predictor.SaveModel.remote()
                    first_start_time,end_time = self.genConfig.SaveTrainInfo(trainIter+1, wrl_train, wrl_test, gmrl_train, gmrl_test,start_time=self.start_time,first_save=(trainIter==startIter))
                    save_info = False
                    self.best_model_wrl = wrl_test
                    self.best_model_gmrl = gmrl_test
                    print(f'---------- Model update ----------')
                ray.get(self.bpm.update_schedule.remote(False))
                runner_ref = self.learner.Runing.remote()
            self.SavePlanner(save_tmp=True)
            if save_info:
                first_start_time,end_time = self.genConfig.SaveTrainInfo(trainIter+1,start_time=self.start_time,first_save=(trainIter==startIter))
        print(f'---------- {totalIter} training iterations completed, first start time: {first_start_time}, current start time: {self.start_time.strftime("%Y/%m/%d %H.%M.%S")}, end time: {end_time.strftime("%Y/%m/%d %H.%M.%S")} ----------')
        print(f'---------- Current training elapsed time: {self.genConfig.GetTotalTime()[1]}, total training elapsed time: {self.genConfig.GetTotalTime()[0]} ----------')
        ray.get(self.bpm.update_schedule.remote(True))
        self._flush_scorer_metrics()
        # self.SelfCheck(totalIter=totalIter)

    def Sim(self, startIter = 0, totalIter = 300, valFreq = 10):
        # start_time = datetime.now()
        startVal = self.genConfig.start_val
        wrl_test, wrl_train, gmrl_test, gmrl_train = 1.0, 1.0, 1.0, 1.0
        speedup_test, speedup_train = 1.0, 1.0
        for trainIter in range(startIter, totalIter):
            print(f'---------- Planner training TrainIter {trainIter+1} ---------- ')
            result = self.planner.train()
            self._log_planner_reward_curves(result, trainIter + 1)
            save_info = True
            if trainIter == totalIter - 1 or ((trainIter + 1) % valFreq == 0 and trainIter >= startVal):
                estotal, best_steps = self.Validate(trainIter + 1)
                train_db_estotal, _ = ray.get(self.bpm.get_scorer_best_by_db.remote())
                self.writer.add_scalar('Others/Train Set Best Plan WRL', estotal / self.baselineTrain, trainIter + 1)
                self._log_train_db_best_plan_wrl(train_db_estotal, trainIter + 1)
                self._validate_train_with_test_phase_if_enabled(trainIter + 1)
                self.Validate(trainIter + 1, isTest = True, record_cross_db = False)
                self._validate_mix_plus_data_drift_if_enabled(trainIter + 1)
                wrl_test, wrl_train, gmrl_test, gmrl_train, speedup_test, speedup_train = self.anaManger.recordMetric(trainIter + 1)
                self.anaManger.writeout()
            if trainIter <= 4 or self._should_save_model(wrl_test, wrl_train, gmrl_test, gmrl_train, speedup_test, speedup_train):
                wrl_train = round(wrl_train, 4)
                wrl_test = round(wrl_test, 4)
                gmrl_train = round(gmrl_train, 4)
                gmrl_test = round(gmrl_test, 4)
                self.SavePlanner()
                self.predictor.SaveModel.remote()
                first_start_time,end_time = self.genConfig.SaveTrainInfo(trainIter+1, wrl_train, wrl_test, gmrl_train, gmrl_test,start_time=self.start_time,first_save=(trainIter==startIter))
                save_info = False
                self.best_model_wrl = wrl_test
                self.best_model_gmrl = gmrl_test
                print(f'---------- Model update ----------')
            self.SavePlanner(save_tmp=True)
            if save_info:
                first_start_time,end_time = self.genConfig.SaveTrainInfo(trainIter+1,start_time=self.start_time,first_save=(trainIter==startIter))
        print(f'---------- {totalIter} training iterations completed, first start time: {first_start_time}, current start time: {self.start_time.strftime("%Y/%m/%d %H.%M.%S")}, end time: {end_time.strftime("%Y/%m/%d %H.%M.%S")} ----------')
        print(f'---------- Current training elapsed time: {self.genConfig.GetTotalTime()[1]}, total training elapsed time: {self.genConfig.GetTotalTime()[0]} ----------')
        ray.get(self.bpm.update_schedule.remote(True))
        self.SelfCheck(totalIter=totalIter)

    def LoadExperiencePool(self, buffer_path):
        print('Loading experience pool.')
        if os.path.exists(buffer_path):
            tmp_buffer_file = open(buffer_path,"r")
            lines = tmp_buffer_file.readlines()
            tmp_buffer_file.close()
            experience_num = 0
            for line in lines:
                data = json.loads(line)
                db = data[0]
                query_id = data[1]
                cardinality = data[2]
                opti_hint = data[3]
                latency = data[4]
                step = data[5]
                sql = self.queryManager.get2all(db,query_id)
                if sql is not None:
                    feature_dict,_,_,_ = ray.get(self.planhelper.GetFeature.remote(opti_hint, sql, True, db, query_id = query_id))
                    feature_dict['steps'] = np.array([step])
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
                        )
                        bestfeature, bestexec = ray.get(self.bpm.get_scorer_esbest.remote(db,query_id))
                        Advbybest = (bestexec - latency[0]) / bestexec
                        if Advbybest >= self.genConfig.splitpoint[-1]:
                            self.bpm.update_scorer_esbest.remote(db,query_id, deepcopy(feature_dict),latency[0])
                    experience_num += 1
            print(f'Loaded {experience_num} experience records. Experience pool load complete.')
        else:
            raise FileNotFoundError(f"Experience pool file does not exist. Please verify the path: {buffer_path}")

    def InitBestPlanPool(self):
        if not getattr(self.genConfig, 'use_best_plan_pool', False):
            return
        initialized = ray.get(self.bpm.initialize_best_plan_records.remote())
        bootstrapped = 0
        if getattr(self.genConfig, 'bootstrap_best_plan_from_latency', False):
            bootstrapped = ray.get(self.bpm.bootstrap_best_plan_from_latencybuffer.remote())
        print(f'best-plan  pool initialization complete: seed={initialized}, bootstrap_updates={bootstrapped}, path={self.genConfig.best_plan_path}')

    def LoadBestPlanPool(self, best_plan_path):
        if not getattr(self.genConfig, 'use_best_plan_pool', False):
            return
        print('Loading best-plan pool.')
        if not os.path.exists(best_plan_path):
            print(f'best-plan pool file does not exist. Skip: {best_plan_path}')
            return
        with open(best_plan_path, 'r', encoding='utf-8') as f:
            best_plan_records = json.load(f)
        loaded = 0
        for db, query_map in best_plan_records.items():
            for query_id, record in query_map.items():
                sql = self.queryManager.get2all(db, query_id)
                if sql is None:
                    continue
                hint = record.get('hint', '')
                latency = record.get('latency', None)
                if latency is None:
                    continue
                timeout = bool(record.get('timeout', False))
                step = int(record.get('step', 0))
                cardinality = record.get('cardinality', -1)
                feature_dict, _, _, _ = ray.get(self.planhelper.GetFeature.remote(hint, sql, True, db, query_id=query_id))
                feature_dict['steps'] = np.array([float(step) / max(float(self.genConfig.maxsteps), 1.0)], dtype=np.float32)
                if self.learner:
                    baseline_plan = self.baseline_info.get(db+'|'+query_id, {}).get('planning_time', 0.0) if hasattr(self, 'baseline_info') else 0.0
                    self.learner.CollectSample.remote(
                        db,
                        query_id,
                        feature_dict,
                        float(latency),
                        cardinality,
                        timeout,
                        planning_time_cum=0.0 if hint != '' else baseline_plan,
                        baseline_planning_time=baseline_plan,
                    )
                    _, bestexec = ray.get(self.bpm.get_scorer_esbest.remote(db, query_id))
                    if float(latency) + 1e-9 < float(bestexec):
                        self.bpm.update_scorer_esbest.remote(db, query_id, deepcopy(feature_dict), float(latency))
                loaded += 1
        print(f'Loaded {loaded} best-plan experience records. best-plan pool load complete.')
    
    def SelfCheck(self, totalIter=1):
        print(f'---------- Start self-check ----------')
        if not os.path.exists(self.genConfig.scorer_path):
            raise FileNotFoundError(f"Self-check failed. Scorer model not found: {self.genConfig.scorer_path}")
        agent_path = os.path.join(self.genConfig.agent_checkpoint,'rllib_checkpoint.json')
        if not os.path.exists(agent_path):
            raise FileNotFoundError(f"Self-check failed. Planner model not found: {agent_path}")
        try:
            self.planner = None
            self.LoadPlanner(self.genConfig.agent_checkpoint)
            self.predictor.LoadModel.remote(scorer_path = self.genConfig.scorer_path)
            self.genConfig.opti_result_path = os.path.join(self.genConfig.opti_result_path,self.genConfig.time)
            self.GetPlanOptiResult(self.queryManager.test_path, train_iter = totalIter)
        except Exception as e:
            raise Exception(f'Self-check failed. Error: {e}')
        print(f'---------- Model self-check completed. Stored at: {os.path.abspath(os.path.dirname(self.genConfig.scorer_path))} ----------')

    def GetPlanOptiResult(self, sql_dir, exe = True, train_iter = None):
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
        sql_files.sort()
        sql_num = len(sql_files)
        save_sql_dir = os.path.abspath(sql_dir)
        opti_num = 0
        query_id_list = []
        opti_necessity_list = []
        sql_list = []
        opti_result_list = []
        plan_time_list = []
        no_opti_latency_list = []
        opti_latency_list = []
        result_latency_list = []
        opti_rate_list = []
        for db in self.queryManager.test_database:
            for sql_file in sql_files:
                with open(os.path.join(sql_dir, sql_file), 'r') as f:
                    sql = f.read()
                actions = {}
                query_id = os.path.splitext(sql_file)[0]
                sqlinfo = {'sql':sql, 'database':db, 'query_id':query_id}
                start_time = time.time()
                try:
                    obs, info = self.evalEnv.reset(options = sqlinfo)
                except:
                    raise Exception(f"Inference failed. Please verify SQL {query_id}.sql is compatible with this model.")
                breakdown = {
                    'use_dynahint': bool(info.get('useDynaHint', False)),
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
                hint_feature = [(info['hint'], deepcopy(obs[0]))]
                steps = 1
                if info['useDynaHint']:
                    done = False
                    while not done:
                        actions.clear()
                        try:
                            action_start = time.time()
                            for i in range(self.genConfig.num_policies):
                                actions[i] = self.planner.compute_single_action(obs[i], policy_id = 'policy_{}'.format(i), explore = False)
                            breakdown['action_ms'] += (time.time() - action_start) * 1000.0
                        except:
                            raise Exception(f"Inference failed while calling the planner model!\n \
                                            Please check:\n \
                                            1. Planner model directory [{self.genConfig.agent_checkpoint}] exists")
                        env_step_start = time.time()
                        obs, reward, terminated, _, info_all = self.evalEnv.step(actions)
                        breakdown['env_step_ms'] += (time.time() - env_step_start) * 1000.0
                        copy_start = time.time()
                        for k in info_all:
                            if info_all[k].get('record_candidate', True):
                                hint_feature.append((info_all[k]['hint'],deepcopy(obs[k])))
                            if info_all[k].get('stop_taken', False):
                                breakdown['stop_taken'] = True
                                breakdown['stop_step'] = int(info_all[k].get('stop_step', self.genConfig.maxsteps))
                            terminate_reason = info_all[k].get('terminate_reason', '')
                            if terminate_reason:
                                breakdown['terminate_reason'] = terminate_reason
                        breakdown['obs_copy_ms'] += (time.time() - copy_start) * 1000.0
                        breakdown['loop_count'] += 1
                        breakdown['candidate_count'] = len(hint_feature)
                        steps += 1
                        if terminated['__all__'] == True:
                            if not breakdown['terminate_reason']:
                                breakdown['terminate_reason'] = 'maxsteps'
                            done = True
                    predict_start = time.time()
                    optimal_hint,optimal_feature,_ = ray.get(self.predictor.GetPrediction.remote(hint_feature, phase='infer'))
                    breakdown['predict_ms'] += (time.time() - predict_start) * 1000.0
                else:
                    optimal_hint = ''
                plan_time = round((time.time() - start_time)*1000, 3)
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
                if exe:
                    if info['useDynaHint']:
                        print(f"Running {db} for SQL inference on database {query_id}")
                    query_id_list.append(db+'|'+query_id)
                    sql_list.append(sql)
                    plan_time = 0
                    plan_time_list.append(plan_time)
                    no_opti_latency,_,_ = ray.get(self.planhelper.GetLatency.remote('', sql, db, query_id, use_buffer = False))
                    no_opti_latency_list.append(no_opti_latency[0])
                    if optimal_hint!='':
                        opti_latency,_,_ = ray.get(self.planhelper.GetLatency.remote(optimal_hint, sql, db, query_id, use_buffer = False, timeout = (no_opti_latency[0]-plan_time)))
                        if (opti_latency[0]+plan_time) < no_opti_latency[0] * (1 - self.genConfig.recommend_threshold):
                            result_latency = opti_latency
                            opti_num += 1
                            opti_necessity_list.append('Y')
                            opti_latency_list.append(opti_latency[0])
                            result_latency_list.append(result_latency[0])
                            # opti_rate_list.append(f"{round((1-(opti_latency[0]+plan_time) / no_opti_latency[0])*100, 2)}%")
                            opti_rate_list.append(f"{round((no_opti_latency[0] / (opti_latency[0]+plan_time))*100, 2)}%")
                            if self.genConfig.DBMS == 'postgres':
                                opti_result_list.append(optimal_hint + sql)
                            # elif self.genConfig.DBMS == 'opengauss' or self.genConfig.DBMS == 'gaussdb':
                            # elif self.genConfig.DBMS == 'hive':
                            #     from Hive.hive_helper import HiveHelper
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
                'QueryId': query_id_list,
                'RequiresOptimization': opti_necessity_list,
                'LatencyBeforeMs': no_opti_latency_list,
                'LatencyAfterMs': opti_latency_list,
                'ImprovementRate': opti_rate_list,
                'OptimizedSQL': opti_result_list
            }
            df = pd.DataFrame(opti_data)
            df = df.sort_values(by='RequiresOptimization', key=lambda x: x.map({'Y': 0, 'N': 1}))

            no_opti_latency_sum = sum(no_opti_latency_list)
            result_latency_sum = sum(result_latency_list)
            plan_time_sum = sum(plan_time_list)
            # total_opti_rate = round((no_opti_latency_sum - result_latency_sum - plan_time_sum) / no_opti_latency_sum * 100, 2)
            total_opti_rate = round((no_opti_latency_sum / (result_latency_sum + plan_time_sum))*100, 2)
            if total_opti_rate < 0:
                total_opti_rate = 0
            summary_data = pd.DataFrame({
                'OverallImprovement': [f"{total_opti_rate}%"]
            })

            plan_end_time = datetime.now().strftime('%Y/%m/%d %H.%M.%S')
            start_data = pd.DataFrame({
                'TestStartTime': [plan_start_time],
                'TestEndTime': [plan_end_time],
                'TestSQLPath': [save_sql_dir],
                'OptimizableSQLCount|Total': [f"{opti_num} | {sql_num}"],
                'OptimizationCoverage': [f"{round(opti_num/sql_num*100, 2)}%"]
            })
            df_sorted = pd.DataFrame(df).reset_index(drop=True)
            final_df = pd.concat([start_data.reset_index(drop=True), df_sorted, summary_data.reset_index(drop=True)], axis=1)
            final_df.to_csv(result_path, index=False, encoding='utf-8-sig') 
            print(f"Inference completed. Overall improvement: {total_opti_rate}%. Other results were saved to: {os.path.abspath(result_path)}")
        self.anaManger.writeout()

    
    def AutoGetParam(self):
        if not is_encoding_cache_ready(self.genConfig.encoding_path):
            raise RuntimeError(f"encoding cache is incomplete before AutoGetParam: {self.genConfig.encoding_path}")
        with open(self.genConfig.encoding_path, 'r') as encoding_file:
            encoding_json = json.load(encoding_file)
        if not os.path.exists(self.genConfig.auto_config):
            raise FileNotFoundError(f"auto_config missing before AutoGetParam: {self.genConfig.auto_config}")
        with open(self.genConfig.auto_config, 'r') as config_file:
            config_data = json.load(config_file)
        required_base_keys = ("maxjoins", "filtmaxnum", "maxnode", "heightsize", "maxpos")
        missing_keys = [key for key in required_base_keys if key not in config_data]
        if missing_keys:
            raise RuntimeError(f"auto_config is incomplete before AutoGetParam: missing {missing_keys}")
        config_data["types"] = len(encoding_json['type2idx'])+2
        config_data["columns"] = len(encoding_json['col2idx'])+2
        config_data["tablenum"] = len(encoding_json['table2idx'])+1
        config_data["opsnum"] = len(encoding_json['op2idx'])+2
        config_data["db_hist_bins_per_col"] = self.genConfig.db_hist_bins_per_col
        config_data["node_hist_dim"] = 3 * (self.genConfig.db_hist_bins_per_col - 1)
        config_data["db_hist_dim"] = self.genConfig.db_hist_bins_per_col*(len(encoding_json['col2idx'])+2)
        config_data["query_feat_dim"] = len(encoding_json['table2idx'])+2
        config_data["num_node_feature"] = 7 + config_data["maxjoins"] + 5 * config_data["filtmaxnum"] + config_data["node_hist_dim"]
        config_data["AutoGetParam"] = 1
        print("Auto parameter detection completed.")

        write_json_atomic(self.genConfig.auto_config, config_data, ensure_ascii=False, indent=4)
        for key, value in config_data.items():
            setattr(self.genConfig, key, value)
        
    def SavePlanner(self,save_tmp = False):
        path = self.tmp_planner_path if save_tmp else self.genConfig.agent_checkpoint
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
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
                "filtmaxnum":self.genConfig.filtmaxnum,
                "maxpos":self.genConfig.maxpos,
                "num_node_feature":self.genConfig.num_node_feature,
                "db_hist_dim":self.genConfig.db_hist_dim,
                "query_feat_dim":self.genConfig.query_feat_dim,
                "AutoGetParam":0
            }
            write_json_atomic(self.genConfig.auto_config, config_data, ensure_ascii=False, indent=4)

    def LoadPlanner(self,planner_path,load_tmp = False):
        if load_tmp:
            self.planner = Algorithm.from_checkpoint(self.tmp_planner_path)
        else:
            self.planner = Algorithm.from_checkpoint(planner_path)

    def Close(self):
        self.anaManger.close()  
        ray.shutdown()

if __name__ == "__main__":
    SupportDBMS = ['gaussdb', 'opengauss', 'hive', 'postgres']
    SupportWorkload = ['JOBRand', 'TPCH', 'STATS']

    parser = argparse.ArgumentParser("DynaHint Controller")
    parser.add_argument("--DBMS",choices = SupportDBMS,
                        help="Choose the DBMS from [GaussDB, openGauss, Hive, postgres]")
    parser.add_argument("--Workload",choices = SupportWorkload,
                        help="Choose the Workload from [JOBRand, TPCH, STATS]")
    parser.add_argument("--ExpName",
                        help="The experiment name must be unique")
    parser.add_argument("--Database",
                        help="The Database Name")
    parser.add_argument("--Seed", type=int, default = None,
                        help="Random Seed")
    parser.add_argument("--Maxsteps", type=int, default = None,
                        help="Max steps of agent")
    parser.add_argument("--Maxsamples", type=int, default = None,
                        help="Maximum number of samples in a single iteration (in best plan manager)")
    parser.add_argument("--TotalIter", type=int, default = None,
                        help="The total number of iterations for which the planner is trained")
    parser.add_argument("--ValidateFreq", type=int, default = None,
                        help="The frequency of validation")
    parser.add_argument("--Agents", type=int, default = None,
                        help="The Num of Agents.")
    parser.add_argument("--PenaltyCoeff", type=int, default = None,
                        help="The coefficient of penalty.")
    parser.add_argument("--SampleStrategy", type=str, default = None,
                        help="The strategy of sampling.")
    parser.add_argument("--HybridSampleThreshold", type=float, default = None,
                        help="The probability threshold used by hybrid sampling.")
    parser.add_argument("--NotUpdateScorer", action='store_true',
                        help="Whether update scorer or not.")
    parser.add_argument("--OffLeftDeep", action='store_true',
                        help="Remove the left-deep restriction")
    parser.add_argument("--ScorerPath", type=str, default = None,
                        help="The model path of Scorer.")
    parser.add_argument("--OptiSQLDir", type=str, default = None,
                        help="The SQL address of model inference.")
    parser.add_argument("--OptiModelPath", type=str, default = None)
    parser.add_argument("--NotAutoGetParam", action='store_true')
    parser.add_argument("--ContinueTrain", type=str, default = None)
    args = parser.parse_args()

    config = Config()
    arg_to_config = {
        'DBMS': 'DBMS',
        'Workload': 'mode',
        'Seed': 'seed',
        'ExpName': 'expname',
        'Database': 'database',
        'Maxsteps': 'maxsteps',
        'Maxsamples': 'maxsamples',
        'TotalIter': 'total_iter',
        'ValidateFreq': 'val_freq',
        'Agents': ('num_agents', 'num_policies'),
        'PenaltyCoeff': 'penalty_coeff',
        'SampleStrategy': 'sample_strategy',
        'HybridSampleThreshold': 'hybrid_sample_threshold'
    }

    for arg, config_attr in arg_to_config.items():
        arg_value = getattr(args, arg, None)
        if arg_value is not None:
            if isinstance(config_attr, tuple):
                for attr in config_attr:
                    setattr(config, attr, arg_value)
            else:
                setattr(config, config_attr, arg_value)
    # if config.DBMS == 'hive':
    #     from Hive.hive_planer import RemotePlanHelper
    #     from Hive.hive_env import DynaHintEnvTrain, DynaHintEnvTest
    # else:
    from planhelper import RemotePlanHelper
    from DynaHintEnv import DynaHintEnvTrain, DynaHintEnvTest
    if args.NotUpdateScorer:
        config.update_scorer = False
    if args.OffLeftDeep:
        config.left_deep_restriction = False
    if not args.ExpName and not config.expname:
        config.expname = f"{config.seed}_{args.Maxsteps}"
    config.ConfirmPath()
    if args.ScorerPath:
        config.scorer_path = args.ScorerPath
        
    if args.NotAutoGetParam :
        config.AutoGetParam = False
        if os.path.exists(config.auto_config):
            config_data = json.load(open(config.auto_config, 'r'))
            config_data["AutoGetParam"] = 0
            write_json_atomic(config.auto_config, config_data, ensure_ascii=False, indent=4)
    
    if args.OptiSQLDir is not None:
        config.CpuDeployment(type = 'test')
        config.update_scorer = False
        if args.OptiModelPath is None:
            model_path = config.LatestModelDir(train_mode=config.train_mode)
            if model_path is None:
                raise FileNotFoundError(f"No model directory available for inference: {config.model_root}")
            config.opti_result_path = os.path.join(config.opti_result_path, os.path.basename(model_path))
        else:
            model_path = config.ResolveModelDir(args.OptiModelPath)
            config.opti_result_path = os.path.join(config.opti_result_path, os.path.basename(model_path))
        print(f'Using model [{os.path.abspath(model_path)}] for inference...')
        config.agent_checkpoint = os.path.join(model_path,'planner/')
        config.scorer_path = os.path.join(model_path,'scorer.pt')
        config.auto_config = os.path.join(model_path,'auto_config.json')
        if os.path.exists(config.auto_config):
            config_data = json.load(open(config.auto_config, 'r'))
            config.AutoGetParam = config_data['AutoGetParam']
        dynahint = DynaHint(config, config.AutoGetParam, istrain=False)
        dynahint.BuildPlanner(train_batch_size = config.planner_batch_size, workersNum = config.PlannerworkersNum, planner_path = config.agent_checkpoint)
        dynahint.GetPlanOptiResult(args.OptiSQLDir, exe=True)
    else:
        StartIter = 0
        if args.ContinueTrain is not None:
            model_path = config.ResolveModelDir(args.ContinueTrain)
            train_info_path = os.path.join(model_path, 'trainfo.json')
            config.CpuDeployment(type = 'continue', print_info = False)
            StartIter, model_wrl_train, model_gmrl_train = config.LoadTrainInfo(train_info_path)
            if StartIter<config.total_iter:
                print(f'Already trained {StartIter} iterations. Resume from iteration {StartIter+1} ...')
                dynahint = DynaHint(config, config.AutoGetParam, isFirst=True, best_model_wrl = model_wrl_train, best_model_gmrl = model_gmrl_train)
                print("Running Baseline......")
                dynahint.RunBaseline(startIter = StartIter)
                dynahint.InitBestPlanPool()
                dynahint.LoadExperiencePool(config.latency_buffer_path)
                dynahint.LoadBestPlanPool(config.best_plan_path)
                print("Initialing Planner......")
                dynahint.BuildPlanner(train_batch_size = config.planner_batch_size, workersNum = config.PlannerworkersNum, planner_path = config.agent_checkpoint)
            else:
                print(f'Already trained {StartIter} iterations. The configured maximum training iterations are {config.total_iter}. Please update the config before resuming training!')
                sys.exit(0)
        else:
            config.CpuDeployment(type = 'train')
            dynahint = DynaHint(config, config.AutoGetParam, isFirst=True)
            print("Running Baseline......")
            dynahint.RunBaseline()
            dynahint.InitBestPlanPool()
            dynahint.LoadBestPlanPool(config.best_plan_path)
            print("Initialing Planner......")
            dynahint.BuildPlanner(train_batch_size = config.planner_batch_size, workersNum = config.PlannerworkersNum)
            if config.update_scorer:
                initialPro = config.initialPro
                print("Initialing Scorer........")
                dynahint.InitScorer(initialPro = initialPro)
                time.sleep(10)
        if config.update_scorer:
            print("Training DynaHint.........")
            dynahint.Run(startIter=StartIter, totalIter = config.total_iter, valFreq = config.val_freq)
        else:
            dynahint.Sim(startIter=StartIter, totalIter = config.total_iter, valFreq = config.val_freq)
    dynahint.Close()
