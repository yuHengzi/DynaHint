import random
import json
import os
from planhelper import PlanHelper
from encoding import is_encoding_cache_ready, write_json_atomic
import numpy as np
from copy import deepcopy
import pandas as pd
import ray
import time
import math
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from util import get_median
class QueryManager:
    def __init__(self, globalconfig, planhelper = None, isremote = True, isfirst = False):
        self.isremote = isremote
        if planhelper != None:
            self.planhelper = planhelper
            self.isremote_1 = isremote
        else:
            self.planhelper = PlanHelper(globalconfig)
            self.isremote_1 = False
        self.config      = globalconfig
        original_train_path = self.config.train_workload_path
        original_test_path = self.config.test_workload_path
        original_train_dir = os.listdir(original_train_path)
        original_test_dir = os.listdir(original_test_path)
        self.train_path  = original_train_path
        self.test_path   = original_test_path
        self.train_dir   = original_train_dir
        self.test_dir    = original_test_dir
        self.trainSet    = pd.DataFrame(columns=['tablenum','sql','base_train_feature'])
        self.validateSet = pd.DataFrame(columns=['sql'])
        self.testSet     = pd.DataFrame(columns=['sql'])
        self.extraEvalSets = {}
        self.extraEvalCursors = {}
        self.allSet      = pd.DataFrame(columns=['sql'])
        self.scorer_esbest_feature = {}
        self.RL_esbest_feature  = {}
        self.median_feature     = {}
        self.buffer             = []
        databases_num = len(self.config.databases)
        train_databases_num = int(databases_num*self.config.train_data_rate)
        if self.config.train_mode == 'data_drift':
            self.train_database = self.config.databases[:train_databases_num]
            self.test_database  = self.config.databases[train_databases_num:]
            workload_source = getattr(self.config, 'data_drift_workload_source', 'test')
            if workload_source == 'test':
                self.train_dir       = self.test_dir
                self.train_path      = self.test_path
            elif workload_source == 'train':
                self.test_dir        = self.train_dir
                self.test_path       = self.train_path
            else:
                raise ValueError(
                    "data_drift_workload_source must be 'train' or 'test', "
                    f"got {workload_source!r}"
                )
        elif self.config.train_mode == 'query_drift':
            self.train_database = [self.config.databases[0]]
            self.test_database  = [self.config.databases[0]]
        elif self.config.train_mode in ('mix', 'mix+'):
            self.train_database = self.config.databases[:train_databases_num]
            self.test_database  = self.config.databases[train_databases_num:]
            if self.config.train_mode == 'mix+':
                workload_source = getattr(self.config, 'data_drift_workload_source', 'train')
                if workload_source == 'train':
                    self.train_dir = original_train_dir
                    self.train_path = original_train_path
                    self.test_dir = original_test_dir
                    self.test_path = original_test_path
                    self.mix_plus_data_drift_dir = original_train_dir
                    self.mix_plus_data_drift_path = original_train_path
                elif workload_source == 'test':
                    self.train_dir = original_test_dir
                    self.train_path = original_test_path
                    self.test_dir = original_train_dir
                    self.test_path = original_train_path
                    self.mix_plus_data_drift_dir = original_test_dir
                    self.mix_plus_data_drift_path = original_test_path
                else:
                    raise ValueError(
                        "data_drift_workload_source must be 'train' or 'test', "
                        f"got {workload_source!r}"
                    )
        else:
            print('The train_mode configuration is incorrect!')

        if globalconfig.AutoGetParam and isfirst:
            if (not self._auto_config_ready()) or (not is_encoding_cache_ready(self.config.encoding_path)):
                print("Auto-detecting parameters...")
                self.AutoGetParam()

        for db in self.train_database:
            for train_dir in self.train_dir:
                with open(self.train_path + train_dir,'r') as f:
                    sql = f.read()
                query_id = train_dir.split('.')[0]
                self.buffer.append((db+'|'+query_id))
                
                self.validateSet.loc[db+'|'+query_id] = [sql]
                self.allSet.loc[db+'|'+query_id] = [sql]
                if self.isremote_1:
                    feature_dict,hintdict,left_deep,cost_plan_json = ray.get(self.planhelper.GetFeature.remote('',sql,True,db,query_id = query_id,is_encode =False))
                else:
                    feature_dict,hintdict,left_deep,cost_plan_json = self.planhelper.get_feature('',sql,True,db,query_id = query_id,is_encode =False)
        for db in self.test_database:
            for test_dir in self.test_dir:
                with open(self.test_path + test_dir,'r') as f:
                    sql = f.read()
                query_id = test_dir.split('.')[0]
                self.testSet.loc[db+'|'+query_id] = [sql]
                self.allSet.loc[db+'|'+query_id] = [sql]
        if self.config.train_mode == 'mix+':
            test_data_drift_set = pd.DataFrame(columns=['sql'])
            for db in self.test_database:
                for train_dir in self.mix_plus_data_drift_dir:
                    with open(self.mix_plus_data_drift_path + train_dir,'r') as f:
                        sql = f.read()
                    query_id = train_dir.split('.')[0]
                    test_data_drift_set.loc[db+'|'+query_id] = [sql]
            self.extraEvalSets['test_data_drift'] = test_data_drift_set
            self.extraEvalCursors['test_data_drift'] = 0
        self.cur_test = 0
        self.numOfTest = len(self.testSet.index)
        self.cur_validate = 0
        self.numOfvalidate = len(self.validateSet.index)
        self._ensure_encoding_cache()

    def _ensure_encoding_cache(self):
        if not is_encoding_cache_ready(self.config.encoding_path):
            encoding_dir = os.path.dirname(self.config.encoding_path)
            if encoding_dir and not os.path.exists(encoding_dir):
                os.makedirs(encoding_dir, exist_ok=True)
            if self.isremote_1:
                save_ref = self.planhelper.SaveEncoding.remote(self.config.encoding_path)
                if isinstance(save_ref, ray.ObjectRef):
                    ray.get(save_ref)
            else:
                self.planhelper.encoding.save_to_file(self.config.encoding_path)
        if not is_encoding_cache_ready(self.config.encoding_path):
            raise RuntimeError(f"encoding cache is incomplete after bootstrap: {self.config.encoding_path}")

    def _auto_config_ready(self):
        if not os.path.exists(self.config.auto_config):
            return False
        try:
            with open(self.config.auto_config, 'r') as config_file:
                config_data = json.load(config_file)
        except Exception:
            return False
        required_keys = (
            "AutoGetParam",
            "heightsize",
            "maxnode",
            "maxjoins",
            "filtmaxnum",
            "maxpos",
            "node_hist_dim",
            "num_node_feature",
        )
        return all(key in config_data for key in required_keys) and int(config_data.get("AutoGetParam", 0)) == 1

    def AutoGetParam(self):
        # if self.config.DBMS == 'opengauss' or self.config.DBMS == 'gaussdb' or self.config.DBMS == 'postgres':
        #     from planhelper import PlanHelper
        # elif self.config.DBMS == 'hive':
        #     from Hive.hive_planer import PlanHelper
        if self.config.DBMS == 'postgres':
            from planhelper import PlanHelper
        else:
            raise ValueError('DBMS not supported')
        planhelper = PlanHelper(self.config)
        max_joinNum = 0
        max_filterNum = 0
        max_pos = 0
        max_heights = 0
        max_nodeNum = 0
        for db in self.train_database:
            for dir in self.train_dir:
                with open(self.train_path + dir,'r') as f:
                    sql = f.read()
                query_id = dir.split('.')[0]
                sqlparam = planhelper.get_param('',sql,True,db,query_id = query_id)
                max_joinNum = max(max_joinNum,sqlparam[0])
                max_filterNum = max(max_filterNum,sqlparam[1])
                max_pos = max(max_pos,sqlparam[2])
                max_heights = max(max_heights,sqlparam[3])
                max_nodeNum = max(max_nodeNum,sqlparam[4])
        for db in self.test_database:
            for dir in self.test_dir:
                with open(self.test_path + dir,'r') as f:
                    sql = f.read()
                query_id = dir.split('.')[0]
                sqlparam = planhelper.get_param('',sql,True,db,query_id = query_id)
                max_joinNum = max(max_joinNum,sqlparam[0])
                max_filterNum = max(max_filterNum,sqlparam[1])
                max_pos = max(max_pos,sqlparam[2])
                max_heights = max(max_heights,sqlparam[3])
                max_nodeNum = max(max_nodeNum,sqlparam[4])
        config_data = {}
        config_data["AutoGetParam"] = 1
        config_data["heightsize"]=max_heights+10
        config_data["maxnode"]=max_nodeNum+10
        config_data["maxjoins"]=max_joinNum+5
        config_data["filtmaxnum"]=max_filterNum+1
        config_data["maxpos"]=max_pos+2+1
        config_data["node_hist_dim"] = 3 * (self.config.db_hist_bins_per_col - 1)
        config_data["num_node_feature"] = 7 + config_data["maxjoins"] + 5 * config_data["filtmaxnum"] + config_data["node_hist_dim"]
        write_json_atomic(self.config.auto_config, config_data, ensure_ascii=False, indent=4)
    
    def creat_trainSet(self):
        for db in self.train_database:
            for train_dir in self.train_dir:
                query_id = train_dir.split('.')[0]
                sql = self.allSet.loc[db+'|'+query_id]['sql']
                if self.isremote_1:
                    feature_dict,hintdict,left_deep,cost_plan_json = ray.get(self.planhelper.GetFeature.remote('',sql,True,db,query_id = query_id))
                else:
                    feature_dict,hintdict,left_deep,cost_plan_json = self.planhelper.get_feature('',sql,True,db,query_id = query_id)
                feature_dict['steps']   = np.array([0])
                # self.buffer.extend([query_id] * len(hintdict['join order']))
                cost_plan_json['steps'] = 0
                self.trainSet.loc[db+'|'+query_id] = [len(hintdict['join order']), sql, (feature_dict, hintdict, cost_plan_json)]
        self.numOfTrain = len(self.trainSet.index)

    def get2eval(self, phase='test'):
        if phase not in (None, 'test'):
            eval_set = self.extraEvalSets[phase]
            cursor = self.extraEvalCursors.get(phase, 0)
            db_and_queryid = eval_set.index[cursor]
            db,query_id = db_and_queryid.split('|')
            cursor = (cursor + 1) % len(eval_set.index)
            self.extraEvalCursors[phase] = cursor
            oneloop = (cursor == 0)
            return eval_set.loc[db_and_queryid,'sql'], db, query_id, oneloop
        db_and_queryid = self.testSet.index[self.cur_test]
        db,query_id = db_and_queryid.split('|')
        oneloop = False
        self.cur_test = (self.cur_test + 1) % self.numOfTest
        if self.cur_test == 0:
            oneloop = True
        return self.testSet.loc[db_and_queryid,'sql'], db, query_id, oneloop

    def get_eval_keys(self, phase='test'):
        if phase in (None, 'test'):
            return list(self.testSet.index)
        return list(self.extraEvalSets.get(phase, pd.DataFrame()).index)
    
    def get2validate(self):
        db_and_queryid = self.validateSet.index[self.cur_validate]
        db,query_id = db_and_queryid.split('|')
        oneloop = False
        self.cur_validate = (self.cur_validate + 1) % self.numOfvalidate
        if self.cur_validate == 0:
            oneloop = True
        return self.validateSet.loc[db_and_queryid,'sql'], db, query_id, oneloop
        
    def get2train(self):
        db,query_id = random.choices(self.buffer)[0].split('|')
        return self.trainSet.loc[db+'|'+query_id,'sql'],self.trainSet.loc[db+'|'+query_id,'base_train_feature'],\
                db,query_id, self.scorer_esbest_feature[db+'|'+query_id], self.median_feature[db+'|'+query_id]
    
    def get2all(self,db,query_id):
        try:
            return self.validateSet.loc[db+'|'+query_id,'sql']
        except:
            return None
    
    def updateBuffer(self,queryImportance):
        self.buffer = []
        for k, v in queryImportance.items():
            self.buffer.extend([k] * v)
    
    def update_scorer_esbest(self,scorer_esbest_feature):
        self.scorer_esbest_feature = deepcopy(scorer_esbest_feature)

    def update_Median(self,median_hint_latency):
        for db_and_queryid,hint_latency in median_hint_latency.items():
            db,query_id = db_and_queryid.split('|')
            if self.isremote:
                feature_dict,_,_,_ = ray.get(self.planhelper.GetFeature.remote(hint_latency['hint'],self.validateSet.loc[db+'|'+query_id,'sql'],False,db,query_id = query_id))
            else:
                feature_dict,_,_,_ = self.planhelper.get_feature(hint_latency['hint'],self.validateSet.loc[db+'|'+query_id,'sql'],False,db,query_id = query_id,source = 'update_Median')
            feature_dict['steps'] = np.array([0])
            self.median_feature[db_and_queryid] = (feature_dict, hint_latency['latency'])

    def update_RL_esbest(self,RL_esbest_feature):
        self.RL_esbest_feature = deepcopy(RL_esbest_feature)
    
class ResultManager():
    def __init__(self, genConfig, writer):
        self.config     = genConfig
        self.resultFile = open(self.config.outfile_path,'w')
        # self.test_expPool    = open(self.config.TestExperiencePool,'w')
        # self.bestplan   = open(self.config.beststeps_record,'a') # TO_Delete
        self.testQuery  = []
        self.trainQuery = []
        self.writer     = writer
        self.ExecutionTime  = {}
        self.PlanningTime   = {}
        self.QueryTimeTrace = {}
        self.queryTimeTraceTrainPath = os.path.join(
            self.config.opti_result_path,
            f'query_time_trace_train_{self.config.expname}.txt',
        )
        self.queryTimeTraceTestPath = os.path.join(
            self.config.opti_result_path,
            f'query_time_trace_test_{self.config.expname}.txt',
        )
        self.queryTimeTraceTrainSafePath = os.path.join(
            self.config.opti_result_path,
            f'query_time_trace_train_safe_{self.config.expname}.txt',
        )
        self.queryTimeTraceTestDataDriftPath = os.path.join(
            self.config.opti_result_path,
            f'query_time_trace_test_data_drift_{self.config.expname}.txt',
        )
        self.ExtraExecutionTime = defaultdict(dict)
        self.ExtraPlanningTime = defaultdict(dict)
        self.CrossDBHintTrace = {}
        self.crossDbHintTracePath = os.path.join(
            self.config.opti_result_path,
            f'cross_db_hint_effect_{self.config.expname}.txt',
        )
        self.GeneralizationDiagnostics = {}
        self.generalizationDiagnosticsPath = os.path.join(
            self.config.opti_result_path,
            f'generalization_diagnostics_{self.config.expname}.txt',
        )
        self.PlanningBreakdownTrace = {}
        self.TrainDbBestWRL = {}
        self.TrainDbBestSpeedup = {}
        self.planningBreakdownPath = os.path.join(
            self.config.opti_result_path,
            f'planning_breakdown_{self.config.expname}.txt',
        )
        self.timeRecord = time.time()

    def _is_aggregate_metric_key(self, db_and_queryid):
        return db_and_queryid in {
            "Train|WRL",
            "Train|GMRL",
            "Train|Speedup",
            "Test|WRL",
            "Test|GMRL",
            "Test|Speedup",
            "TrainSafe|WRL",
            "TrainSafe|GMRL",
            "TrainSafe|Speedup",
            "TestDataDrift|WRL",
            "TestDataDrift|GMRL",
            "TestDataDrift|Speedup",
        }

    def recordQuery(self,db,queryId, istest):
        db_and_queryid = db+'|'+queryId
        if istest:
            self.testQuery.append(db_and_queryid)
        else:
            self.trainQuery.append(db_and_queryid)

    def recordRuning(self,key,value):
        self.resultFile.write(json.dumps([key,value])+"\n")
        self.resultFile.flush()

    # def recordExp(self,queryId, hint, agentNo, steps):
    #     self.test_expPool.write(json.dumps([queryId,'|'.join([hint,str(agentNo),str(steps)])])+"\n")
    #     self.test_expPool.flush()

    def recordTime(self,name):
        _time = str(round(time.time() - self.timeRecord, 3) * 1000)
        self.timeRecord = time.time()
        self.resultFile.write(json.dumps([name,_time])+"\n")
        self.resultFile.flush()

    def recordeval(self,db,queryId, execution_time, planning_time, val_iter = None, phase = None):
        execution_time = round(execution_time, 3)
        planning_time = round(planning_time, 3)
        db_and_queryid = db+'|'+queryId
        if db_and_queryid not in self.ExecutionTime:
            self.ExecutionTime[db_and_queryid] = [execution_time]
            self.PlanningTime[db_and_queryid]  = [planning_time]
        else:
            self.ExecutionTime[db_and_queryid].append(execution_time)
            self.PlanningTime[db_and_queryid].append(planning_time)
        if queryId not in {"WRL", "GMRL", "Speedup"}:
            self.QueryTimeTrace.setdefault(db_and_queryid, [])
            self.QueryTimeTrace[db_and_queryid].append({
                'val_iter': 0 if val_iter is None else val_iter,
                'execution_time': execution_time,
                'planning_time': planning_time,
                'is_baseline': (val_iter in (None, 0)),
                'phase': phase if phase is not None else 'unknown',
            })

    def recordExtraEval(self, phase, db, queryId, execution_time, planning_time, val_iter=None, is_baseline=False):
        phase = str(phase)
        execution_time = round(execution_time, 3)
        planning_time = round(planning_time, 3)
        db_and_queryid = db + '|' + queryId
        self.ExtraExecutionTime[phase].setdefault(db_and_queryid, [])
        self.ExtraPlanningTime[phase].setdefault(db_and_queryid, [])
        self.ExtraExecutionTime[phase][db_and_queryid].append(execution_time)
        self.ExtraPlanningTime[phase][db_and_queryid].append(planning_time)
        if queryId not in {"WRL", "GMRL", "Speedup"}:
            self.QueryTimeTrace.setdefault(db_and_queryid, [])
            has_phase_baseline = any(
                record.get('phase') == phase and record.get('is_baseline', False)
                for record in self.QueryTimeTrace[db_and_queryid]
            )
            if is_baseline:
                self.QueryTimeTrace[db_and_queryid].append({
                    'val_iter': 0 if val_iter is None else val_iter,
                    'execution_time': execution_time,
                    'planning_time': planning_time,
                    'is_baseline': True,
                    'phase': phase,
                })
                return
            if not has_phase_baseline and db_and_queryid in self.ExecutionTime and db_and_queryid in self.PlanningTime:
                self.QueryTimeTrace[db_and_queryid].append({
                    'val_iter': 0,
                    'execution_time': round(float(self.ExecutionTime[db_and_queryid][0]), 3),
                    'planning_time': round(float(self.PlanningTime[db_and_queryid][0]), 3),
                    'is_baseline': True,
                    'phase': phase,
                })
            self.QueryTimeTrace[db_and_queryid].append({
                'val_iter': 0 if val_iter is None else val_iter,
                'execution_time': execution_time,
                'planning_time': planning_time,
                'is_baseline': False,
                'phase': phase,
            })

    def recordPlanningBreakdown(self, db, queryId, val_iter, breakdown, phase='test'):
        db_and_queryid = db + '|' + queryId
        rounded_breakdown = {
            'val_iter': int(val_iter),
            'phase': str(phase),
            'use_dynahint': bool(breakdown.get('use_dynahint', False)),
            'loop_count': int(breakdown.get('loop_count', 0)),
            'candidate_count': int(breakdown.get('candidate_count', 0)),
            'selected_step': int(breakdown.get('selected_step', 0)),
            'stop_taken': bool(breakdown.get('stop_taken', False)),
            'stop_step': int(breakdown.get('stop_step', 0)),
            'terminate_reason': str(breakdown.get('terminate_reason', '')),
            'reset_ms': round(float(breakdown.get('reset_ms', 0.0)), 3),
            'action_ms': round(float(breakdown.get('action_ms', 0.0)), 3),
            'env_step_ms': round(float(breakdown.get('env_step_ms', 0.0)), 3),
            'obs_copy_ms': round(float(breakdown.get('obs_copy_ms', 0.0)), 3),
            'predict_ms': round(float(breakdown.get('predict_ms', 0.0)), 3),
            'other_ms': round(float(breakdown.get('other_ms', 0.0)), 3),
            'planning_total_ms': round(float(breakdown.get('planning_total_ms', 0.0)), 3),
        }
        self.PlanningBreakdownTrace.setdefault(db_and_queryid, [])
        self.PlanningBreakdownTrace[db_and_queryid].append(rounded_breakdown)

    def _summarize_breakdown_records(self, records):
        summary = {'records': len(records)}
        metric_keys = [
            'selected_step',
            'loop_count',
            'candidate_count',
            'reset_ms',
            'action_ms',
            'env_step_ms',
            'obs_copy_ms',
            'predict_ms',
            'other_ms',
            'planning_total_ms',
        ]
        if len(records) == 0:
            for key in metric_keys:
                summary[f'mean_{key}'] = 0.0
                summary[f'max_{key}'] = 0.0
            return summary
        for key in metric_keys:
            values = [float(record.get(key, 0.0)) for record in records]
            summary[f'mean_{key}'] = round(sum(values) / len(values), 3)
            summary[f'max_{key}'] = round(max(values), 3)
        return summary

    def recordwrl(self, valIter):
        dynahint_test = 0
        dynahint_train = 0
        pg_test = 0
        pg_train = 0
        dynahint_test_total = 0
        dynahint_train_total = 0
        pg_test_total = 0
        pg_train_total = 0
        db_wrl_stats = defaultdict(lambda: {
            'train_dynahint': 0,
            'train_pg': 0,
            'test_dynahint': 0,
            'test_pg': 0,
            'train_dynahint_total': 0,
            'train_pg_total': 0,
            'test_dynahint_total': 0,
            'test_pg_total': 0,
        })
        wrl_test = 1.0
        wrl_train = 1.0
        speedup_test = 1.0
        speedup_train = 1.0
        for k,v in self.ExecutionTime.items():
            if not self._is_aggregate_metric_key(k):
                splits = k.split('|')
                if len(splits) != 2:
                    continue
                db, queryid = splits
                if k in self.testQuery:
                    dynahint_test += v[-1]
                    pg_test += v[0]
                    dynahint_test_total += v[-1] + self.PlanningTime[k][-1]
                    pg_test_total += v[0] + self.PlanningTime[k][0]
                    db_wrl_stats[db]['test_dynahint'] += v[-1]
                    db_wrl_stats[db]['test_pg'] += v[0]
                    db_wrl_stats[db]['test_dynahint_total'] += v[-1] + self.PlanningTime[k][-1]
                    db_wrl_stats[db]['test_pg_total'] += v[0] + self.PlanningTime[k][0]
                elif k in self.trainQuery:
                    dynahint_train += v[-1]
                    pg_train += v[0]
                    dynahint_train_total += v[-1] + self.PlanningTime[k][-1]
                    pg_train_total += v[0] + self.PlanningTime[k][0]
                    db_wrl_stats[db]['train_dynahint'] += v[-1]
                    db_wrl_stats[db]['train_pg'] += v[0]
                    db_wrl_stats[db]['train_dynahint_total'] += v[-1] + self.PlanningTime[k][-1]
                    db_wrl_stats[db]['train_pg_total'] += v[0] + self.PlanningTime[k][0]

        if pg_test != 0:
            wrl_test = dynahint_test / pg_test
            self.recordeval("Test","WRL", wrl_test, 0)
            self.writer.add_scalar('Test/WRL',wrl_test,valIter)
        if pg_train != 0:
            wrl_train = dynahint_train / pg_train
            self.recordeval("Train","WRL", wrl_train, 0)
            self.writer.add_scalar('Train/WRL',wrl_train,valIter)
        if dynahint_test_total != 0:
            speedup_test = pg_test_total / dynahint_test_total
            self.recordeval("Test", "Speedup", speedup_test, 0)
            self.writer.add_scalar('Test/Speedup', speedup_test, valIter)
        if dynahint_train_total != 0:
            speedup_train = pg_train_total / dynahint_train_total
            self.recordeval("Train", "Speedup", speedup_train, 0)
            self.writer.add_scalar('Train/Speedup', speedup_train, valIter)
        for db, stat in db_wrl_stats.items():
            train_dynahint = stat['train_dynahint']
            train_pg = stat['train_pg']
            if train_pg != 0:
                wrl_train_db = train_dynahint / train_pg
                self.writer.add_scalar(f'Train/{db}_WRL', wrl_train_db, valIter)
                best_wrl = min(self.TrainDbBestWRL.get(db, float('inf')), wrl_train_db)
                self.TrainDbBestWRL[db] = best_wrl
                self.writer.add_scalar(f'Others/TrainBestWRL/{db}', best_wrl, valIter)
            train_dynahint_total = stat['train_dynahint_total']
            train_pg_total = stat['train_pg_total']
            if train_dynahint_total != 0:
                speedup_train_db = train_pg_total / train_dynahint_total
                self.writer.add_scalar(f'Train/{db}_Speedup', speedup_train_db, valIter)
                best_speedup = max(
                    self.TrainDbBestSpeedup.get(db, 0.0),
                    speedup_train_db,
                )
                self.TrainDbBestSpeedup[db] = best_speedup
                self.writer.add_scalar(f'Others/TrainBestSpeedup/{db}', best_speedup, valIter)
            test_dynahint = stat['test_dynahint']
            test_pg = stat['test_pg']
            if test_pg != 0:
                wrl_test_db = test_dynahint / test_pg
                self.writer.add_scalar(f'Test/{db}_WRL', wrl_test_db, valIter)
            test_dynahint_total = stat['test_dynahint_total']
            test_pg_total = stat['test_pg_total']
            if test_dynahint_total != 0:
                speedup_test_db = test_pg_total / test_dynahint_total
                self.writer.add_scalar(f'Test/{db}_Speedup', speedup_test_db, valIter)
        return wrl_test, wrl_train, speedup_test, speedup_train

    def recordgmrl(self, valIter):
        gmrl_train   = 1
        gmrl_test    = 1
        counts_train = 0
        counts_test  = 0
        db_gmrl_stats = defaultdict(lambda: {'train_prod': 1, 'train_count': 0, 'test_prod': 1, 'test_count': 0})
        for k, v in self.ExecutionTime.items():
            if not self._is_aggregate_metric_key(k):
                splits = k.split('|')
                if len(splits) != 2:
                    continue
                db, queryid = splits
                if k in self.testQuery:
                    gmrl_test = gmrl_test * (v[-1] / v[0])
                    counts_test += 1
                    db_gmrl_stats[db]['test_prod'] *= (v[-1] / v[0])
                    db_gmrl_stats[db]['test_count'] += 1
                elif k in self.trainQuery:
                    gmrl_train = gmrl_train * (v[-1] / v[0])
                    counts_train += 1
                    db_gmrl_stats[db]['train_prod'] *= (v[-1] / v[0])
                    db_gmrl_stats[db]['train_count'] += 1
        if counts_test != 0:
            gmrl_test = pow(gmrl_test, 1 / counts_test)
            self.recordeval("Test","GMRL", gmrl_test, 0)
            self.writer.add_scalar('Test/GMRL',gmrl_test,valIter)
        if counts_train != 0:
            gmrl_train = pow(gmrl_train, 1 / counts_train)
            self.recordeval("Train","GMRL", gmrl_train, 0)
            self.writer.add_scalar('Train/GMRL',gmrl_train,valIter)
        for db, stat in db_gmrl_stats.items():
            # train
            if stat['train_count'] != 0:
                gmrl_train_db = pow(stat['train_prod'], 1 / stat['train_count'])
                self.writer.add_scalar(f'Train/{db}_GMRL', gmrl_train_db, valIter)
            # test
            if stat['test_count'] != 0:
                gmrl_test_db = pow(stat['test_prod'], 1 / stat['test_count'])
                self.writer.add_scalar(f'Test/{db}_GMRL', gmrl_test_db, valIter)
        return gmrl_test, gmrl_train

    def recordExtraPhaseMetric(self, valIter, phase, query_keys, writer_prefix):
        phase = str(phase)
        execution_by_query = self.ExtraExecutionTime.get(phase, {})
        planning_by_query = self.ExtraPlanningTime.get(phase, {})
        if query_keys is None:
            query_keys = sorted(execution_by_query.keys())
        opti_total = 0.0
        pg_total = 0.0
        opti_exec = 0.0
        pg_exec = 0.0
        gmrl_prod = 1.0
        gmrl_count = 0
        db_stats = defaultdict(lambda: {
            'opti_total': 0.0,
            'pg_total': 0.0,
            'opti_exec': 0.0,
            'pg_exec': 0.0,
            'gmrl_prod': 1.0,
            'gmrl_count': 0,
        })
        for key in query_keys:
            if key not in execution_by_query or key not in planning_by_query:
                continue
            if len(execution_by_query[key]) == 0 or len(planning_by_query[key]) == 0:
                continue
            if len(execution_by_query[key]) >= 2 and len(planning_by_query[key]) >= 2:
                baseline_exec = float(execution_by_query[key][0])
                baseline_plan = float(planning_by_query[key][0])
                curr_exec = float(execution_by_query[key][-1])
                curr_plan = float(planning_by_query[key][-1])
            elif phase != 'train_safe':
                baseline_exec = float(execution_by_query[key][0])
                baseline_plan = float(planning_by_query[key][0])
                curr_exec = float(execution_by_query[key][-1])
                curr_plan = float(planning_by_query[key][-1])
            else:
                if key not in self.ExecutionTime or key not in self.PlanningTime:
                    continue
                baseline_exec = float(self.ExecutionTime[key][0])
                baseline_plan = float(self.PlanningTime[key][0])
                curr_exec = float(execution_by_query[key][-1])
                curr_plan = float(planning_by_query[key][-1])
            if baseline_exec <= 0:
                continue
            db = key.split('|', 1)[0]
            opti_total += curr_exec + curr_plan
            pg_total += baseline_exec + baseline_plan
            opti_exec += curr_exec
            pg_exec += baseline_exec
            ratio = curr_exec / baseline_exec
            gmrl_prod *= ratio
            gmrl_count += 1
            db_stats[db]['opti_total'] += curr_exec + curr_plan
            db_stats[db]['pg_total'] += baseline_exec + baseline_plan
            db_stats[db]['opti_exec'] += curr_exec
            db_stats[db]['pg_exec'] += baseline_exec
            db_stats[db]['gmrl_prod'] *= ratio
            db_stats[db]['gmrl_count'] += 1
        wrl = opti_exec / pg_exec if pg_exec > 0 else 1.0
        speedup = pg_total / opti_total if opti_total > 0 else 1.0
        gmrl = pow(gmrl_prod, 1 / gmrl_count) if gmrl_count > 0 else 1.0
        self.writer.add_scalar(f'{writer_prefix}/WRL', wrl, valIter)
        self.writer.add_scalar(f'{writer_prefix}/Speedup', speedup, valIter)
        self.writer.add_scalar(f'{writer_prefix}/GMRL', gmrl, valIter)
        for db, stat in db_stats.items():
            if stat['pg_exec'] > 0:
                self.writer.add_scalar(
                    f'{writer_prefix}/{db}_WRL',
                    stat['opti_exec'] / stat['pg_exec'],
                    valIter,
                )
            if stat['opti_total'] > 0:
                self.writer.add_scalar(
                    f'{writer_prefix}/{db}_Speedup',
                    stat['pg_total'] / stat['opti_total'],
                    valIter,
                )
            if stat['gmrl_count'] > 0:
                self.writer.add_scalar(
                    f'{writer_prefix}/{db}_GMRL',
                    pow(stat['gmrl_prod'], 1 / stat['gmrl_count']),
                    valIter,
                )
        return wrl, gmrl, speedup

    def _record_step_means(self, valIter):
        latest_records = []
        for records in self.PlanningBreakdownTrace.values():
            latest_by_phase = {}
            for record in records:
                phase = record.get('phase', 'test')
                if (phase not in latest_by_phase) or (record['val_iter'] >= latest_by_phase[phase]['val_iter']):
                    latest_by_phase[phase] = record
            latest_records.extend(latest_by_phase.values())
        phase_to_records = defaultdict(list)
        for record in latest_records:
            phase_to_records[record.get('phase', 'test')].append(record)
        phase_prefix = {
            'train': 'Train',
            'test': 'Test',
            'train_safe': 'TrainSafe',
            'test_data_drift': 'TestDataDrift',
            'infer': 'Infer',
        }
        for phase, records in phase_to_records.items():
            if len(records) == 0:
                continue
            prefix = phase_prefix.get(phase, phase)
            mean_selected_step = sum(float(record.get('selected_step', 0)) for record in records) / len(records)
            mean_loop_count = sum(float(record.get('loop_count', 0)) for record in records) / len(records)
            stop_rate = sum(1 for record in records if record.get('stop_taken', False)) / len(records)
            mean_planning_time = sum(float(record.get('planning_total_ms', 0.0)) for record in records) / len(records)
            self.writer.add_scalar(f'{prefix}/SelectedStepMean', mean_selected_step, valIter)
            self.writer.add_scalar(f'{prefix}/PlannerStepLenMean', mean_loop_count, valIter)
            self.writer.add_scalar(f'{prefix}/PlanningTimeMean', mean_planning_time, valIter)
            self.writer.add_scalar(f'{prefix}/StopTakenRate', stop_rate, valIter)
    
    def recordExecutionTime(self,valIter):
        for k, v in self.ExecutionTime.items():
            if not self._is_aggregate_metric_key(k):
                splits = k.split('|')
                if len(splits) != 2:
                    continue
                db, queryid = splits
                if isinstance(v, list) and len(v) > 0 and k in self.PlanningTime and len(self.PlanningTime[k]) > 0:
                    base_exec = v[0]
                    curr_exec = v[-1]
                    if base_exec == 0:
                        continue
                    query_wrl = curr_exec / base_exec
                    tag = f'{db}/{queryid}_WRL'
                    self.writer.add_scalar(tag, query_wrl, valIter)

                    base_total = v[0] + self.PlanningTime[k][0]
                    curr_total = v[-1] + self.PlanningTime[k][-1]
                    if base_total == 0 or curr_total == 0:
                        continue
                    query_speedup = base_total / curr_total
                    tag = f'{db}/{queryid}_Speedup'
                    self.writer.add_scalar(tag, query_speedup, valIter)

    def _format_hint_text(self, hint):
        if hint == '':
            return 'EMPTY_HINT'
        return ' '.join(str(hint).split())

    def _format_ratio_text(self, value):
        if value is None:
            return 'nan'
        if isinstance(value, str):
            return value
        return f'{float(value):.3f}'

    def _write_query_time_trace(self, out_path, phase):
        with open(out_path, "w") as out:
            for db_and_queryid in sorted(self.QueryTimeTrace.keys()):
                records = [record for record in self.QueryTimeTrace[db_and_queryid] if record.get('phase') == phase]
                if len(records) == 0:
                    continue
                baseline = next((record for record in records if record['is_baseline']), None)
                if baseline is None:
                    continue
                baseline_exec = baseline['execution_time']
                baseline_plan = baseline['planning_time']
                baseline_total = baseline_exec + baseline_plan
                line_parts = [f"{db_and_queryid} baseline={baseline_exec:.3f}+{baseline_plan:.3f}"]
                non_baseline_records = [record for record in records if not record['is_baseline']]
                non_baseline_records.sort(key=lambda record: record['val_iter'])
                for record in non_baseline_records:
                    exec_arrow = '↓' if record['execution_time'] < baseline_exec else '↑'
                    curr_total = record['execution_time'] + record['planning_time']
                    plan_arrow = '↓' if curr_total < baseline_total else '↑'
                    line_parts.append(
                        f"{record['val_iter']}={record['execution_time']:.3f}{exec_arrow}+{record['planning_time']:.3f}{plan_arrow}"
                    )
                out.write(' | '.join(line_parts) + '\n')
                out.flush()

    def recordCrossDBHintEffect(self, test_db, query_id, val_iter, test_hint,
                                test_baseline_exec, test_baseline_plan,
                                test_selected_exec, test_selected_plan,
                                cross_hint_results):
        test_key = test_db + '|' + query_id
        self.CrossDBHintTrace[test_key] = {
            'val_iter': val_iter,
            'test_db': test_db,
            'query_id': query_id,
            'test_hint': test_hint,
            'test_baseline_exec': round(test_baseline_exec, 3),
            'test_baseline_plan': round(test_baseline_plan, 3),
            'test_selected_exec': round(test_selected_exec, 3),
            'test_selected_plan': round(test_selected_plan, 3),
            'cross_hint_results': cross_hint_results,
        }

    def recordGeneralizationDiagnostic(self, test_db, query_id, val_iter, diagnostic):
        test_key = test_db + '|' + query_id
        payload = dict(diagnostic)
        payload['val_iter'] = int(val_iter)
        payload['test_db'] = test_db
        payload['query_id'] = query_id
        self.GeneralizationDiagnostics[test_key] = payload

    def recordMetric(self,valIter):
        wrl_test, wrl_train, speedup_test, speedup_train = self.recordwrl(valIter)
        gmrl_test, gmrl_train = self.recordgmrl(valIter)
        if 'train_safe' in self.ExtraExecutionTime:
            self.recordExtraPhaseMetric(valIter, 'train_safe', self.trainQuery, 'TrainSafe')
        if 'test_data_drift' in self.ExtraExecutionTime:
            self.recordExtraPhaseMetric(valIter, 'test_data_drift', None, 'TestDataDrift')
        self._record_step_means(valIter)
        self.recordExecutionTime(valIter)
        return wrl_test, wrl_train, gmrl_test, gmrl_train, speedup_test, speedup_train

    def writeout(self):
        with open(self.config.eval_output_path, "w") as out:
            for k,v in self.ExecutionTime.items():
                out.write(json.dumps([k, v]) + '\n')
                out.flush()
        if not os.path.exists(self.config.opti_result_path):
            os.makedirs(self.config.opti_result_path, exist_ok=True)
        self._write_query_time_trace(self.queryTimeTraceTrainPath, phase='train')
        self._write_query_time_trace(self.queryTimeTraceTestPath, phase='test')
        self._write_query_time_trace(self.queryTimeTraceTrainSafePath, phase='train_safe')
        self._write_query_time_trace(self.queryTimeTraceTestDataDriftPath, phase='test_data_drift')
        with open(self.crossDbHintTracePath, "w") as out:
            for test_key in sorted(self.CrossDBHintTrace.keys()):
                record = self.CrossDBHintTrace[test_key]
                header = (
                    f"test={test_key} | iter={record['val_iter']} | "
                    f"test_hint={self._format_hint_text(record['test_hint'])} | "
                    f"baseline={record['test_baseline_exec']:.3f}+{record['test_baseline_plan']:.3f} | "
                    f"selected={record['test_selected_exec']:.3f}+{record['test_selected_plan']:.3f}"
                )
                out.write(header + '\n')
                sorted_cross_results = sorted(
                    record['cross_hint_results'],
                    key=lambda item: (not item['same_as_test_hint'], tuple(item['source_train_dbs']))
                )
                for cross_result in sorted_cross_results:
                    out.write(
                        "train_dbs=[{}] | same_hint={} | hint={} | exec={} | plan={} | timeout={} | vs_base={} | vs_test={}\n".format(
                            ','.join(cross_result['source_train_dbs']),
                            'yes' if cross_result['same_as_test_hint'] else 'no',
                            self._format_hint_text(cross_result['hint']),
                            f"{cross_result['exec_time']:.3f}",
                            f"{cross_result['planning_time']:.3f}",
                            cross_result['timeout'],
                            self._format_ratio_text(cross_result['vs_test_baseline_total']),
                            self._format_ratio_text(cross_result['vs_test_selected_total']),
                        )
                    )
                out.write('\n')
                out.flush()
        with open(self.generalizationDiagnosticsPath, "w") as out:
            for test_key in sorted(self.GeneralizationDiagnostics.keys()):
                record = self.GeneralizationDiagnostics[test_key]
                out.write(
                    "test={}|iter={}|selected_hint={}|selected_score={}|selected_adjusted_score={}|"
                    "selected_risk={}|raw_selected_adjusted_score={}|raw_selected_risk={}|"
                    "safe_fallback_to_baseline={}|score_margin_top1_top2={}|"
                    "baseline_total={}|selected_total={}|oracle_best_exec={}|oracle_best_total={}|"
                    "same_as_train_best={}|negative_cross_db={}|stop_taken={}|stop_step={}|"
                    "terminate_reason={}|planner_found_oracle={}|scorer_ranked_oracle_top1={}\n".format(
                        test_key,
                        record.get('val_iter', 0),
                        self._format_hint_text(record.get('selected_hint', '')),
                        self._format_ratio_text(record.get('selected_score', 'nan')),
                        self._format_ratio_text(record.get('selected_adjusted_score', 'nan')),
                        self._format_ratio_text(record.get('selected_risk', 'nan')),
                        self._format_ratio_text(record.get('raw_selected_adjusted_score', 'nan')),
                        self._format_ratio_text(record.get('raw_selected_risk', 'nan')),
                        'yes' if record.get('safe_fallback_to_baseline', False) else 'no',
                        self._format_ratio_text(record.get('score_margin_top1_top2', 'nan')),
                        self._format_ratio_text(record.get('baseline_total', 'nan')),
                        self._format_ratio_text(record.get('selected_total', 'nan')),
                        self._format_ratio_text(record.get('oracle_best_exec', 'unknown')),
                        self._format_ratio_text(record.get('oracle_best_total', 'unknown')),
                        'yes' if record.get('same_as_train_best', False) else 'no',
                        'yes' if record.get('negative_cross_db', False) else 'no',
                        'yes' if record.get('stop_taken', False) else 'no',
                        record.get('stop_step', 0),
                        record.get('terminate_reason', ''),
                        'yes' if record.get('planner_found_oracle', False) else 'no',
                        'yes' if record.get('scorer_ranked_oracle_top1', False) else 'no',
                    )
                )
                out.flush()
        with open(self.planningBreakdownPath, "w") as out:
            latest_records = []
            for db_and_queryid in sorted(self.PlanningBreakdownTrace.keys()):
                records = self.PlanningBreakdownTrace[db_and_queryid]
                if len(records) == 0:
                    continue
                latest_by_phase = {}
                for record in records:
                    phase = record.get('phase', 'test')
                    if (phase not in latest_by_phase) or (record['val_iter'] >= latest_by_phase[phase]['val_iter']):
                        latest_by_phase[phase] = record
                for phase, latest in latest_by_phase.items():
                    latest_records.append((db_and_queryid, latest))
            all_latest = [record for _, record in latest_records]
            phase_scopes = [('all', all_latest)]
            for phase in sorted({record.get('phase', 'test') for _, record in latest_records}):
                phase_scopes.append((phase, [record for _, record in latest_records if record.get('phase') == phase]))
            for scope, records in phase_scopes:
                summary = self._summarize_breakdown_records(records)
                out.write(
                    "summary|scope={}|records={}|mean_selected_step={}|mean_planner_step_len={}|mean_candidates={}|"
                    "mean_planning_total_ms={}|mean_reset_ms={}|mean_action_ms={}|mean_env_step_ms={}|"
                    "mean_obs_copy_ms={}|mean_predict_ms={}|mean_other_ms={}|max_planning_total_ms={}\n".format(
                        scope,
                        summary['records'],
                        summary['mean_selected_step'],
                        summary['mean_loop_count'],
                        summary['mean_candidate_count'],
                        summary['mean_planning_total_ms'],
                        summary['mean_reset_ms'],
                        summary['mean_action_ms'],
                        summary['mean_env_step_ms'],
                        summary['mean_obs_copy_ms'],
                        summary['mean_predict_ms'],
                        summary['mean_other_ms'],
                        summary['max_planning_total_ms'],
                    )
                )
            out.write('\n')
            slowest_latest = sorted(latest_records, key=lambda item: item[1]['planning_total_ms'], reverse=True)[:20]
            for db_and_queryid, record in slowest_latest:
                out.write(
                    "top|{}|iter={}|phase={}|use_dynahint={}|loops={}|candidates={}|selected_step={}|stop_taken={}|stop_step={}|"
                    "terminate_reason={}|reset_ms={}|action_ms={}|env_step_ms={}|obs_copy_ms={}|predict_ms={}|other_ms={}|"
                    "planning_total_ms={}\n".format(
                        db_and_queryid,
                        record['val_iter'],
                        record.get('phase', 'test'),
                        int(record['use_dynahint']),
                        record['loop_count'],
                        record['candidate_count'],
                        record.get('selected_step', 0),
                        'yes' if record.get('stop_taken', False) else 'no',
                        record.get('stop_step', 0),
                        record.get('terminate_reason', ''),
                        record['reset_ms'],
                        record['action_ms'],
                        record['env_step_ms'],
                        record['obs_copy_ms'],
                        record['predict_ms'],
                        record['other_ms'],
                        record['planning_total_ms'],
                    )
                )
            out.write('\n')
            for db_and_queryid, record in latest_records:
                out.write(
                    "query|{}|iter={}|phase={}|use_dynahint={}|loops={}|candidates={}|selected_step={}|stop_taken={}|stop_step={}|"
                    "terminate_reason={}|reset_ms={}|action_ms={}|env_step_ms={}|obs_copy_ms={}|predict_ms={}|other_ms={}|"
                    "planning_total_ms={}\n".format(
                        db_and_queryid,
                        record['val_iter'],
                        record.get('phase', 'test'),
                        int(record['use_dynahint']),
                        record['loop_count'],
                        record['candidate_count'],
                        record.get('selected_step', 0),
                        'yes' if record.get('stop_taken', False) else 'no',
                        record.get('stop_step', 0),
                        record.get('terminate_reason', ''),
                        record['reset_ms'],
                        record['action_ms'],
                        record['env_step_ms'],
                        record['obs_copy_ms'],
                        record['predict_ms'],
                        record['other_ms'],
                        record['planning_total_ms'],
                    )
                )
            out.flush()
    
    def close(self):
        self.resultFile.close()
        # self.test_expPool.close()

@ray.remote
class BestPlanManager:
    def __init__(self, genConfig):
        self.config = genConfig
        self.globalCandidate = pd.DataFrame(columns = ['db_queryid', 'hint','feature','sql','planning_time_cum'])
        self.iterCandidate = pd.DataFrame(columns = ['db_queryid', 'hint','feature','sql','prob','planning_time_cum'])
        self.balances = pd.DataFrame(columns = ['db_queryid', 'hint','feature','sql','planning_time_cum'])
        self.sampleNum = self.config.maxsamples
        self.iterNo      = 0
        self.globalNo    = 0
        self.masked    = {}
        self.scorer_esbest_feature = {}
        self.RL_estbest = {}
        self.medianplan = {}
        self.queryImportance = {}
        self.latencyBuffer = {}
        self.bestPlanRecords = {}
        self.endSignal = False
        # self.validationSet = None
        self.coeff_1 = 0.4
        self.coeff_2 = 0.6
        # self.iterCandidate.to_csv(self.config.ExperiencePool, mode='a', index=False, columns = ['queryid','hint','prob'])
        self._load_best_plan_records()

    def _best_plan_file_exists(self):
        return os.path.exists(self.config.best_plan_path)

    def _load_best_plan_records(self):
        if not self._best_plan_file_exists():
            self.bestPlanRecords = {}
            return self.bestPlanRecords
        try:
            with open(self.config.best_plan_path, 'r', encoding='utf-8') as f:
                self.bestPlanRecords = json.load(f)
        except Exception as e:
            print(f"[Warning] Failed to load best-plan file: path={self.config.best_plan_path}, error={e}")
            self.bestPlanRecords = {}
        return self.bestPlanRecords

    def _save_best_plan_records(self):
        best_plan_dir = os.path.dirname(self.config.best_plan_path)
        if best_plan_dir and not os.path.exists(best_plan_dir):
            os.makedirs(best_plan_dir, exist_ok=True)
        tmp_path = self.config.best_plan_path + '.tmp'
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(self.bestPlanRecords, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.config.best_plan_path)

    def _extract_cardinality_for_query(self, query_records):
        cardinality = query_records.get('cardinality', -1)
        try:
            if cardinality is None or pd.isna(cardinality):
                return -1
        except Exception:
            pass
        return cardinality

    def _iter_query_hint_records(self, query_records):
        for hint, latency_info in query_records.items():
            if hint in ('cardinality', '_cardinality_evidence'):
                continue
            if not isinstance(latency_info, list) or len(latency_info) == 0:
                continue
            latency = float(latency_info[0])
            timeout = bool(latency_info[1]) if len(latency_info) > 1 else False
            yield hint, latency, timeout

    def initialize_best_plan_records(self):
        self._load_best_plan_records()
        if len(self.bestPlanRecords) > 0:
            return 0
        initialized = 0
        for db, query_map in self.latencyBuffer.items():
            self.bestPlanRecords.setdefault(db, {})
            for query_id, query_records in query_map.items():
                baseline_record = query_records.get('', None)
                if not isinstance(baseline_record, list) or len(baseline_record) == 0:
                    candidate_records = list(self._iter_query_hint_records(query_records))
                    if len(candidate_records) == 0:
                        continue
                    hint, latency, timeout = min(candidate_records, key=lambda item: item[1])
                else:
                    hint = ''
                    latency = float(baseline_record[0])
                    timeout = bool(baseline_record[1]) if len(baseline_record) > 1 else False
                self.bestPlanRecords[db][query_id] = {
                    'hint': hint,
                    'latency': latency,
                    'timeout': timeout,
                    'step': 0,
                    'cardinality': self._extract_cardinality_for_query(query_records),
                    'source': 'baseline_seed',
                }
                initialized += 1
        if initialized > 0:
            self._save_best_plan_records()
        return initialized

    def bootstrap_best_plan_from_latencybuffer(self):
        self._load_best_plan_records()
        updated = 0
        for db, query_map in self.latencyBuffer.items():
            self.bestPlanRecords.setdefault(db, {})
            for query_id, query_records in query_map.items():
                current_best = self.bestPlanRecords[db].get(query_id, None)
                current_latency = float(current_best['latency']) if current_best is not None else None
                best_hint = None
                best_latency = None
                best_timeout = False
                for hint, latency, timeout in self._iter_query_hint_records(query_records):
                    if best_latency is None or latency < best_latency:
                        best_hint = hint
                        best_latency = latency
                        best_timeout = timeout
                if best_hint is None:
                    continue
                if current_latency is None or best_latency + 1e-9 < current_latency:
                    self.bestPlanRecords[db][query_id] = {
                        'hint': best_hint,
                        'latency': best_latency,
                        'timeout': best_timeout,
                        'step': int(current_best.get('step', 0)) if current_best is not None else 0,
                        'cardinality': self._extract_cardinality_for_query(query_records),
                        'source': 'latency_buffer_bootstrap',
                    }
                    updated += 1
        if updated > 0:
            self._save_best_plan_records()
        return updated

    def get_best_plan_records(self):
        self._load_best_plan_records()
        return deepcopy(self.bestPlanRecords)

    def update_best_plan_record(self, db, query_id, hint, latency, timeout, step=0, cardinality=-1, source='runtime'):
        self._load_best_plan_records()
        self.bestPlanRecords.setdefault(db, {})
        prev = self.bestPlanRecords[db].get(query_id, None)
        prev_latency = float(prev['latency']) if prev is not None else None
        candidate_latency = float(latency)
        if prev_latency is None or candidate_latency + 1e-9 < prev_latency:
            self.bestPlanRecords[db][query_id] = {
                'hint': hint,
                'latency': candidate_latency,
                'timeout': bool(timeout),
                'step': int(step),
                'cardinality': cardinality if cardinality is not None else -1,
                'source': source,
            }
            self._save_best_plan_records()
            return True
        return False

    def update_scorer_esbest(self,db,query_id,feature_dict, exectime):
        self.scorer_esbest_feature[db+'|'+query_id] = (feature_dict,exectime)

    def add_globalCandidate(self,db,query_id,hint,feature_dict,sql,planning_time_cum=0.0):  # No drop
        self.globalCandidate.loc[self.globalNo] = \
        {'db_queryid':db+'|'+query_id,'hint':hint,'feature':feature_dict,'sql':sql,'planning_time_cum':float(planning_time_cum)}
        self.globalNo += 1
    
    def clear_globalCandidate(self):
        self.globalCandidate.drop(self.globalCandidate.index, inplace = True)

    def add_iterCandidate(self,db,query_id,hint,feature_dict,sql,prob,planning_time_cum=0.0):  #Drop every iter or every scorer upadtes
        if not ((self.iterCandidate['db_queryid'] == db+'|'+query_id) & (self.iterCandidate['hint'] == hint)).any():
            self.iterCandidate.loc[self.iterNo] = \
            {'db_queryid':db+'|'+query_id,'hint':hint,'feature':feature_dict,'sql':sql,'prob':prob,'planning_time_cum':float(planning_time_cum)}
            self.iterNo += 1
            
    def clear_iterCandidate(self):
        self.iterCandidate.drop(self.iterCandidate.index, inplace = True)

    # def write_iterCandidate(self):
    #     self.iterCandidate.to_csv(self.config.ExperiencePool, mode='a', index=False, columns = ['queryid','hint','prob'], header = 0)

    def add_balances(self,db,query_id,hint,feature_dict,sql,planning_time_cum=0.0):
        self.balances.loc[len(self.balances)] = \
        {'db_queryid':db+'|'+query_id,'hint':hint,'feature':feature_dict,'sql':sql,'planning_time_cum':float(planning_time_cum)}

    def get_balances(self):
        if len(self.balances) > 0:
            samples = self.balances.sample(frac=1)
            self.balances = pd.DataFrame(columns = self.balances.columns)
            return samples
        else:
            return None
        
    def get_stateNo(self):
        return self.globalNo, self.iterNo
        
    def get_scorer_esbest(self,db=None,query_id = None):
        if query_id == None:
            return self.scorer_esbest_feature
        else:
            return self.scorer_esbest_feature[db+'|'+query_id]
        
    def get_scorer_best(self):
        total = 0
        best_steps = {}
        for k,v in self.scorer_esbest_feature.items():
            total += v[1]
            best_steps[k] = v[0]['steps']
        return total, best_steps

    def get_scorer_best_by_db(self):
        totals_by_db = {}
        best_steps_by_db = {}
        for key, value in self.scorer_esbest_feature.items():
            db, query_id = key.split('|', 1)
            totals_by_db[db] = totals_by_db.get(db, 0.0) + value[1]
            best_steps_by_db.setdefault(db, {})[query_id] = value[0]['steps']
        return totals_by_db, best_steps_by_db
    
    def update_RL_esbest(self,db,query_id,featuredict,exectime):
        self.RL_estbest[db+'|'+query_id] = (featuredict, exectime)

    def get_RL_esbest(self,db=None,query_id = None):
        if query_id == None:
            return self.RL_estbest
        else:
            return self.RL_estbest[db+'|'+query_id]
        
    def get_median_plan(self):
        for k in self.scorer_esbest_feature:
            db,query_id = k.split('|')
            hint_latency = self.latencyBuffer[db][query_id]
            baseline_record = hint_latency.get('')
            if not isinstance(baseline_record, list) or len(baseline_record) == 0:
                continue
            baselatency = baseline_record[0]
            hints  = ['']
            latency= [baselatency]
            for hint,lt in hint_latency.items():
                if hint in ('cardinality', '_cardinality_evidence'):
                    continue
                if not isinstance(lt, list) or len(lt) == 0:
                    continue
                if lt[0] < baselatency:
                    hints.append(hint)
                    latency.append(lt[0])
            median_value, median_hint = get_median(latency,hints)
            self.medianplan[k] = {'hint':median_hint, 'latency':median_value}
        return self.medianplan

    def get_best_hints_by_query(self, query_id, train_dbs):
        if isinstance(train_dbs, str):
            train_dbs = [train_dbs]
        best_hint_by_db = {}
        hint_to_dbs = defaultdict(list)
        for db in train_dbs:
            if db not in self.latencyBuffer or query_id not in self.latencyBuffer[db]:
                continue
            hint_latency = self.latencyBuffer[db][query_id]
            best_hint = None
            best_latency = None
            for hint, latency_info in hint_latency.items():
                if hint == 'cardinality':
                    continue
                if not isinstance(latency_info, list) or len(latency_info) == 0:
                    continue
                latency = latency_info[0]
                if best_latency is None or latency < best_latency:
                    best_hint = hint
                    best_latency = latency
            if best_hint is None:
                continue
            best_hint_by_db[db] = {'hint': best_hint, 'latency': best_latency}
            hint_to_dbs[best_hint].append(db)
        for hint in hint_to_dbs:
            hint_to_dbs[hint].sort()
        return {'by_db': best_hint_by_db, 'by_hint': dict(hint_to_dbs)}

    def get_query_latency_records(self, db, query_id):
        if db not in self.latencyBuffer or query_id not in self.latencyBuffer[db]:
            return {}
        return deepcopy(self.latencyBuffer[db][query_id])

    
    def update_weightsByRLesbest(self):
        for k,v in self.RL_estbest.items():
            weights_1 = max(1, (v[1] - self.scorer_esbest_feature[k][1]) / 10)
            weights_2 = max(1, v[1]/ 10)
            if k not in self.masked or not self.masked[k]:
                self.queryImportance[k] = int(self.coeff_1 * (2 ** (math.floor(math.log10(weights_1)))) + self.coeff_2 * (2 ** (math.floor(math.log10(weights_2)))))
            else:
                self.queryImportance[k] = 0
        return self.queryImportance
    
    def getqueryImportance(self):
        return self.queryImportance
    
    def update_latencyBuffer(self,latencybuffer):
        self.latencyBuffer = latencybuffer

    def get_latencyBuffer(self):
        return self.latencyBuffer
    
    def update_schedule(self,endSignal):
        self.endSignal = endSignal

    def get_schedule(self):
        return self.endSignal
    
    def updateMask(self, toTrain = None, toMask = None):
        if toTrain:
            for k in toTrain:
                self.masked[k] = False
        if toMask:
            for k in toMask:
                self.masked[k] = True
                
    def random_sample(self, sampleNum = 0, frac = 0.1):
        if sampleNum == 0:
            return None, 0
        if len(self.iterCandidate) < sampleNum:
            sampleNum = len(self.iterCandidate)
        if sampleNum != 0:
            samples = self.iterCandidate.sample(sampleNum)
            self.iterCandidate.drop(samples.index, inplace=True)
            print(f'Random Sample:{len(samples)}')
            return samples, len(samples)
        else:
            samples = self.iterCandidate.sample(frac = frac)
            self.iterCandidate.drop(samples.index, inplace=True)
            print(f'Random Sample:{len(samples)}')
            return samples, len(samples)
    def uncertainty_sample(self, sampleNum = 0, threshold = 0.85):
        if sampleNum == 0:
            return None, 0
        self.iterCandidate['prob'] = self.iterCandidate['prob'].apply(lambda x: max(x))
        self.iterCandidate_filter = self.iterCandidate[self.iterCandidate['prob'] < threshold]
        if len(self.iterCandidate_filter) == 0:
            return None, 0
        samples = self.iterCandidate_filter.nsmallest(sampleNum, 'prob')
        self.iterCandidate.drop(samples.index, inplace = True)
        print('Uncertainty Samples: ', len(samples))
        return samples, len(samples)
    
    def hybrid_sample(self, predictor, sampleNum, threshold=None):
        if sampleNum == 0:
            return None, 0
        if len(self.iterCandidate) == 0:
            return None, 0
        if threshold is None:
            threshold = float(getattr(self.config, 'hybrid_sample_threshold', 0.9))
        self.iterCandidate['prob'] = self.iterCandidate['prob'].apply(lambda x: max(x))
        self.iterCandidate_filter = self.iterCandidate[self.iterCandidate['prob'] < threshold]
        if len(self.iterCandidate_filter) == 0:
            return None, 0
        uncertainty_samples = self.iterCandidate_filter.nsmallest(sampleNum * 10, 'prob')
        embeddings = ray.get(predictor.get_embed.remote(uncertainty_samples['feature']))
        if len(uncertainty_samples) > sampleNum:
            cosine_sim_matrix = cosine_similarity(embeddings)
            similarity_scores = np.sum(cosine_sim_matrix, axis=1) - 1
            q_idxs = np.argsort(similarity_scores)[:sampleNum]
            samples = uncertainty_samples.iloc[q_idxs]
        else:
            samples = uncertainty_samples
        self.iterCandidate.drop(samples.index, inplace = True)
        print(f'Hybrid Sample:{len(samples)}')
        return samples, len(samples)
    
    def hybrid_sample_global(self, predictor, sampleNum, currPool):
        if sampleNum == 0 or len(self.iterCandidate) == 0:
            return None, 0
        self.iterCandidate['prob'] = self.iterCandidate['prob'].apply(lambda x: max(x))
        uncertainty_samples = self.iterCandidate.nsmallest(sampleNum * 10, 'prob')
        undetermined = ray.get(predictor.get_embed.remote(uncertainty_samples['feature']))
        curr_embeddings = ray.get(predictor.get_embed.remote(currPool))
        if len(uncertainty_samples) > sampleNum:
            farthest_points_indices = []
            cosine_sim_matrix = cosine_similarity(undetermined, curr_embeddings)
            sum_similarities = np.sum(cosine_sim_matrix, axis=1)
            for _ in range(sampleNum):
                farthest_point_idx = np.argmin(sum_similarities)
                farthest_points_indices.append(farthest_point_idx)
                # curr_embeddings = np.vstack([curr_embeddings, undetermined[farthest_point_idx]])
                new_similarities = cosine_similarity(undetermined, undetermined[farthest_point_idx].reshape(1, -1)).flatten()
                sum_similarities += new_similarities
                sum_similarities[farthest_point_idx] += len(curr_embeddings) # avoid chosen in the next iter
            samples = uncertainty_samples.iloc[farthest_points_indices]
        else:
            samples = uncertainty_samples
        self.iterCandidate.drop(samples.index, inplace=True)
        print(f'Hybrid Sample Global: {len(samples)}')
        return samples, len(samples)
    
    def heuristic_sample(self, predictor, sampleNum):
        if sampleNum == 0:
            return None, 0
        inputs = []
        idxlist = []
        for idx, samples in self.globalCandidate.iterrows():
            inputs.append({'left': self.scorer_esbest_feature[samples['db_queryid']][0],'right':samples['feature']})
            idxlist.append(idx)
        to_validate = []
        prediction = ray.get(predictor.GetListPrediction.remote(inputs))
        for i, pred in enumerate(prediction):
            if pred != 0:
                to_validate.append(idxlist[i])
        if len(to_validate) >= sampleNum:
            sample_idx = random.sample(to_validate, sampleNum)
            samples = self.globalCandidate.loc[sample_idx]
            self.globalCandidate.drop(sample_idx, inplace=True)
        else:
            random.shuffle(to_validate)
            samples = self.globalCandidate.loc[to_validate]
            self.globalCandidate.drop(to_validate, inplace=True)
        print(f'Heuristic Sample:{len(samples)}')
        return samples, len(samples)
    
    def get_toExecuted(self, strategy, predictor = None, sampleNum = None, currPool = None):
        start = time.time()
        if strategy == 'random':
            samples, num_samples = self.random_sample(sampleNum = sampleNum)
        elif strategy == 'uncertainty':
            samples, num_samples = self.uncertainty_sample(sampleNum)
        elif strategy == 'hybrid_global':
            samples, num_samples = self.hybrid_sample_global(predictor, sampleNum, currPool)
        elif strategy == 'heuristic':
            samples, num_samples = self.heuristic_sample(predictor, sampleNum)
        elif strategy == 'hybrid':
            samples, num_samples = self.hybrid_sample(
                predictor,
                sampleNum,
                threshold=float(getattr(self.config, 'hybrid_sample_threshold', 0.9)),
            )
        sample_time = time.time() - start
        # print(f'{strategy} sample time:{sample_time}')
        return samples, num_samples
