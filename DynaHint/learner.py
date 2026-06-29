import ray
from datacollector import DataColletor
from pointestimator import PointTrainer,MyDataset
import pandas as pd
import torch,os
import numpy as np
from copy import deepcopy
from config import Config
global_config = Config()
global_config.CpuDeployment(print_info=False)
def _safe_as_tensor(x, device=None):
    if isinstance(x, np.ndarray) and hasattr(x, 'flags') and (not x.flags.writeable):
        x = np.array(x, copy=True)
    t = torch.as_tensor(x)
    return t.to(device) if device is not None else t

@ray.remote(num_gpus = 0.3, num_cpus = global_config.ScorerworkersNum)
class RemoteScorer(): 
    def __init__(self, config, device = None, istrain=True, tmp_model=False):
        self.config      = config
        self.pointtrainer = PointTrainer(config, device = device)
        self.tmp_scorer_dir = os.path.dirname(self.config.scorer_path)
        self.tmp_scorer_path = os.path.join(self.tmp_scorer_dir, 'tmp_scorer.pt')
        self.has_loaded_model = False
        if istrain and not tmp_model:
            if os.path.exists(self.config.scorer_path):
                self.has_loaded_model = self.pointtrainer.try_load_model(
                    self.config.scorer_path,
                    load_tag='remote_scorer_train_init',
                )
                if not self.has_loaded_model:
                    print("Failed to load scorer model. Continue with the current initialized model...")
        elif istrain and tmp_model:
            if os.path.exists(self.tmp_scorer_path):
                self.has_loaded_model = self.pointtrainer.try_load_model(
                    self.tmp_scorer_path,
                    load_tag='remote_scorer_tmp_init',
                )
                if not self.has_loaded_model:
                    print("Failed to load temporary scorer model. Continue with the current initialized model...")
        else:
            if os.path.exists(self.config.scorer_path):
                self.has_loaded_model = self.pointtrainer.try_load_model(
                    self.config.scorer_path,
                    load_tag='remote_scorer_eval_init',
                )
                if not self.has_loaded_model:
                    raise Exception("Failed to load scorer model. Please verify the model files.")
                
    def TrainModel(self, dataset,batch_size,epochs, train_round=0):
        if dataset is None or len(dataset) <= 0:
            print("[Warning] scorer training skipped: empty dataset")
            return {
                'total_loss': [],
                'score_loss': [],
                'card_loss': [],
                'risk_loss': [],
                'val_mse': [],
                'val_reward': [],
                'test_mse': [],
                'test_reward': [],
                'train_round': int(train_round),
                'epochs': int(epochs),
                'global_step_base': int(train_round) * max(int(epochs), 1),
                'skipped': True,
            }
        self.pointtrainer.retrainmodel()
        history = self.pointtrainer.fit(
            dataset,
            mybatch_size = batch_size,
            epochs = epochs,
        )
        # self.pointtrainer.save_model(self.config.scorer_path)
        self.pointtrainer.save_model(self.tmp_scorer_path)
        history['train_round'] = int(train_round)
        history['epochs'] = int(epochs)
        history['global_step_base'] = int(train_round) * max(int(epochs), 1)
        return history
        
    def GetPrediction(self,hint_feature, phase='test'):
        return self.pointtrainer.predict_epi_parallel(hint_feature, phase=phase)
        # return self.pointtrainer.predict_epi(hint_feature)
    
    def GetListPrediction(self,inputs):
        return self.pointtrainer.compare_feature_list(inputs)
    
    def LoadModel(self, scorer_path = None):
        target_path = scorer_path if scorer_path else self.tmp_scorer_path
        load_tag = 'remote_scorer_manual' if scorer_path else 'remote_scorer_tmp_update'
        loaded = self.pointtrainer.try_load_model(target_path, load_tag=load_tag)
        if loaded:
            self.has_loaded_model = True
        return loaded

    def SaveModel(self,save_tmp = False):
        if save_tmp:
            self.pointtrainer.save_model(self.tmp_scorer_path)
        else:
            self.pointtrainer.save_model(self.config.scorer_path)

    def get_embed(self, plan_feature):
        batch_size = self.config.get_embed_batch
        keys = plan_feature.iloc[0].keys()
        all_embeddings = []

        for i in range(0, len(plan_feature), batch_size):
            batch_features = {k: torch.cat([_safe_as_tensor(features[k], device=self.pointtrainer.device).unsqueeze(0)
                                            for features in plan_feature.iloc[i:i+batch_size].values], dim=0) for k in keys}
            embeddings = self.pointtrainer.get_embed(batch_features)
            embeddings = embeddings.cpu().detach()
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0).numpy()
        

@ray.remote(num_gpus = 0.3, num_cpus = global_config.LearnerworkersNum)
class Learner():
    def __init__(self, bpm, planhelper, predictor, genConfig):
        self.bpm           = bpm
        self.planhelper    = planhelper
        self.predictor     = predictor
        self.config        = genConfig
        self.baseline      = None
        self.maxsampleNum  = self.config.maxsamples
        self.uncertainty_sample  = 0
        self.hybrid_global_sample = 0 
        self.hybrid_sample        = 0
        if  self.config.sample_strategy  == 'uncertainty':
            self.uncertainty_sample  = self.config.maxsamples 
        elif self.config.sample_strategy == 'hybrid_global':
            self.hybrid_global_sample = self.config.maxsamples
        elif self.config.sample_strategy == 'hybrid':
            self.hybrid_sample        = self.config.maxsamples
        self.random_samples= self.maxsampleNum - self.uncertainty_sample - self.hybrid_sample - self.hybrid_global_sample
        self.datacollector = DataColletor(self.config)
        self.remote_scorer     = RemoteScorer.remote(genConfig, tmp_model=True)
        if self.config.update_scorer:
            self.remote_scorer.SaveModel.remote(save_tmp=True)
        self.trainref      = None
        self.IsReady       = []
        self.globalNo      = 0
        self.iterNo        = 0
        self.trainTimes    = 0
        self.updateTimes   = 0
        self.toExecuted_counts = 0
        self.accumulatedExecuted  = 0
        self.pending_train_metrics = []
        self.baseline_info = {}

    def Runing(self):
        print('Start Learner!')
        balances = ray.get(self.bpm.get_balances.remote())
        self.ExecutePlans(balances)
        toExecuted = []
        while True:
            globalNo, iterNo = ray.get(self.bpm.get_stateNo.remote())
            if self.trainTimes != self.updateTimes:
                self.predictor.LoadModel.remote()
                self.updateTimes = self.trainTimes
            if iterNo != self.iterNo: # process the rest of the samples
                # ray.get(self.bpm.write_iterCandidate.remote())
                tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('hybrid', sampleNum = self.hybrid_sample, predictor = self.predictor))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('uncertainty', sampleNum = self.uncertainty_sample))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('random', sampleNum = self.random_samples))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                if self.hybrid_global_sample != 0:
                    curpool = self.datacollector.get_featuresPool()
                    tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('hybrid_global', sampleNum = self.hybrid_global_sample, 
                                                                                  predictor = self.predictor, currPool = curpool))
                if tmpExecuted is not None:
                    toExecuted.append(tmpExecuted)
                ray.get(self.bpm.clear_iterCandidate.remote())
                self.iterNo = iterNo
            if len(toExecuted) > 0:
                toExecuted = pd.concat(toExecuted, ignore_index=True)  
                self.ExecutePlans(toExecuted)
                toExecuted = []
            if self.trainref != None:
                self.IsReady,_ = ray.wait([self.trainref], timeout = 0.01)
                if self.trainref in self.IsReady:
                    train_history = ray.get(self.trainref)
                    if train_history is not None:
                        self.pending_train_metrics.append(train_history)
                    self.trainref = None
                    self.IsReady = []
            current_train_eval = self.config.get_current_train_eval(self.trainTimes)
            if self.toExecuted_counts >= current_train_eval:
                if self.trainref == None:
                    inputs,labels,weights = self.datacollector.get_samples()
                    dataset = MyDataset(inputs,labels,weights)
                    dataset_len = dataset.__len__()
                    print('train data length:',dataset_len, 'train_eval threshold:', current_train_eval, 'train_times:', self.trainTimes)
                    if dataset_len <= 0:
                        print('[Warning] scorer retrain skipped: no valid pointwise samples after baseline/objective filtering')
                    else:
                        self.trainref = self.remote_scorer.TrainModel.remote(
                            dataset,
                            self.config.scorer_batch_size,
                            self.config.scorer_epochs,
                            self.trainTimes,
                        )
                    self.toExecuted_counts = 0       
                    self.IsReady = [] 
                    self.trainTimes += 1
            endSignal = ray.get(self.bpm.get_schedule.remote())
            if endSignal:
                print('Stop Learner!')
                return self.accumulatedExecuted
            
    def TrainModel(self):
        inputs,labels,weights = self.datacollector.get_samples()
        dataset = MyDataset(inputs,labels,weights)
        dataset_len = dataset.__len__()
        print('train data length:',dataset_len)
        if dataset_len <= 0:
            print('[Warning] manual scorer retrain skipped: no valid pointwise samples')
            return
        ray.get(self.remote_scorer.TrainModel.remote(dataset,self.config.scorer_batch_size,self.config.scorer_epochs))

    def pop_train_metrics(self):
        metrics = self.pending_train_metrics
        self.pending_train_metrics = []
        return metrics

    def CollectSample(self, database, query_id, optimal_feature,latency,cardinality,timeout,
                      planning_time_cum=None, baseline_planning_time=None):
        db_and_queryid = database + '|' + query_id
        baseline_latency = self.baseline.get(db_and_queryid) if self.baseline is not None else None
        baseline_meta = self.baseline_info.get(db_and_queryid, {})
        if baseline_planning_time is None:
            baseline_planning_time = baseline_meta.get('planning_time', 0.0)
        if planning_time_cum is None:
            planning_time_cum = baseline_planning_time if optimal_feature.get('steps', [0])[0] == 0 else 0.0
        step = int(round(float(optimal_feature.get('steps', [0])[0]) * self.config.maxsteps)) if 'steps' in optimal_feature else 0
        self.datacollector.collect_planVecPool(
            database,
            query_id,
            optimal_feature,
            (latency,cardinality),
            timeout,
            baseline_latency=baseline_latency,
            step=step,
            planning_time_cum=planning_time_cum,
            baseline_planning_time=baseline_planning_time,
        )
        return True
    
    
    def GetPrediction(self,hint_feature, phase='test'):
        return ray.get(self.remote_scorer.GetPrediction.remote(hint_feature, phase=phase))
    
    def ExecutePlans(self,candidates):
        if isinstance(candidates, pd.DataFrame):
            for idx, samples in candidates.iterrows():
                # if hints:
                db, query_id = samples['db_queryid'].split('|')
                sql = samples['sql']
                hint = samples['hint']
                feature_dict = samples['feature']
                _,bestexec = ray.get(self.bpm.get_scorer_esbest.remote(db,query_id))
                latency_timeout,iscollect,cardinality = ray.get(self.planhelper.GetLatency.remote(hint, sql, db, query_id,timeout = self.config.timeoutcoeff * self.baseline[db+'|'+query_id], step=feature_dict['steps'][0]))
                if iscollect:
                    self.toExecuted_counts += 1
                    self.accumulatedExecuted += 1
                    db_and_queryid = db + '|' + query_id
                    baseline_latency = self.baseline.get(db_and_queryid) if self.baseline is not None else None
                    baseline_planning_time = self.baseline_info.get(db_and_queryid, {}).get('planning_time', 0.0)
                    planning_time_cum = float(samples.get('planning_time_cum', 0.0))
                    self.datacollector.collect_planVecPool(
                        db,
                        query_id,
                        deepcopy(feature_dict),
                        (latency_timeout[0],cardinality),
                        latency_timeout[1],
                        baseline_latency=baseline_latency,
                        step=int(round(float(feature_dict['steps'][0]) * self.config.maxsteps)),
                        planning_time_cum=planning_time_cum,
                        baseline_planning_time=baseline_planning_time,
                    )
                    Advbybest = (bestexec - latency_timeout[0]) / bestexec
                    steps = int(feature_dict['steps'][0] * self.config.maxsteps)
                    print(
                        'SampleExec | db={} | query={} | step={} | best_so_far_ms={:.3f} | candidate_exec_ms={:.3f} | vs_best_pct={:.3f} | timed_out={}'.format(
                            db,
                            query_id,
                            steps,
                            bestexec,
                            latency_timeout[0],
                            Advbybest * 100,
                            'yes' if latency_timeout[1] else 'no',
                        )
                    )
                    if getattr(self.config, 'use_best_plan_pool', False):
                        ray.get(
                            self.bpm.update_best_plan_record.remote(
                                db,
                                query_id,
                                hint,
                                latency_timeout[0],
                                latency_timeout[1],
                                step=steps,
                                cardinality=cardinality,
                                source='execute_plans',
                            )
                        )
                    
                    # resources = ray.available_resources()
                    # total_resources = ray.cluster_resources()

                    if Advbybest >= self.config.splitpoint[-1]:
                        ray.get(self.bpm.update_scorer_esbest.remote(db, query_id, deepcopy(feature_dict),latency_timeout[0]))
                        bestexec = latency_timeout[0]

    def updateBaseline(self,baseline):
        if baseline is None:
            self.baseline = None
            self.baseline_info = {}
            return
        if len(baseline) > 0 and isinstance(next(iter(baseline.values())), dict):
            self.baseline_info = baseline
            self.baseline = {
                key: float(value.get('execution_time', 0.0))
                for key, value in baseline.items()
            }
        else:
            self.baseline = baseline
            self.baseline_info = {
                key: {'execution_time': float(value), 'planning_time': 0.0, 'total_time': float(value)}
                for key, value in baseline.items()
            }
        self.datacollector.backfill_baseline_info(self.baseline_info)
