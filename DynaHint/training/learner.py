"""Scorer helper and active-sampling learner."""
import ray
from collection.datacollector import DataColletor
from training.scorer import ScorerTrainer,ScorerDataset
import pandas as pd
import torch,os,time
import numpy as np
from copy import deepcopy
from shared.config import Config
global_config = Config()   # Global configuration.
global_config.CpuDeployment(print_info=False)

def scorer_epochs_for_round(config, train_round):
    if int(train_round) == 0:
        return int(config.scorer_epochs)
    return int(getattr(config, 'scorer_incremental_epochs', config.scorer_epochs))

def should_sample_candidates(config, planner_iter, last_planner_iter, iter_no, last_iter_no):
    if bool(getattr(config, 'active_sample_once_per_planner_iter', True)):
        return planner_iter != last_planner_iter
    return iter_no != last_iter_no

def _safe_as_tensor(x, device=None):
    if isinstance(x, np.ndarray) and hasattr(x, 'flags') and (not x.flags.writeable):
        x = np.array(x, copy=True)
    t = torch.as_tensor(x)
    return t.to(device) if device is not None else t

@ray.remote(num_gpus = 0.3, num_cpus = global_config.ScorerworkersNum)
class RemoteScorer():
    # Scorer helper.
    def __init__(self, config, device = None, istrain=True, tmp_model=False):
        """Initialize the scorer helper for training, inference or temporary checkpoints."""
        self.config      = config
        self.scorer_trainer = ScorerTrainer(config, device = device)
        self.tmp_scorer_dir = getattr(self.config, 'model_dir', os.path.dirname(self.config.scorer_path))
        self.tmp_scorer_path = os.path.join(self.tmp_scorer_dir, 'tmp_scorer.pt')
        self.has_loaded_model = False
        if istrain and not tmp_model:
            if os.path.exists(self.config.scorer_path):
                self.has_loaded_model = self.scorer_trainer.try_load_model(
                    self.config.scorer_path,
                    load_tag='remote_scorer_train_init',
                )
                if not self.has_loaded_model:
                    print("Failed to load scorer; keeping the initialized model...")
        elif istrain and tmp_model:
            if os.path.exists(self.tmp_scorer_path):
                self.has_loaded_model = self.scorer_trainer.try_load_model(
                    self.tmp_scorer_path,
                    load_tag='remote_scorer_tmp_init',
                )
                if not self.has_loaded_model:
                    print("Failed to load temporary scorer; keeping the initialized model...")
        else:
            if os.path.exists(self.config.scorer_path):
                self.has_loaded_model = self.scorer_trainer.try_load_model(
                    self.config.scorer_path,
                    load_tag='remote_scorer_eval_init',
                )
                if not self.has_loaded_model:
                    raise Exception("加载优势模型失败，请检查模型是否正确！")
                
    def TrainModel(self, dataset,batch_size,epochs, train_round=0):
        # Train the model.
        train_started = time.perf_counter()
        initial_epochs = int(getattr(self.config, 'scorer_epochs', epochs))
        incremental_epochs = int(getattr(self.config, 'scorer_incremental_epochs', epochs))
        global_step_base = 0 if int(train_round) == 0 else initial_epochs + (int(train_round) - 1) * incremental_epochs
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
                'global_step_base': global_step_base,
                'elapsed_seconds': time.perf_counter() - train_started,
                'skipped': True,
            }
        reset_model = bool(getattr(self.config, 'scorer_reinitialize_each_train', False))
        self.scorer_trainer.retrainmodel(reset=reset_model)
        history = self.scorer_trainer.fit(
            dataset,
            mybatch_size = batch_size,
            epochs = epochs,
            warmup_epochs=3 if int(train_round) == 0 else 0,
        )
        # self.scorer_trainer.save_model(self.config.scorer_path)
        self.scorer_trainer.save_model(self.tmp_scorer_path)
        history['train_round'] = int(train_round)
        history['epochs'] = int(epochs)
        history['global_step_base'] = global_step_base
        history['elapsed_seconds'] = time.perf_counter() - train_started
        return history
        
    def GetPrediction(self,hint_feature, phase='test'):
        # Get predictions.
        return self.scorer_trainer.predict_epi_parallel(hint_feature, phase=phase)
        # return self.scorer_trainer.predict_epi(hint_feature)
    
    def GetListPrediction(self,inputs):
        # Get list predictions.
        return self.scorer_trainer.predict_list(inputs)
    
    def LoadModel(self, scorer_path = None):
        # Load the model.
        target_path = scorer_path if scorer_path else self.tmp_scorer_path
        load_tag = 'remote_scorer_manual' if scorer_path else 'remote_scorer_tmp_update'
        loaded = self.scorer_trainer.try_load_model(target_path, load_tag=load_tag)
        if loaded:
            self.has_loaded_model = True
        return loaded

    def SaveModel(self,save_tmp = False, model_path=None):
        # Save the model.
        if model_path is not None:
            self.scorer_trainer.save_model(model_path)
        elif save_tmp:
            self.scorer_trainer.save_model(self.tmp_scorer_path)
        else:
            self.scorer_trainer.save_model(self.config.scorer_path)

    def get_embed(self, plan_feature):
        # Get embeddings.
        batch_size = self.config.get_embed_batch
        keys = plan_feature.iloc[0].keys()
        all_embeddings = []

        for i in range(0, len(plan_feature), batch_size):
            batch_features = {k: torch.cat([_safe_as_tensor(features[k], device=self.scorer_trainer.device).unsqueeze(0)
                                            for features in plan_feature.iloc[i:i+batch_size].values], dim=0) for k in keys}
            embeddings = self.scorer_trainer.get_embed(batch_features)
            embeddings = embeddings.cpu().detach()
            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0).numpy()
        

@ray.remote(num_gpus = 0, num_cpus = global_config.LearnerworkersNum)
class Learner():
    # Learner for plan selection.
    def __init__(self, bpm, planhelper, scorer_actor, genConfig):
        """Initialize plan-selection learning with managers, plan helpers and scorer actors."""
        self.bpm           = bpm
        self.planhelper    = planhelper
        self.scorer_actor     = scorer_actor
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
        self.remote_scorer     = RemoteScorer.remote(
            genConfig, tmp_model=not bool(getattr(genConfig, 'offline_scorer_only', False))
        )
        if self.config.update_scorer:
            self.remote_scorer.SaveModel.remote(save_tmp=True)
        self.trainref      = None
        self.IsReady       = []
        self.globalNo      = 0
        self.iterNo        = 0
        self.plannerIterNo = 0
        self.trainTimes    = 0
        self.updateTimes   = 0
        self.toExecuted_counts = 0
        self.accumulatedExecuted  = 0
        self.pending_train_metrics = []
        self.baseline_info = {}

    def Runing(self):
        """Run active-sampling learning and track the number of executed plans."""
        print('Start Learner!')
        balances = ray.get(self.bpm.get_balances.remote())
        self.ExecutePlans(balances)
        toExecuted = []
        while True:
            globalNo, iterNo, plannerIterNo = ray.get(self.bpm.get_stateNo.remote())
            should_sample = should_sample_candidates(
                self.config,
                plannerIterNo,
                self.plannerIterNo,
                iterNo,
                self.iterNo,
            )
            if should_sample: # process the rest of the samples
                # ray.get(self.bpm.write_iterCandidate.remote())
                tmpExecuted, num_samples = ray.get(self.bpm.get_toExecuted.remote('hybrid', sampleNum = self.hybrid_sample, scorer_actor = self.scorer_actor))
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
                                                                                  scorer_actor = self.scorer_actor, currPool = curpool))
                    if tmpExecuted is not None:
                        toExecuted.append(tmpExecuted)
                ray.get(self.bpm.clear_iterCandidate.remote())
                self.iterNo = iterNo
                self.plannerIterNo = plannerIterNo
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
                    # tmp_scorer.pt is written only when the future completes.
                    ray.get(self.scorer_actor.LoadModel.remote())
                    self.updateTimes = self.trainTimes
                    self.trainref = None
                    self.IsReady = []
            current_train_eval = self.config.get_current_train_eval(self.trainTimes)  # Choose the scorer-retraining threshold based on completed training updates.
            if self.toExecuted_counts >= current_train_eval:
                if self.trainref == None:
                    inputs,labels,weights = self.datacollector.get_samples()
                    dataset = ScorerDataset(inputs,labels,weights)
                    dataset_len = dataset.__len__()
                    print('train data length:',dataset_len, 'train_eval threshold:', current_train_eval, 'train_times:', self.trainTimes)
                    if dataset_len <= 0:
                        print('[Warning] scorer retrain skipped: no valid pointwise samples after baseline/objective filtering')
                    else:
                        scorer_epochs = scorer_epochs_for_round(self.config, self.trainTimes)
                        self.trainref = self.remote_scorer.TrainModel.remote(
                            dataset,
                            self.config.scorer_batch_size,
                            scorer_epochs,
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
        # Train the scorer.
        inputs,labels,weights = self.datacollector.get_samples()
        dataset = ScorerDataset(inputs,labels,weights)
        dataset_len = dataset.__len__()
        print('train data length:',dataset_len)
        if dataset_len <= 0:
            print('[Warning] manual scorer retrain skipped: no valid pointwise samples')
            return
        ray.get(self.remote_scorer.TrainModel.remote(dataset,self.config.scorer_batch_size,self.config.scorer_epochs))

    def TrainOfflineScorer(self, output_path, epochs=None):
        """Train only the scorer from the already loaded real-latency experience."""
        inputs, labels, weights = self.datacollector.get_samples()
        dataset = ScorerDataset(inputs, labels, weights)
        dataset_len = len(dataset)
        if dataset_len <= 0:
            raise ValueError('offline scorer training has no valid samples')
        train_epochs = int(self.config.scorer_epochs if epochs is None else epochs)
        history = ray.get(self.remote_scorer.TrainModel.remote(
            dataset,
            self.config.scorer_batch_size,
            train_epochs,
            0,
        ))
        ray.get(self.remote_scorer.SaveModel.remote(model_path=output_path))
        return {
            'output_path': os.path.abspath(output_path),
            'sample_count': dataset_len,
            'epochs': train_epochs,
            'history': history,
        }

    def ResetOfflineSamples(self):
        """Drop source baseline samples before loading target-only experience."""
        self.datacollector = DataColletor(self.config)
        return True

    def pop_train_metrics(self):
        # Drain remote scorer history for TensorBoard logging by the main process.
        metrics = self.pending_train_metrics
        self.pending_train_metrics = []
        return metrics

    def CollectSample(self, database, query_id, optimal_feature,latency,cardinality,timeout,
                      planning_time_cum=None, baseline_planning_time=None, is_baseline=None):
        # Collect samples.
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
            is_baseline=is_baseline,
        )
        return True
    
    
    def GetPrediction(self,hint_feature, phase='test'):
        # Get predictions.
        return ray.get(self.remote_scorer.GetPrediction.remote(hint_feature, phase=phase))
    
    def ExecutePlans(self,candidates):
        # Execute the plan.
        if isinstance(candidates, pd.DataFrame):
            for idx, samples in candidates.iterrows():
                # if hints:
                db, query_id = samples['db_queryid'].split('|')
                sql = samples['sql']
                hint = samples['hint']
                feature_dict = samples['feature']
                _,bestexec = ray.get(self.bpm.get_best_executed_plan.remote(db,query_id))
                latency_timeout,iscollect,cardinality = ray.get(self.planhelper.GetLatency.remote(hint, sql, db, query_id,timeout = self.config.timeoutcoeff * self.baseline[db+'|'+query_id], step=feature_dict['steps'][0], measurement_phase='experience'))
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
                    )  # Print a compact sample-execution summary.
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
                    
                    # Optional resource-usage diagnostics.
                    # resources = ray.available_resources()
                    # total_resources = ray.cluster_resources()
                    # Optional resource-usage heading.
                    # Optional CPU-usage diagnostic.
                    # Optional total-CPU diagnostic.

                    if Advbybest >= self.config.splitpoint[-1]:
                        ray.get(self.bpm.update_best_executed_plan.remote(db, query_id, deepcopy(feature_dict),latency_timeout[0]))
                        bestexec = latency_timeout[0]

    def updateBaseline(self,baseline):
        # Update the baseline.
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
        self.datacollector.backfill_baseline_info(self.baseline_info)  # Backfill historical sample baselines when baseline data arrives late.
