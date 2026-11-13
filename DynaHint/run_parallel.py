# -*- coding: utf-8 -*-
"""Parallel DynaHint training and inference entry point."""
from shared.config import Config
import ray
from ray.rllib.models import ModelCatalog
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
import time
from datetime import datetime
import argparse
import os
import json
import shutil
from training.model import create_custom_model
from shared.manager import QueryManager, BestPlanManager, ResultManager
from training.learner import Learner, RemoteScorer
from evaluation.workflow import EvaluationWorkflow
from training.workflow import TrainingWorkflow
from training.DynaHintEnv import DynaHintEnvTest
from collection.planhelper import RemotePlanHelper
from collection.histogram_preflight import ensure_histograms
from collection.encoding import is_encoding_cache_ready, write_json_atomic
import signal
import sys


class DynaHint(EvaluationWorkflow, TrainingWorkflow):
    # Manage the DynaHint runtime components.
    @staticmethod
    def _ensure_ray_initialized(genConfig):
        # Declare GPU resources explicitly to avoid GPUtil failures when NVML is unavailable.
        if ray.is_initialized():
            return
        ray.init(
            num_cpus=int(getattr(genConfig, 'use_resources', 1)),
            num_gpus=1,
            include_dashboard=False,
            ignore_reinit_error=True,
        )

    def __init__(self, config, AutoGetParam, isFirst=False, istrain=True,best_model_wrl = 1.0,best_model_gmrl = 1.0):
        """Initialize DynaHint configuration, runtime managers and checkpoint metrics."""
        self.genConfig    = config
        self.istrain      = istrain
        ensure_histograms(self.genConfig)
        self._ensure_ray_initialized(self.genConfig)
        self.planhelper   = RemotePlanHelper.remote(self.genConfig)
        self.bpm          = BestPlanManager.options(name="bpm").remote(self.genConfig)
        self.queryManager = QueryManager(self.genConfig, planhelper = self.planhelper, isfirst=isFirst)
        self._wait_for_encoding_cache()
        if AutoGetParam:
            self.AutoGetParam()
        self.scorer_actor    = RemoteScorer.remote(self.genConfig, istrain=istrain)
        if self.genConfig.update_scorer:
            self.learner  = Learner.remote(self.bpm, self.planhelper, self.scorer_actor, self.genConfig)
        else:
            self.learner  = None
        self.planner      = None
        self.writer       = SummaryWriter(log_dir = self.genConfig.runstate)
        self.evalEnv      = DynaHintEnvTest({'planhelper':self.planhelper, 'genConfig':self.genConfig})
        self.anaManger    = ResultManager(self.genConfig, self.writer)
        self.best_model_wrl = min(best_model_wrl, 1.0)
        self.best_model_gmrl = min(best_model_gmrl, 1.0)
        self.best_checkpoints = deepcopy(getattr(self.genConfig, 'best_checkpoints', {}))
        self.has_legacy_checkpoint = os.path.exists(self.genConfig.agent_checkpoint)
        self.tmp_planner_path = os.path.join(self.genConfig.model_dir, 'tmp_planner/')
        self.skipped_eval_queries = {}
        self.start_time = datetime.now()
        # Register signal handlers.
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

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and shut down safely."""
        print('\n---------- Shutting down safely... ----------')
        try:
            # Close all managers.
            if hasattr(self, 'anaManger'):
                self.anaManger.close()
            # Shut down Ray.
            ray.shutdown()
            print('---------- Program shut down safely ----------')
        except Exception as e:
            print(f'Error during shutdown: {e}')
        finally:
            sys.exit(0)

    def AutoGetParam(self):
        """Discover encoding dimensions and save the resulting configuration."""
        if not is_encoding_cache_ready(self.genConfig.encoding_path):
            raise RuntimeError(f"encoding cache is incomplete before AutoGetParam: {self.genConfig.encoding_path}")
        with open(self.genConfig.encoding_path, 'r') as encoding_file:
            encoding_json = json.load(encoding_file)
        if not os.path.exists(self.genConfig.auto_config):
            raise FileNotFoundError(f"auto_config missing before AutoGetParam: {self.genConfig.auto_config}")
        with open(self.genConfig.auto_config, 'r') as config_file:
            config_data = json.load(config_file)
        required_base_keys = ("maxjoins", "max_logical_edges", "filtmaxnum", "maxnode", "heightsize", "maxpos")
        missing_keys = [key for key in required_base_keys if key not in config_data]
        if missing_keys:
            raise RuntimeError(f"auto_config is incomplete before AutoGetParam: missing {missing_keys}")
        config_data["maxnode"] = max(
            int(config_data["maxnode"]),
            int(self.genConfig.maxnode),
        )
        config_data["heightsize"] = max(
            int(config_data["heightsize"]),
            int(self.genConfig.heightsize),
            config_data["maxnode"] + 1,
        )
        config_data["types"] = len(encoding_json['type2idx'])+2
        config_data["columns"] = len(encoding_json['col2idx'])+2
        config_data["tablenum"] = len(encoding_json['table2idx'])+1
        config_data["opsnum"] = len(encoding_json['op2idx'])+2
        config_data["db_hist_bins_per_col"] = self.genConfig.db_hist_bins_per_col
        config_data["db_bitmap_samples_per_table"] = self.genConfig.db_bitmap_samples_per_table
        config_data["node_hist_dim"] = 3 * (self.genConfig.db_hist_bins_per_col - 1)
        config_data["db_hist_dim"] = self.genConfig.db_hist_bins_per_col*(len(encoding_json['col2idx'])+2)
        if getattr(self.genConfig, "use_db_bitmap_features", True):
            config_data["db_bitmap_dim"] = self.genConfig.db_bitmap_samples_per_table*(len(encoding_json['table2idx'])-1)
        else:
            config_data["db_bitmap_dim"] = 0
        config_data["query_feat_dim"] = len(encoding_json['table2idx'])+2
        config_data["use_db_bitmap_features"] = bool(getattr(self.genConfig, "use_db_bitmap_features", True))
        config_data["num_node_feature"] = 7 + config_data["maxjoins"] + 5 * config_data["filtmaxnum"] + config_data["node_hist_dim"]
        config_data["AutoGetParam"] = 1
        print("Automatic parameter discovery completed.")

        write_json_atomic(self.genConfig.auto_config, config_data, ensure_ascii=False, indent=4)
        for key, value in config_data.items():
            setattr(self.genConfig, key, value)
        
    def Close(self):
        """Close managers and shut down Ray."""
        self.anaManger.close()  
        ray.shutdown()

if __name__ == "__main__":
    SupportDBMS = ['gaussdb', 'opengauss', 'hive', 'postgres']
    SupportWorkload = ['JOBRand', 'TPCH', 'STATS', 'TPCDS']

    parser = argparse.ArgumentParser("DynaHint Controller")
    parser.add_argument("--DBMS",choices = SupportDBMS,
                        help="Choose the DBMS from [GaussDB, openGauss, Hive, postgres]")    # Database backend.
    parser.add_argument("--Workload",choices = SupportWorkload,
                        help="Choose the Workload from [JOBRand, TPCH, STATS, TPCDS]")    # Workload.
    parser.add_argument("--ExpName",
                        help="The experiment name must be unique")    # Experiment name.
    parser.add_argument("--Database",
                        help="The Database Name")
    parser.add_argument("--Seed", type=int, default = None,
                        help="Random Seed")    # Random seed.
    parser.add_argument("--Maxsteps", type=int, default = None,
                        help="Max steps of agent")    # Maximum search steps.
    parser.add_argument("--Maxsamples", type=int, default = None,
                        help="Maximum number of samples in a single iteration (in best plan manager)")  # Maximum samples per iteration.
    parser.add_argument("--TotalIter", type=int, default = None,
                        help="The total number of iterations for which the planner is trained")  # Total planner-training iterations.
    parser.add_argument("--ValidateFreq", type=int, default = None,
                        help="The frequency of validation")    # Validation and checkpoint frequency.
    parser.add_argument("--Agents", type=int, default = None,
                        help="The Num of Agents.")    # Agent and policy counts.
    parser.add_argument("--PenaltyCoeff", type=int, default = None,
                        help="The coefficient of penalty.")    # Penalty coefficient.
    parser.add_argument("--SampleStrategy", type=str, default = None,
                        help="The strategy of sampling.")    # Sampling strategy.
    parser.add_argument("--HybridSampleThreshold", type=float, default = None,
                        help="The probability threshold used by hybrid sampling.")    # Hybrid-sampling threshold.
    parser.add_argument("--OffLeftDeep", action='store_true',
                        help="Remove the left-deep restriction")    # Whether to remove the left-deep restriction.
    parser.add_argument("--ScorerPath", type=str, default = None,
                        help="The model path of Scorer.")    # Scorer model path.
    parser.add_argument("--SQLDir", type=str, default = None,
                        help="The SQL address of model inference.")   # SQL directory for inference.
    parser.add_argument("--ModelPath", type=str, default = None)   # Inference model path.
    parser.add_argument(
        "--BestMetric",
        choices=('wrl', 'gmrl', 'train_wrl', 'train_gmrl'),
        default='wrl',
        help="Select the best checkpoint reference used for inference.",
    )
    parser.add_argument("--NotAutoGetParam", action='store_true')   # Disable automatic dimension discovery.
    parser.add_argument("--ContinueTrain", type=str, default = None)   # Model path for resumed training.
    parser.add_argument("--ScorerOnly", action='store_true',
                        help="Only retrain the scorer from an existing latency experience pool.")
    parser.add_argument("--ScorerOutput", type=str, default=None,
                        help="Output path for --ScorerOnly; defaults to <model>/offline_scorer.pt.")
    parser.add_argument("--ScorerEpochs", type=int, default=None,
                        help="Epochs for --ScorerOnly; defaults to scorer_epochs.")
    parser.add_argument("--ScorerTrainDBs", type=str, default=None,
                        help="Comma-separated DB allowlist for offline scorer training.")
    parser.add_argument("--ScorerTrainSource", choices=('train', 'test_and_drift'), default='train',
                        help="Use train data or in-sample Test and TestDataDrift data for offline scorer training.")
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
    if args.OffLeftDeep:
        config.left_deep_restriction = False
    if not args.ExpName and not config.expname:
        config.expname = f"{config.seed}_{args.Maxsteps}"
    config.ConfirmPath()
    if args.ScorerPath:
        config.scorer_path = args.ScorerPath
        
    if args.NotAutoGetParam :
        # Disable automatic dimension discovery.
        config.AutoGetParam = False
        if os.path.exists(config.auto_config):
            config_data = json.load(open(config.auto_config, 'r'))
            config_data["AutoGetParam"] = 0
            write_json_atomic(config.auto_config, config_data, ensure_ascii=False, indent=4)

    if args.ScorerOnly:
        if args.ContinueTrain is None:
            raise ValueError('--ScorerOnly requires --ContinueTrain to identify the saved run and experience pool.')
        model_path = config.ResolveModelDir(args.ContinueTrain)
        train_info_path = os.path.join(model_path, 'trainfo.json')
        config.CpuDeployment(type='continue', print_info=False)
        config.LoadTrainInfo(train_info_path)
        config.offline_scorer_only = True
        if not os.path.isfile(config.scorer_path):
            raise FileNotFoundError(f'Offline scorer initialization checkpoint is missing: {config.scorer_path}')
        output_path = os.path.abspath(args.ScorerOutput or os.path.join(config.model_dir, 'offline_scorer.pt'))
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        # Keep temporary scorer weights and run artifacts away from the source checkpoint.
        if os.path.abspath(config.auto_config) != os.path.join(output_dir, 'auto_config.json'):
            shutil.copyfile(config.auto_config, os.path.join(output_dir, 'auto_config.json'))
            config.auto_config = os.path.join(output_dir, 'auto_config.json')
        config.model_dir = output_dir
        config.runstate = os.path.join(output_dir, 'runstate')
        config.eval_output_path = os.path.join(output_dir, 'baseline_metrics.json')
        config.outfile_path = os.path.join(output_dir, 'baseline_queries.json')
        dynahint = DynaHint(config, config.AutoGetParam, isFirst=True)
        target_training = args.ScorerTrainSource == 'test_and_drift'
        print('Loading cached latencies and rebuilding baseline features......', flush=True)
        dynahint.RunBaseline(train_only=not target_training)
        if target_training:
            ray.get(dynahint.learner.ResetOfflineSamples.remote())
            dynahint.baseline_info.update(
                dynahint.baseline_info_by_phase.get('test_data_drift', {})
            )
            ray.get(dynahint.learner.updateBaseline.remote(dynahint.baseline_info))
        if args.ScorerTrainDBs:
            scorer_train_dbs = [db.strip() for db in args.ScorerTrainDBs.split(',') if db.strip()]
        elif target_training:
            scorer_train_dbs = list(config.databases[config.train_databases:])
        else:
            scorer_train_dbs = list(config.databases[:config.train_databases])
        if target_training and not scorer_train_dbs:
            raise ValueError('No Test or TestDataDrift databases configured for target-set training')
        loaded_count = dynahint.LoadExperiencePool(
            config.latency_buffer_path,
            databases=scorer_train_dbs,
            need_card_label=not target_training,
            update_best_plan=not target_training,
        )
        offline_result = ray.get(dynahint.learner.TrainOfflineScorer.remote(
            output_path,
            epochs=args.ScorerEpochs,
        ))
        print(
            'Offline scorer complete | records={} | samples={} | epochs={} | output={}'.format(
                loaded_count,
                offline_result['sample_count'],
                offline_result['epochs'],
                offline_result['output_path'],
            )
        )
    elif args.SQLDir is not None:
        # Inference workflow.
        config.CpuDeployment(type = 'test')
        config.update_scorer = False
        if args.ModelPath is None:   # Use the latest model when no model path is specified.
            model_path = config.LatestModelDir(train_mode=config.train_mode)
            if model_path is None:
                raise FileNotFoundError(f"没有找到可用于推理的模型目录: {config.model_root}")
            config.opti_result_path = os.path.join(config.opti_result_path, os.path.basename(model_path))
        else:
            model_path = config.ResolveModelDir(args.ModelPath)
            config.opti_result_path = os.path.join(config.opti_result_path, os.path.basename(model_path))
        print(f'Running inference with model: {os.path.abspath(model_path)}......')
        config.model_dir = model_path
        train_info_path = os.path.join(model_path, 'trainfo.json')
        with open(train_info_path, 'r') as train_info_file:
            saved_action_mode = json.load(train_info_file).get('action_mode', 'set_then_join_order')
        config.action_mode = saved_action_mode
        config.agent_checkpoint, config.scorer_path = config.ResolveBestCheckpointPaths(
            model_path,
            metric=args.BestMetric,
        )
        config.auto_config = os.path.join(model_path,'auto_config.json')
        if os.path.exists(config.auto_config):
            config_data = json.load(open(config.auto_config, 'r'))
            config.AutoGetParam = config_data['AutoGetParam']
        config.encoding_frozen = True
        dynahint = DynaHint(config, config.AutoGetParam, istrain=False)
        dynahint.BuildPlanner(train_batch_size = config.planner_batch_size, workersNum = config.PlannerworkersNum, planner_path = config.agent_checkpoint)
        dynahint.RunInference(args.SQLDir, exe=True)  # Run inference.
    else:
        if not config.update_scorer:
            raise ValueError('update_scorer=False is no longer supported because the Sim training path was removed.')
        StartIter = 0
        if args.ContinueTrain is not None:
            # Resume training.
            model_path = config.ResolveModelDir(args.ContinueTrain)
            train_info_path = os.path.join(model_path, 'trainfo.json')
            config.CpuDeployment(type = 'continue', print_info = False)
            StartIter, model_wrl_train, model_gmrl_train = config.LoadTrainInfo(train_info_path)
            if StartIter<config.total_iter:
                print(f'Completed iterations: {StartIter}; resuming at iteration {StartIter+1}......')
                dynahint = DynaHint(config, config.AutoGetParam, isFirst=True, best_model_wrl = model_wrl_train, best_model_gmrl = model_gmrl_train)
                print("Running Baseline......")
                dynahint.RunBaseline(startIter = StartIter)
                dynahint.LoadExperiencePool(config.latency_buffer_path)
                print("Initialing Planner......")
                dynahint.BuildPlanner(train_batch_size = config.planner_batch_size, workersNum = config.PlannerworkersNum, planner_path = config.agent_checkpoint)
            else:
                print(f'Completed iterations: {StartIter}; configured iteration limit: {config.total_iter}. Increase the limit before resuming.')
                sys.exit(0)
        else:
            # Training workflow.
            config.CpuDeployment(type = 'train')
            dynahint = DynaHint(config, config.AutoGetParam, isFirst=True)
            print("Running Baseline......")
            dynahint.RunBaseline()
            print("Initialing Planner......")
            dynahint.BuildPlanner(train_batch_size = config.planner_batch_size, workersNum = config.PlannerworkersNum)
            initialPro = config.initialPro  # Set the initial sampling fraction for scorer training.
            print("Initializing Scorer........")
            dynahint.InitScorer(initialPro = initialPro)
            time.sleep(10)  # Wait for scorer initialization.
        print("Training DynaHint.........")
        dynahint.Run(startIter=StartIter, totalIter = config.total_iter, valFreq = config.val_freq)
    dynahint.Close()
