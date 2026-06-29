import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# import ray
import torch
from datetime import datetime
from multiprocessing import cpu_count
import json
import re


class Config:
    def __init__(self):
        # ======   DBMS Config   =======
        self.max_time_out = 300000
        # # ====== PostgreSQL ========
        self.time = datetime.now().strftime('%y%m%d')
        self.version = "v5"
        self.expname = '{}_{}_mix+'.format(self.time, self.version)
        self.DBMS = 'postgres'
        self.mode = 'JOB'
        # self.mode = 'STATS'
        # self.mode = 'TPCDS'
        self.ip = os.getenv('DYNAHINT_PG_HOST', 'localhost')
        self.port = int(os.getenv('DYNAHINT_PG_PORT', '5432'))
        self.user = os.getenv('DYNAHINT_PG_USER', '')
        self.password = os.getenv('DYNAHINT_PG_PASSWORD', '')
        self.databases = ['imdb1900cut','imdb1910cut','imdb1920cut','imdb1930cut','imdb1940cut','imdb1950cut','imdb1960cut','imdb1970cut','imdb1980cut','imdb1990cut','imdb2000cut','imdb2010cut','imdb']
        # self.databases = ['stats_201112','stats_201206','stats_201212','stats_201306','stats_201312','stats']
        # self.databases = ['tpcds10g_199912','tpcds10g_200006','tpcds10g_200012','tpcds10g_200106','tpcds10g_200112','tpcds10g_200206','tpcds10g']
        self.reverse_experiment = False
        if self.reverse_experiment:
            self.databases = list(reversed(self.databases))
        self.cardinality_extract_mode = 'aggregate_input'
      
        # ====== Embed Config ========
        self.train_mode = 'mix+'  # data_drift, query_drift, mix, mix+
        self.train_data_rate = 0.8
        self.data_drift_workload_source = 'train'
        self.AutoGetParam = True
        self.types = 20
        self.columns = 63
        self.heightsize = 30
        self.maxnode = 60
        self.maxjoins = 10
        self.tablenum = 21
        self.opsnum = 18
        self.filtmaxnum = 6
        self.maxpos = 5
        # ======     Model    ======== 
        self.emb_size = 32
        self.ffn_dim = 32
        self.head_size = 10
        self.num_layers = 10
        self.dropout = 0.05
        self.hidden_dim = self.emb_size * 7 + 8 * (self.emb_size // 8) + self.emb_size // 2 + 1
        # ====== Data Feature (DB hist) ========
        #  'db_hist'   : shape [db_hist_dim]
        #  'query_feat': shape [query_feat_dim] (optional, e.g., table mask + simple stats)
        self.use_db_features = True
        self.db_feat_generate_if_missing = True
        self.db_hist_bins_per_col = 51
        self.node_hist_dim = 3 * (self.db_hist_bins_per_col - 1)
        self.num_node_feature = 7 + self.maxjoins + 5 * self.filtmaxnum + self.node_hist_dim
        self.db_hist_dim = self.columns * self.db_hist_bins_per_col      # e.g., (#columns * #bins) or any fixed-length vector
        self.query_feat_dim = self.tablenum + 1 + 2  # default: table-mask + (NA, num_joins, num_filters, db_id?, 

        self.db_feature_npz = "./db_features/db_features.npz"
        self.db_feature_source = 'DynaHint'
        self.hist_json_dir = './experiment/histogram'
        # ====== DB meta / drift features (table/column stats) ======
        self.enable_db_meta = True
        # [log1p(exact_row_count), log1p(total_bytes_mb), log1p(relpages), dead_ratio, tanh(stats_age_days/30), has_stats]
        self.db_table_feat_dim = 6
        # [log1p(sum_exact_row_count), log1p(total_bytes_mb), mean_dead_ratio, mean_stats_age_norm]
        self.db_global_feat_dim = 4
        # [mean_log1p(ndv), mean_null_frac, mean_log1p(avg_width), mean_mcv_freq1]
        self.db_col_feat_dim = 4

        self.db_token_hidden_dim = 128
        self.db_token_nhead = 4
        self.db_token_layers = 2
        self.db_token_ffn_mult = 4
        self.max_tables_in_schema = 256
        # ====== Train Config ========
        self.total_iter = 2000
        self.scorer_epochs = 20
        self.scorer_batch_size = 128
        self.scorer_lr = 3e-4
        self.maxsteps = 5
        workload_source_dir = 'train'
        if self.train_mode == 'data_drift':
            if self.data_drift_workload_source not in ('train', 'test'):
                raise ValueError("data_drift_workload_source must be 'train' or 'test'")
            workload_source_dir = self.data_drift_workload_source
        self.train_sql_count = len([
            f for f in os.listdir('./experiment/{}/{}/'.format(self.mode, workload_source_dir))
            if f.endswith('.sql')
        ])
        self.planner_batch_size = self.train_sql_count*self.maxsteps+100
        self.num_agents = 1
        self.num_policies = 1
        self.stop_warmup_iters = 100
        self.stop_min_steps = 2
        self.test_stop_min_steps = 1
        self.reward_use_total_time = False
        self.balanced_db_sampling = True
        self.checkpoint_metric = 'train_wrl'
        self.scorer_use_baseline_context = True
        self.scorer_data_fusion_mode = 'cross_attention'
        self.scorer_cross_attn_layers = 1
        self.scorer_cross_attn_heads = 3
        self.planner_data_fusion_mode = 'cross_attention'
        self.planner_cross_attn_actor_only = True
        self.planner_cross_attn_share_with_scorer = False
        self.cross_attn_use_local_plan_q = False
        self.scorer_use_risk_head = True
        self.scorer_risk_loss_weight = 0.2
        self.scorer_risk_inference_weight = 0.1
        self.risk_negative_margin = 1.05
        self.negative_sample_weight = 2.0
        self.high_cost_weight_log_base = 100.0
        self.high_cost_weight_cap = 3.0
        self.safe_select_enable = True
        self.safe_select_enable_train = False
        self.safe_select_enable_test = True
        self.safe_select_risk_threshold = 0.65
        self.safe_select_min_adjusted_margin = 0.05
        self.safe_select_baseline_score_eps = -0.02
        self.enable_train_safe_eval = False
        self.distill_best_plan_topk = 2
        self.distill_best_plan_weight = 1.5
        self.dynamic_hard_weight = True
        self.dynamic_hard_weight_cap = 2.0
        self.use_best_plan_pool = False
        self.bootstrap_best_plan_from_latency = False
        # ====== Cross-Attention Feature Config ======
        self.cross_attn_q_feature_mask = 'all'
        self.cross_attn_kv_feature_mask = 'all'
        self.cross_attn_q_use_query_feat = True
        self.cross_attn_q_use_plan_nodes = True

        self.maxbounty = 12
        self.penalty_coeff = 0.0
        self.planner_config_even = {
            "gamma_": 0.99,
            "lr": 5e-5,
        }
        self.planner_config_odd = {
            "gamma_": 0.95,
            "lr": 1e-4,
        }
        self.entropy_coeff = 0.01
        self.kl_coeff = 1.0
        self.lambda_ = 0.9
        self.alpha = 0.5
        self.vf_share_layers = True
        self.fcnet_hiddens = [384, 384, 384]
        self.fcnet_activation = "tanh"
        # ====== General ========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.PlannerworkersNum = 1
        self.LearnerworkersNum = 0
        self.ScorerworkersNum = 1
        self.use_cpu_rate = True
        self.CPU_thread_utilization_rate = 0.3
        self.get_embed_batch = 16
        self.predict_list_batch = 64
        self.seed = 2026
        self.recommend_threshold = 0.05
        self.initialPro = 0.2
        self.train_eval = 25
        self.train_eval_schedule = [[0, 25], [100, 50], [300, 75]]
        self.val_freq = 10
        self.start_val = 10
        self.splitpoint = [0.5, 0.05]
        self.classNum = len(self.splitpoint)
        self.timeoutcoeff = 1.01 + self.splitpoint[0]
        self.sample_strategy = 'hybrid'
        self.maxsamples = 5     # JOB & STATS = 5, TPC-DS = 10
        self.hybrid_sample_threshold = 0.9  # JOB & STATS = 0.9, TPC-DS = 0.98
        self.update_scorer = True
        self.left_deep_restriction = True

        # ====== Basic ========
        self.operator_pg2hint = {'Hash Join':'HASHJOIN','Merge Join': 'MERGEJOIN','Nested Loop':'NESTLOOP'}
        self.OperatorDict = {'NESTLOOP':1,'HASHJOIN':2,'MERGEJOIN':3}
        self.Operatortype = ['NESTLOOP','HASHJOIN','MERGEJOIN']

    def _safe_path_component(self, value):
        value = str(value).strip()
        value = value.replace('+', '_plus')
        value = re.sub(r'[^0-9A-Za-z._-]+', '_', value)
        return value.strip('._-') or 'default'

    def BuildModelRunId(self):
        expname = self._safe_path_component(self.expname)
        train_mode = self._safe_path_component(self.train_mode)
        if expname.endswith('_' + train_mode):
            return expname
        return f'{expname}_{train_mode}'

    def ResolveModelDir(self, model_ref):
        if model_ref is None:
            return None
        if os.path.isabs(model_ref) or os.path.sep in model_ref:
            return model_ref
        model_root = getattr(self, 'model_root', './model/{}_{}/'.format(self.DBMS, self.mode))
        candidates = [
            os.path.join(model_root, model_ref),
            os.path.join(model_root, f'{model_ref}_{self.train_mode}'),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        prefix_matches = [
            os.path.join(model_root, name)
            for name in os.listdir(model_root) if name.startswith(model_ref + '_')
        ] if os.path.isdir(model_root) else []
        prefix_matches = [
            path for path in prefix_matches
            if os.path.isdir(path) and (
                os.path.exists(os.path.join(path, 'trainfo.json')) or
                os.path.exists(os.path.join(path, 'scorer.pt'))
            )
        ]
        if prefix_matches:
            return max(prefix_matches, key=lambda path: os.path.getmtime(path))
        return candidates[0]

    def LatestModelDir(self, train_mode=None):
        model_root = getattr(self, 'model_root', './model/{}_{}/'.format(self.DBMS, self.mode))
        if not os.path.isdir(model_root):
            return None
        suffix = f'_{train_mode}' if train_mode else None
        candidates = []
        for name in os.listdir(model_root):
            path = os.path.join(model_root, name)
            if not os.path.isdir(path) or name == 'dbmeta':
                continue
            if suffix and not name.endswith(suffix):
                continue
            if os.path.exists(os.path.join(path, 'trainfo.json')) or os.path.exists(os.path.join(path, 'scorer.pt')):
                candidates.append(path)
        if not candidates:
            return None
        return max(candidates, key=lambda path: os.path.getmtime(path))

    def ConfirmPath(self):
        self.model_root = './model/{}_{}/'.format(self.DBMS, self.mode)
        self.model_run_id = self.BuildModelRunId()
        self.model_dir = os.path.join(self.model_root, self.model_run_id)
        self.total_latency_buffer = './latencybuffer/{}.json'.format(self.mode)
        self.train_workload_path = './experiment/{}/train/'.format(self.mode)
        self.test_workload_path = './experiment/{}/test/'.format(self.mode)
        self.encoding_path = os.path.join(self.model_root, 'encoding.json')
        self.scorer_path = os.path.join(self.model_dir, 'scorer.pt')
        self.agent_checkpoint = os.path.join(self.model_dir, 'planner/')
        self.db_meta_cache_dir = os.path.join(self.model_root, 'dbmeta/')
        self.latency_buffer_path = './latencybuffer/{}/{}_buffer_{}.json'.format(self.time,self.mode,self.expname)
        safe_train_mode = self._safe_path_component(self.train_mode)
        latency_name = '{}_{}'.format(self.mode, safe_train_mode)
        if self.reverse_experiment:
            latency_name += '_reverse'
        self.pg_latency = './latency/{}.json'.format(latency_name)
        self.cardinality_cache_path = './latency/{}_cardinality.jsonl'.format(self.mode)
        self.best_plan_path = './latency/{}_{}_best_plan.json'.format(self.mode, safe_train_mode)
        self.eval_output_path = './timely_result/{}.json'.format(self.mode)
        self.outfile_path = './timely_result/{}_{}.json'.format(self.mode,self.expname)
        self.runstate = './runstate/{}_{}_{}'.format(self.DBMS,self.mode,self.expname)
        self.auto_config = os.path.join(self.model_dir, 'auto_config.json')
        self.train_info_path = os.path.join(self.model_dir, 'trainfo.json')
        self.opti_result_path = './result/{}_{}/'.format(self.DBMS,self.mode)

    def get_current_train_eval(self, train_times=0):
        current_train_eval = self.train_eval
        for min_train_times, scheduled_train_eval in self.train_eval_schedule:
            if train_times >= min_train_times:
                current_train_eval = scheduled_train_eval
            else:
                break
        return current_train_eval

    def CpuDeployment(self,print_info = True, type = None):
        self.cpu_resources=cpu_count()
        if self.use_cpu_rate is True:
            self.use_resources = int(self.cpu_resources*self.CPU_thread_utilization_rate)
            if self.use_resources == 0:
                self.use_resources = 1
            scorer_rate = 0. if type == 'test' else 0.2
            self.LearnerworkersNum = int(self.use_resources*0.2)
            self.ScorerworkersNum = int(self.use_resources*scorer_rate)
            self.PlannerworkersNum = int(self.use_resources - self.LearnerworkersNum - self.ScorerworkersNum*2)
            if print_info is True:
                print("Compute resource allocation:")
                print(f"CPU - Used: {self.use_resources:.1f} (Planner: {self.PlannerworkersNum:.1f} + Learner: {self.LearnerworkersNum:.1f} + Scorer: {self.ScorerworkersNum*2:.1f}) / "
                f"Total: {self.cpu_resources:.1f}")
        else:
            scorer_rate = 0. if type == 'test' else 0.2
            if type == 'test':
                self.ScorerworkersNum = 0 
            if self.PlannerworkersNum + self.LearnerworkersNum + self.ScorerworkersNum*2 > self.cpu_resources:
                self.LearnerworkersNum = int(self.cpu_resources*0.2)
                self.ScorerworkersNum = int(self.cpu_resources*scorer_rate)
                self.PlannerworkersNum = int(self.cpu_resources - self.LearnerworkersNum - self.ScorerworkersNum*2)
                if print_info is True:
                    print("Insufficient CPU resources. Worker counts were adjusted to the CPU limit.")
                    print(f"CPU - Used: {self.cpu_resources:.1f} (Planner: {self.PlannerworkersNum:.1f} + Learner: {self.LearnerworkersNum:.1f} + Scorer: {self.ScorerworkersNum*2:.1f}) / "
                    f"Total: {self.cpu_resources:.1f}")

    def SaveTrainInfo(self, train_iter = 0, wrl_train = None, wrl_test = None, gmrl_train = None, gmrl_test = None,start_time = None, first_save = False):
        self.train_info = {}
        end_time = datetime.now()
        if train_iter==1 or not os.path.exists(self.train_info_path):
            self.train_info = {
                'start_time': [start_time.strftime('%Y/%m/%d %H.%M.%S')],
                'end_time': [end_time.strftime('%Y/%m/%d %H.%M.%S')],
                'time': self.time,
                'expname': self.expname,
                'train_mode': self.train_mode,
                'model_root': self.model_root,
                'model_run_id': self.model_run_id,
                'model_dir': self.model_dir,
                'train_iter': 1,
                'model_saved_iter': 1,
                'model_wrl_train': 1.0,
                'model_wrl_test': 1.0,
                'model_gmrl_train': 1.0,
                'model_gmrl_test': 1.0,
                'scorer_epochs': self.scorer_epochs,
                'train_eval': self.train_eval,
                'train_eval_schedule': self.train_eval_schedule,
                'hybrid_sample_threshold': self.hybrid_sample_threshold,
                'planner_batch_size': self.planner_batch_size,
                'maxsteps': self.maxsteps,
                'num_agents': self.num_agents,
                'num_policies': self.num_policies,
                'stop_warmup_iters': self.stop_warmup_iters,
                'stop_min_steps': self.stop_min_steps,
                'test_stop_min_steps': self.test_stop_min_steps,
                'reward_use_total_time': self.reward_use_total_time,
                'balanced_db_sampling': self.balanced_db_sampling,
                'checkpoint_metric': self.checkpoint_metric,
                'scorer_data_fusion_mode': self.scorer_data_fusion_mode,
                'scorer_cross_attn_layers': self.scorer_cross_attn_layers,
                'scorer_cross_attn_heads': self.scorer_cross_attn_heads,
                'planner_data_fusion_mode': self.planner_data_fusion_mode,
                'planner_cross_attn_actor_only': self.planner_cross_attn_actor_only,
                'planner_cross_attn_share_with_scorer': self.planner_cross_attn_share_with_scorer,
                'cross_attn_use_local_plan_q': self.cross_attn_use_local_plan_q,
                'cross_attn_q_feature_mask': self.cross_attn_q_feature_mask,
                'cross_attn_kv_feature_mask': self.cross_attn_kv_feature_mask,
                'cross_attn_q_use_query_feat': self.cross_attn_q_use_query_feat,
                'cross_attn_q_use_plan_nodes': self.cross_attn_q_use_plan_nodes,
                'use_db_features': self.use_db_features,
                'enable_db_meta': self.enable_db_meta,
                'scorer_use_risk_head': self.scorer_use_risk_head,
                'scorer_risk_loss_weight': self.scorer_risk_loss_weight,
                'scorer_risk_inference_weight': self.scorer_risk_inference_weight,
                'risk_negative_margin': self.risk_negative_margin,
                'negative_sample_weight': self.negative_sample_weight,
                'high_cost_weight_log_base': self.high_cost_weight_log_base,
                'high_cost_weight_cap': self.high_cost_weight_cap,
                'safe_select_enable': self.safe_select_enable,
                'safe_select_enable_train': self.safe_select_enable_train,
                'safe_select_enable_test': self.safe_select_enable_test,
                'safe_select_risk_threshold': self.safe_select_risk_threshold,
                'safe_select_min_adjusted_margin': self.safe_select_min_adjusted_margin,
                'safe_select_baseline_score_eps': self.safe_select_baseline_score_eps,
                'enable_train_safe_eval': self.enable_train_safe_eval,
                'data_drift_workload_source': self.data_drift_workload_source,
                'distill_best_plan_topk': self.distill_best_plan_topk,
                'distill_best_plan_weight': self.distill_best_plan_weight,
                'dynamic_hard_weight': self.dynamic_hard_weight,
                'dynamic_hard_weight_cap': self.dynamic_hard_weight_cap,
                'use_best_plan_pool': self.use_best_plan_pool,
                'bootstrap_best_plan_from_latency': self.bootstrap_best_plan_from_latency,
                'maxbounty': self.maxbounty,
                'penalty_coeff': self.penalty_coeff,
                'planner_config_even': self.planner_config_even,
                'planner_config_odd': self.planner_config_odd,
                'entropy_coeff': self.entropy_coeff,
                'kl_coeff': self.kl_coeff,
                'lambda_': self.lambda_,
                'vf_share_layers': self.vf_share_layers,
                'fcnet_hiddens': self.fcnet_hiddens,
                'fcnet_activation': self.fcnet_activation,
                'encoding_path': self.encoding_path,
                'auto_config': self.auto_config,
                'latency_buffer_path': self.latency_buffer_path,
                'best_plan_path': self.best_plan_path,
                'scorer_path': self.scorer_path,
                'agent_checkpoint': self.agent_checkpoint,
                'train_info_path': self.train_info_path,
                'PlannerworkersNum': self.PlannerworkersNum,
                'LearnerworkersNum': self.LearnerworkersNum,
                'ScorerworkersNum': self.ScorerworkersNum,
                'use_cpu_rate': self.use_cpu_rate,
                'CPU_thread_utilization_rate': self.CPU_thread_utilization_rate,
                'total_time': [(end_time-start_time).total_seconds()],
            }
        else:
            with open(self.train_info_path, 'r') as f:
                self.train_info = json.load(f)
            self.train_info['train_iter'] = train_iter
            self.train_info['expname'] = self.expname
            self.train_info['train_mode'] = self.train_mode
            self.train_info['model_root'] = self.model_root
            self.train_info['model_run_id'] = self.model_run_id
            self.train_info['model_dir'] = self.model_dir
            self.train_info['train_eval'] = self.train_eval
            self.train_info['train_eval_schedule'] = self.train_eval_schedule
            self.train_info['planner_batch_size'] = self.planner_batch_size
            self.train_info['maxsteps'] = self.maxsteps
            self.train_info['stop_warmup_iters'] = self.stop_warmup_iters
            self.train_info['stop_min_steps'] = self.stop_min_steps
            self.train_info['test_stop_min_steps'] = self.test_stop_min_steps
            self.train_info['reward_use_total_time'] = self.reward_use_total_time
            self.train_info['balanced_db_sampling'] = self.balanced_db_sampling
            self.train_info['checkpoint_metric'] = self.checkpoint_metric
            self.train_info['scorer_data_fusion_mode'] = self.scorer_data_fusion_mode
            self.train_info['scorer_cross_attn_layers'] = self.scorer_cross_attn_layers
            self.train_info['scorer_cross_attn_heads'] = self.scorer_cross_attn_heads
            self.train_info['planner_data_fusion_mode'] = self.planner_data_fusion_mode
            self.train_info['planner_cross_attn_actor_only'] = self.planner_cross_attn_actor_only
            self.train_info['planner_cross_attn_share_with_scorer'] = self.planner_cross_attn_share_with_scorer
            self.train_info['cross_attn_use_local_plan_q'] = self.cross_attn_use_local_plan_q
            self.train_info['cross_attn_q_feature_mask'] = self.cross_attn_q_feature_mask
            self.train_info['cross_attn_kv_feature_mask'] = self.cross_attn_kv_feature_mask
            self.train_info['cross_attn_q_use_query_feat'] = self.cross_attn_q_use_query_feat
            self.train_info['cross_attn_q_use_plan_nodes'] = self.cross_attn_q_use_plan_nodes
            self.train_info['use_db_features'] = self.use_db_features
            self.train_info['enable_db_meta'] = self.enable_db_meta
            self.train_info['scorer_use_risk_head'] = self.scorer_use_risk_head
            self.train_info['scorer_risk_loss_weight'] = self.scorer_risk_loss_weight
            self.train_info['scorer_risk_inference_weight'] = self.scorer_risk_inference_weight
            self.train_info['risk_negative_margin'] = self.risk_negative_margin
            self.train_info['negative_sample_weight'] = self.negative_sample_weight
            self.train_info['high_cost_weight_log_base'] = self.high_cost_weight_log_base
            self.train_info['high_cost_weight_cap'] = self.high_cost_weight_cap
            self.train_info['safe_select_enable'] = self.safe_select_enable
            self.train_info['safe_select_enable_train'] = self.safe_select_enable_train
            self.train_info['safe_select_enable_test'] = self.safe_select_enable_test
            self.train_info['safe_select_risk_threshold'] = self.safe_select_risk_threshold
            self.train_info['safe_select_min_adjusted_margin'] = self.safe_select_min_adjusted_margin
            self.train_info['safe_select_baseline_score_eps'] = self.safe_select_baseline_score_eps
            self.train_info['enable_train_safe_eval'] = self.enable_train_safe_eval
            self.train_info['data_drift_workload_source'] = self.data_drift_workload_source
            self.train_info['distill_best_plan_topk'] = self.distill_best_plan_topk
            self.train_info['distill_best_plan_weight'] = self.distill_best_plan_weight
            self.train_info['dynamic_hard_weight'] = self.dynamic_hard_weight
            self.train_info['dynamic_hard_weight_cap'] = self.dynamic_hard_weight_cap
            self.train_info['use_best_plan_pool'] = self.use_best_plan_pool
            self.train_info['bootstrap_best_plan_from_latency'] = self.bootstrap_best_plan_from_latency
            self.train_info['encoding_path'] = self.encoding_path
            self.train_info['auto_config'] = self.auto_config
            self.train_info['latency_buffer_path'] = self.latency_buffer_path
            self.train_info['best_plan_path'] = self.best_plan_path
            self.train_info['scorer_path'] = self.scorer_path
            self.train_info['agent_checkpoint'] = self.agent_checkpoint
            self.train_info['train_info_path'] = self.train_info_path
            if first_save is True:
                self.train_info['start_time'].append(start_time.strftime('%Y/%m/%d %H.%M.%S'))
                self.train_info['end_time'].append(end_time.strftime('%Y/%m/%d %H.%M.%S'))
                self.train_info['total_time'].append((end_time-start_time).total_seconds())
            else:
                self.train_info['start_time'][-1] = start_time.strftime('%Y/%m/%d %H.%M.%S')
                self.train_info['end_time'][-1] = end_time.strftime('%Y/%m/%d %H.%M.%S')
                self.train_info['total_time'][-1] = (end_time-start_time).total_seconds()
            if wrl_train is not None:
                self.train_info['model_saved_iter'] = train_iter
                self.train_info['model_wrl_train'] = wrl_train
                self.train_info['model_wrl_test'] = wrl_test
                self.train_info['model_gmrl_train'] = gmrl_train
                self.train_info['model_gmrl_test'] = gmrl_test
        if not os.path.exists(os.path.dirname(self.train_info_path)):
            os.makedirs(os.path.dirname(self.train_info_path))
        with open(self.train_info_path, 'w') as json_file:
            json.dump(self.train_info, json_file, ensure_ascii=False, indent=4)
        return self.train_info['start_time'][0],end_time

    def LoadTrainInfo(self,train_info_path):
        train_iter = 0
        if os.path.exists(train_info_path):
            with open(train_info_path, 'r') as f:
                self.train_info = json.load(f)
            self.time = self.train_info['time']
            self.expname = self.train_info.get('expname', self.expname)
            self.train_mode = self.train_info.get('train_mode', self.train_mode)
            self.model_root = self.train_info.get('model_root', './model/{}_{}/'.format(self.DBMS, self.mode))
            self.model_dir = self.train_info.get('model_dir', os.path.dirname(train_info_path))
            self.model_run_id = self.train_info.get('model_run_id', os.path.basename(self.model_dir))
            self.scorer_epochs = self.train_info['scorer_epochs']
            self.train_eval = self.train_info.get('train_eval', self.train_eval)
            self.train_eval_schedule = self.train_info.get('train_eval_schedule', self.train_eval_schedule)
            self.hybrid_sample_threshold = self.train_info.get('hybrid_sample_threshold', self.hybrid_sample_threshold)
            self.planner_batch_size = self.train_info['planner_batch_size']
            self.maxsteps = self.train_info['maxsteps']
            self.num_agents = self.train_info['num_agents']
            self.num_policies = self.train_info['num_policies']
            self.stop_warmup_iters = self.train_info.get('stop_warmup_iters', self.stop_warmup_iters)
            self.stop_min_steps = self.train_info.get('stop_min_steps', self.stop_min_steps)
            self.test_stop_min_steps = self.train_info.get('test_stop_min_steps', self.test_stop_min_steps)
            self.reward_use_total_time = self.train_info.get('reward_use_total_time', self.reward_use_total_time)
            self.balanced_db_sampling = self.train_info.get('balanced_db_sampling', self.balanced_db_sampling)
            self.checkpoint_metric = self.train_info.get('checkpoint_metric', self.checkpoint_metric)
            self.scorer_data_fusion_mode = self.train_info.get('scorer_data_fusion_mode', self.scorer_data_fusion_mode)
            self.scorer_cross_attn_layers = self.train_info.get('scorer_cross_attn_layers', self.scorer_cross_attn_layers)
            self.scorer_cross_attn_heads = self.train_info.get('scorer_cross_attn_heads', self.scorer_cross_attn_heads)
            self.planner_data_fusion_mode = self.train_info.get('planner_data_fusion_mode', self.planner_data_fusion_mode)
            self.planner_cross_attn_actor_only = self.train_info.get('planner_cross_attn_actor_only', self.planner_cross_attn_actor_only)
            self.planner_cross_attn_share_with_scorer = self.train_info.get('planner_cross_attn_share_with_scorer', self.planner_cross_attn_share_with_scorer)
            self.cross_attn_use_local_plan_q = self.train_info.get('cross_attn_use_local_plan_q', self.cross_attn_use_local_plan_q)
            self.cross_attn_q_feature_mask = self.train_info.get('cross_attn_q_feature_mask', self.cross_attn_q_feature_mask)
            self.cross_attn_kv_feature_mask = self.train_info.get('cross_attn_kv_feature_mask', self.cross_attn_kv_feature_mask)
            self.cross_attn_q_use_query_feat = self.train_info.get('cross_attn_q_use_query_feat', self.cross_attn_q_use_query_feat)
            self.cross_attn_q_use_plan_nodes = self.train_info.get('cross_attn_q_use_plan_nodes', self.cross_attn_q_use_plan_nodes)
            self.use_db_features = self.train_info.get('use_db_features', self.use_db_features)
            self.enable_db_meta = self.train_info.get('enable_db_meta', self.enable_db_meta)
            self.scorer_use_risk_head = self.train_info.get('scorer_use_risk_head', self.scorer_use_risk_head)
            self.scorer_risk_loss_weight = self.train_info.get('scorer_risk_loss_weight', self.scorer_risk_loss_weight)
            self.scorer_risk_inference_weight = self.train_info.get('scorer_risk_inference_weight', self.scorer_risk_inference_weight)
            self.risk_negative_margin = self.train_info.get('risk_negative_margin', self.risk_negative_margin)
            self.negative_sample_weight = self.train_info.get('negative_sample_weight', self.negative_sample_weight)
            self.high_cost_weight_log_base = self.train_info.get('high_cost_weight_log_base', self.high_cost_weight_log_base)
            self.high_cost_weight_cap = self.train_info.get('high_cost_weight_cap', self.high_cost_weight_cap)
            self.safe_select_enable = self.train_info.get('safe_select_enable', self.safe_select_enable)
            self.safe_select_enable_train = self.train_info.get('safe_select_enable_train', self.safe_select_enable_train)
            self.safe_select_enable_test = self.train_info.get('safe_select_enable_test', self.safe_select_enable_test)
            self.safe_select_risk_threshold = self.train_info.get('safe_select_risk_threshold', self.safe_select_risk_threshold)
            self.safe_select_min_adjusted_margin = self.train_info.get('safe_select_min_adjusted_margin', self.safe_select_min_adjusted_margin)
            self.safe_select_baseline_score_eps = self.train_info.get('safe_select_baseline_score_eps', self.safe_select_baseline_score_eps)
            self.enable_train_safe_eval = self.train_info.get('enable_train_safe_eval', self.enable_train_safe_eval)
            self.data_drift_workload_source = self.train_info.get('data_drift_workload_source', self.data_drift_workload_source)
            self.distill_best_plan_topk = self.train_info.get('distill_best_plan_topk', self.distill_best_plan_topk)
            self.distill_best_plan_weight = self.train_info.get('distill_best_plan_weight', self.distill_best_plan_weight)
            self.dynamic_hard_weight = self.train_info.get('dynamic_hard_weight', self.dynamic_hard_weight)
            self.dynamic_hard_weight_cap = self.train_info.get('dynamic_hard_weight_cap', self.dynamic_hard_weight_cap)
            self.use_best_plan_pool = self.train_info.get('use_best_plan_pool', self.use_best_plan_pool)
            self.bootstrap_best_plan_from_latency = self.train_info.get('bootstrap_best_plan_from_latency', self.bootstrap_best_plan_from_latency)
            self.maxbounty = self.train_info['maxbounty']
            self.penalty_coeff = self.train_info['penalty_coeff']
            self.planner_config_even = self.train_info['planner_config_even']
            self.planner_config_odd = self.train_info['planner_config_odd']
            self.entropy_coeff = self.train_info['entropy_coeff']
            self.kl_coeff = self.train_info['kl_coeff']
            self.lambda_ = self.train_info['lambda_']
            self.vf_share_layers = self.train_info['vf_share_layers']
            self.fcnet_hiddens = self.train_info['fcnet_hiddens']
            self.fcnet_activation = self.train_info['fcnet_activation']
            self.encoding_path = self.train_info.get('encoding_path', os.path.join(self.model_root, 'encoding.json'))
            self.auto_config = self.train_info.get('auto_config', os.path.join(self.model_dir, 'auto_config.json'))
            self.latency_buffer_path = self.train_info.get('latency_buffer_path', self.latency_buffer_path)
            self.best_plan_path = self.train_info.get('best_plan_path', self.best_plan_path)
            self.scorer_path = self.train_info.get('scorer_path', os.path.join(self.model_dir, 'scorer.pt'))
            self.agent_checkpoint = self.train_info.get('agent_checkpoint', os.path.join(self.model_dir, 'planner/'))
            self.train_info_path = self.train_info.get('train_info_path', train_info_path)
            train_iter = self.train_info['train_iter']
            if self.checkpoint_metric == 'test_speedup':
                model_wrl_train = self.train_info.get('model_wrl_test', self.train_info['model_wrl_train'])
                model_gmrl_train = self.train_info.get('model_gmrl_test', self.train_info['model_gmrl_train'])
            else:
                model_wrl_train = self.train_info.get('model_wrl_test', self.train_info['model_wrl_train'])
                model_gmrl_train = self.train_info.get('model_gmrl_test', self.train_info['model_gmrl_train'])
            if self.train_info['PlannerworkersNum'] + self.LearnerworkersNum + self.ScorerworkersNum*2 > self.cpu_resources:
                print(f"Insufficient CPU resources for resumed training. Please update the config:")
                if self.train_info['use_cpu_rate'] is True:
                    if self.train_info['CPU_thread_utilization_rate'] != self.CPU_thread_utilization_rate:
                        print(f"You can align resumed CPU thread utilization with the initial run:")
                        print(f"CPU_thread_utilization_rate: { self.CPU_thread_utilization_rate } -> initial:{ self.train_info['CPU_thread_utilization_rate'] }")
                else:
                    if self.train_info['PlannerworkersNum'] != self.PlannerworkersNum or self.train_info['LearnerworkersNum'] != self.LearnerworkersNum or self.train_info['ScorerworkersNum'] != self.ScorerworkersNum:
                        print(f"You can align resumed worker counts with the initial run:")
                        print(f"PlannerworkersNum: { self.PlannerworkersNum } -> initial:{ self.train_info['PlannerworkersNum'] }")
                        print(f"LearnerworkersNum: { self.LearnerworkersNum } -> initial:{ self.train_info['LearnerworkersNum'] }")
                        print(f"ScorerworkersNum: { self.ScorerworkersNum } -> initial:{ self.train_info['ScorerworkersNum'] }")
                raise ValueError(f"ERROR: insufficient CPU resources for resumed training!")
            else:
                print("Resumed training resource allocation:")
                use_resources = self.train_info['PlannerworkersNum'] + self.LearnerworkersNum + self.ScorerworkersNum*2
                print(f"CPU - Used: {use_resources:.1f} (Planner: {self.train_info['PlannerworkersNum']:.1f} + Learner: {self.LearnerworkersNum:.1f} + Scorer: {self.ScorerworkersNum*2:.1f}) / "
                f"Total: {self.cpu_resources:.1f}")
        else:
            raise FileNotFoundError(f"Training info file [{train_info_path}] does not exist")
        return train_iter, model_wrl_train, model_gmrl_train
    
    def GetTotalTime(self):
        total_seconds = sum(self.train_info['total_time'])
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        last_time = self.train_info['total_time'][-1]
        last_hours = int(last_time // 3600)
        last_minutes = int((last_time % 3600) // 60)
        last_seconds = int(last_time % 60)
        return [f"{hours}:{minutes:02d}:{seconds:02d}",f"{last_hours}:{last_minutes:02d}:{last_seconds:02d}"]

    
