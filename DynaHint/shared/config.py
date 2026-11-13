"""Runtime configuration and training metadata."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
from datetime import datetime
from multiprocessing import cpu_count
import json
import re


class Config:
    def __init__(self):
        # ======   DBMS Config   =======
        self.max_time_out = 300000  # Query timeout in milliseconds.
        # Applies to real SQL latency measurements; plan/cardinality calls are separate.
        self.latency_repeat_phases = ['baseline', 'experience', 'evaluation']
        self.latency_warmup_runs = 0
        self.latency_measure_runs = 1
        self.latency_aggregate = 'median'  # median or last
        # # ====== PostgreSQL ========
        self.time = datetime.now().strftime('%y%m%d')  # Current date in yymmdd format.
        self.expname = '{}_stop2_full'.format(self.time)
        self.DBMS = 'postgres'
        # self.mode = 'JOB'
        self.mode = 'STATS'
        # self.mode = 'TPCDS'
        self.ip = 'localhost'
        self.port  = 5432
        self.user = ''
        self.password = ''
        # self.databases = ['imdb1980cut_rel_d1','imdb1980cut_rel_d2','imdb1980cut_rel_d3','imdb1980cut_rel_d4','imdb1980cut_rel_d5','imdb1980cut','imdb2000cut','imdb','imdb1940cut','imdb1960cut','imdb_nb_1980base_d01_sf05','imdb_nb_1980base_d03_sf07','imdb_nb_1980base_d05_sf09']
        

        # Full training uses five generated databases followed by the source database.
        self.databases = ['stats_201306_rel_d1','stats_201306_rel_d2','stats_201306_rel_d3','stats_201306_rel_d4','stats_201306_rel_d5','stats_201306','stats_201206','stats_201212','stats_201312','stats','stats_nb_201306dir_d01_sf05','stats_nb_201306dir_d03_sf07','stats_nb_201306dir_d05_sf09']

        # self.databases = ['tpcds_200012_rel_d1','tpcds_200012_rel_d2','tpcds_200012_rel_d3','tpcds_200012_rel_d4','tpcds_200012_rel_d5','tpcds_200012','tpcds_199812','tpcds_199912','tpcds_200112','tpcds','tpcds_nb_200012dir_d01_sf05','tpcds_nb_200012dir_d03_sf07','tpcds_nb_200012dir_d05_sf09']

        # Cardinality-label extraction mode.
        # count_rewrite: rewrite count(*) as SELECT 1 and read the EXPLAIN root row count.
        # aggregate_input: read rows before aggregation, falling back to the root for non-aggregate queries.
        self.cardinality_extract_mode = 'aggregate_input'
      
        # ====== Embed Config ========
        self.train_mode = 'mix+'  # data_drift, query_drift, mix, mix+
        self.train_data_rate = 0.8
        self.train_databases = 6
        self.data_drift_workload_source = 'train'
        self.AutoGetParam = True
        self.types = 20  # Number of node types.
        self.columns = 63  # Number of columns.
        self.heightsize = 29  # Automatic dimension discovery raises heightsize to at least maxnode + 1.
        self.maxnode = 45  # Maximum plan-node capacity; automatic discovery can increase it.
        self.maxjoins = 10  # Maximum joins.
        self.max_logical_edges = 32  # Padded logical join-edge capacity, updated by automatic dimension discovery.
        self.tablenum = 21  # Number of tables.
        self.opsnum = 18  # Number of operators.
        self.filtmaxnum = 6  # Maximum filters.
        self.maxpos = 3+2  # Maximum tree branching factor.
        # ======     Model    ========
        self.emb_size = 32    # Embedding dimension.
        self.ffn_dim = 32     # Feed-forward dimension.
        self.head_size = 10   # Attention-head dimension.
        self.num_layers = 10  # Number of attention layers.
        self.dropout = 0.05   # Dropout rate.
        self.hidden_dim = self.emb_size * 7 + 8 * (self.emb_size // 8) + self.emb_size // 2 + 1  # Hidden dimension.
        # ====== Database Features ========
        # If you enable this, make sure your env/PlanHelper attaches the following keys into obs:
        #   - 'db_hist'   : shape [db_hist_dim]
        #   - 'query_feat': shape [query_feat_dim] (optional, e.g., table mask + simple stats)
        self.use_db_features = True
        self.use_db_bitmap_features = False
        # Histogram generation: how many values we resample each column's histogram_bounds into
        self.db_hist_bins_per_col = 51
        self.node_hist_dim = 3 * (self.db_hist_bins_per_col - 1)
        self.db_bitmap_samples_per_table = 1000        
        self.num_node_feature = 7 + self.maxjoins + 5 * self.filtmaxnum + self.node_hist_dim  # Node-feature dimension.
        self.db_hist_dim = self.columns * self.db_hist_bins_per_col      # e.g., (#columns * #bins) or any fixed-length vector
        self.db_bitmap_dim = self.tablenum * self.db_bitmap_samples_per_table
        self.query_feat_dim = self.tablenum + 1 + 2  # default: table-mask + (NA, num_joins, num_filters, db_id?, 

        # Optional: path to a pre-saved .npz that stores per-database features.
        self.db_feature_npz = "./db_features/db_features.npz"
        # ===== Feature Sources =====
        # Set to 'bank' to use db_feature_npz bank (older path).
        self.db_feature_source = 'DynaHint'
        self.workload_file_name = './experiment/JOBRand/synthetic'
        # collect.py histogram json dir: <dir>/<db>_histogram_string.json
        self.hist_json_dir = './experiment/histogram'
        self.bitmap_pickle_dir = './experiment/sample'
        self.sample_size = 1000
        # ====== DB meta / drift features (table/column stats) ======
        # Enable database statistics for data-drift and query-drift features.
        self.enable_db_meta = True
        self.refresh_column_range_cache = False  # Force column-range collection after rebuilding databases under the same names.
        # Table-feature dimension, with one token per table.
        # [log1p(exact_row_count), log1p(total_bytes_mb), log1p(relpages), dead_ratio, tanh(stats_age_days/30), has_stats]
        self.db_table_feat_dim = 6
        # Database-wide feature dimension.
        # [log1p(sum_exact_row_count), log1p(total_bytes_mb), mean_dead_ratio, mean_stats_age_norm]
        self.db_global_feat_dim = 4
        # Query-column statistics summary dimension.
        # [mean_log1p(ndv), mean_null_frac, mean_log1p(avg_width), mean_mcv_freq1]
        self.db_col_feat_dim = 4

        # Transformer database-token encoder settings.
        self.db_token_hidden_dim = 128
        self.db_token_nhead = 4
        self.db_token_layers = 2
        self.db_token_ffn_mult = 4
        self.max_tables_in_schema = 256  # Schema-table capacity for positional embeddings.
        # ====== Train Config ========
        self.total_iter = 2000  # Total training iterations.
        self.scorer_epochs = 20  # Initial scorer-training epochs.
        self.scorer_incremental_epochs = 3  # Use a few incremental epochs after the initial scorer fit.
        self.scorer_reinitialize_each_train = False  # Preserve scorer weights and optimizer state across incremental updates.
        self.scorer_batch_size = 128  # Scorer-training batch size.
        self.pair_lr = 3e-4  # Scorer learning rate.
        self.maxsteps = 5  # Maximum search steps; the policy may STOP earlier.
        self.action_mode = 'join_order_and_operator'
        self.set_hint_gucs = [
            'enable_hashjoin',
            'enable_mergejoin',
            'enable_nestloop',
            'enable_indexscan',
            'enable_seqscan',
            'enable_indexonlyscan',
        ]
        self.set_size = 2 ** len(self.set_hint_gucs)
        # Keep all scan methods enabled and let the planner choose only the
        # global join-method combination.
        self.hint_sets_switch = [15, 23, 31, 39, 47, 55, 63]
        # self.hint_sets_switch = [13, 14, 15, 23, 27, 28, 31, 39, 42, 46, 47, 50, 55, 63]
        # Count SQL in the actual training workload, including shared data-drift workloads.
        workload_source_dir = 'train'
        if self.train_mode == 'data_drift':
            if self.data_drift_workload_source not in ('train', 'test'):
                raise ValueError("data_drift_workload_source must be 'train' or 'test'")
            workload_source_dir = self.data_drift_workload_source
        self.train_sql_count = len([
            f for f in os.listdir('./experiment/{}/{}/'.format(self.mode, workload_source_dir))
            if f.endswith('.sql')
        ])
        self.planner_batch_size = self.train_sql_count*self.maxsteps+100  # Planner-training batch size.
        self.planner_num_sgd_iter = 15  # PPO SGD passes per training batch.
        self.num_agents = 1  # Number of agents.
        self.num_policies = 1  # Number of policies.
        self.stop_warmup_iters = 100  # Disable STOP in early episodes to prevent premature termination.
        self.stop_min_steps = 2  # Minimum exploration steps before training permits STOP.
        self.test_stop_min_steps = 2  # Minimum search steps before evaluation permits STOP.
        self.selection_strategy_diagnostics = True  # Evaluate all candidates to record oracle, scorer and safe-selection diagnostics.
        self.reward_use_total_time = False  # Use execution plus planning time if True; otherwise use execution time only.
        self.balanced_db_sampling = True  # Balance scorer samples across databases to avoid overrepresenting frequent databases.
        self.checkpoint_metric = 'train_wrl'  # WRL excludes planning time; train_wrl_withPlanning includes it.
        self.scorer_use_baseline_context = True  # Score baseline, candidate and delta features jointly to reduce cross-database bias.
        self.scorer_data_fusion_mode = 'cross_attention'  # Choose concatenation or query-conditioned cross-attention for the scorer.
        self.scorer_cross_attn_layers = 1  # Data-memory encoder layers; keep the encoder small to limit overfitting.
        self.scorer_cross_attn_heads = 3  # Attention heads; the implementation reduces this to a divisor of hidden_dim if needed.
        self.planner_data_fusion_mode = 'cross_attention'  # Choose concatenation or query-conditioned cross-attention for the planner.
        self.planner_cross_attn_actor_only = True  # Whether only the actor uses cross-attention features.
        self.cross_attn_use_local_plan_q = False  # Use local plan queries with node histograms, without repeated global-drift fusion.
        self.scorer_use_risk_head = True  # Learn regression risk for conservative candidate scoring at inference.
        self.scorer_label_clip = 8.0  # Absolute bound for log-gain training labels, not inference scores.
        self.scorer_risk_loss_weight = 0.2  # Risk-head training-loss weight.
        self.scorer_risk_inference_weight = 0.1  # Risk penalty subtracted from the inference score.
        self.planner_surrogate_risk_weight = 0.0  # Use raw scorer gains for planner surrogate rewards unless risk is calibrated.
        self.scorer_rollout_update_mode = 'planner_boundary'  # When rollout workers load scorer updates: immediate, planner_boundary or frozen.
        self.planner_reward_diagnostics = True  # Compare surrogate rewards with real-latency rewards when measurements exist.
        self.risk_negative_margin = 1.05  # Label a candidate risky only when its objective exceeds the baseline by 5%.
        self.negative_sample_weight = 2.0  # Additional weight for regression samples to protect long-query workload performance.
        self.high_cost_weight_log_base = 100.0  # Baseline-latency weighting scale; smaller values emphasize long queries.
        self.high_cost_weight_cap = 3.0  # Cap extra weights so extremely slow queries do not dominate training.
        self.safe_select_enable = True  # Fall back to the baseline when candidate risk is high and gain is uncertain.
        self.safe_select_enable_train = False  # Disable conservative fallback during Train validation to expose learned gains.
        self.safe_select_enable_test = True  # Enable conservative fallback for cross-database evaluation.
        self.safe_select_risk_threshold = 0.65  # Risk probability above which conservative selection is checked.
        self.safe_select_min_adjusted_margin = 0.05  # Required adjusted-score margin over baseline for high-risk candidates.
        self.safe_select_baseline_score_eps = -0.02  # Allow baseline fallback when its adjusted score is close to the best candidate.
        self.validation_time_seconds = self._normalize_validation_time(None)  # Cumulative wall time for the three formal validation phases.
        self.distill_best_plan_topk = 2  # Number of top best-plan samples emphasized per database and query.
        self.distill_best_plan_weight = 1.5  # Distillation weight for top best-plan samples.
        self.dynamic_hard_weight = True  # Increase weights for high-regret samples without fixed query-family priors.
        self.dynamic_hard_weight_cap = 2.0  # Cap extra hard-sample weights to prevent a few queries dominating training.
        self.use_best_plan_pool = False  # Enable reuse of persisted best-plan JSON records.
        # ====== Ablation Config ======
        self.cross_attn_q_feature_mask = 'all'  # Ablation mask for database and query signals in the Q-side PlanNetwork.
        self.cross_attn_kv_feature_mask = 'all'  # Ablation mask for database signals in K/V data memory.
        self.cross_attn_q_use_query_feat = True  # Include the query-feature token in cross-attention queries.
        self.cross_attn_q_use_plan_nodes = True  # Include plan-node or node-summary tokens in cross-attention queries.
        # Ablation switches default to False; configure variants centrally here.
        self.ablation_name = 'Full'
        self.ablation_disable_cardinality_aux_loss = False  # Disable cardinality auxiliary supervision for scorer and planner.
        self.ablation_disable_risk_head = False  # Disable risk-head training and inference penalties.
        self.ablation_disable_safe_select = False  # Disable safe-selection fallback.
        self.ablation_disable_db_state_features = False  # Disable database-state inputs.
        self.ablation_disable_global_context = False  # Bypass global context, retain node-local encoding.
        self.ablation_disable_topk_best_weighting = False  # Disable top-k best-plan distillation weighting.
        self.ablation_disable_negative_risk_weighting = False  # Disable extra weights for regression and risk samples.
        self.ablation_disable_hard_sample_weighting = False  # Disable long-query and regret-based hard-sample weighting.
        self.ablation_disable_db_balancing = False  # Disable database-balanced scorer sampling.
        self.use_cardinality_aux_loss = not self.ablation_disable_cardinality_aux_loss
        if self.ablation_disable_risk_head:
            self.scorer_use_risk_head = False
            self.scorer_risk_loss_weight = 0.0
            self.scorer_risk_inference_weight = 0.0
        if self.ablation_disable_safe_select:
            self.safe_select_enable = False
            self.safe_select_enable_train = False
            self.safe_select_enable_test = False
        if self.ablation_disable_db_state_features:
            self.use_db_features = False
            self.enable_db_meta = False
            self.node_hist_dim = 0
            self.cross_attn_q_feature_mask = ''
            self.cross_attn_kv_feature_mask = ''
        if self.ablation_disable_topk_best_weighting:
            self.distill_best_plan_topk = 0
            self.distill_best_plan_weight = 1.0
        if self.ablation_disable_negative_risk_weighting:
            self.negative_sample_weight = 1.0
        if self.ablation_disable_hard_sample_weighting:
            self.high_cost_weight_cap = 1.0
            self.dynamic_hard_weight = False
            self.dynamic_hard_weight_cap = 1.0
        if self.ablation_disable_db_balancing:
            self.balanced_db_sampling = False

        self.maxbounty = 8  # Maximum bounty.
        self.penalty_coeff = 0.0  # Explicit step penalty; zero leaves exploration depth unpenalized.
        self.planner_config_even = { # Planner settings for even-numbered policies.
            "gamma_": 0.99,
            "lr": 5e-5,
        }
        self.planner_config_odd = { # Planner settings for odd-numbered policies.
            "gamma_": 0.95,
            "lr": 1e-4,
        }
        self.entropy_coeff = 0.01  # Planner entropy coefficient.
        self.kl_coeff = 1.0  # Planner KL-divergence coefficient.
        self.lambda_ = 0.9  # Planner GAE lambda.
        self.alpha = 0.5  # Loss weight.
        self.vf_share_layers = True  # Share planner value-function layers.
        self.fcnet_hiddens = [384, 384, 384]  # Planner hidden-layer dimensions.
        self.fcnet_activation = "tanh"  # Planner activation function.
        # ====== General ========
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Compute device.
        self.PlannerworkersNum = 1  # Planner worker count.
        self.LearnerworkersNum = 0  # Learner scheduling worker count; zero disables scheduling resources.
        self.ScorerworkersNum = 1  # Scorer scheduling worker count; two scorer actors are created.
        self.use_cpu_rate = True  # Allocate workers by a fraction of available CPU threads.
        # Each rollout worker opens a PostgreSQL connection to every configured database.
        # High worker counts can exhaust PostgreSQL max_connections.
        # Limit concurrency to keep startup connections within the server capacity.
        self.CPU_thread_utilization_rate = 0.3  # CPU-thread fraction; account for per-worker, per-database PostgreSQL connections.
        self.get_embed_batch = 16  # Embedding batch size; large values increase memory requirements.
        self.seed = 2026  # Random seed.
        self.recommend_threshold = 0.05  # Minimum relative gain required to recommend optimization.
        self.adaptive_gate_mode = 'nearest'  # off / baseline / nearest; never gates Train.
        self.nearest_tag_model_path = ''  # Empty: train from Train records; evaluation loads checkpoint copy.
        self.adaptive_gate_min_records = 6
        self.initialPro = 0.2  # Initial sampling fraction for scorer initialization.
        self.train_eval = 25  # Default number of newly executed samples before scorer retraining.
        self.train_eval_schedule = [[0, 25], [100, 50], [300, 75]]  # Increase retraining thresholds over time for more stable later updates.
        self.val_freq = 20  # Validation and model-save frequency.
        self.validate_test_only_on_train_improvement = True  # Evaluate Test and TestDataDrift only when Train WRL or GMRL improves.
        self.full_validation_after_iter = 0  # After this iteration, evaluate Test and TestDataDrift at every validation point.
        self.tmp_checkpoint_freq = 20  # Temporary planner-checkpoint frequency.
        self.start_val = 20  # First recorded validation iteration.
        self.splitpoint = [0.5, 0.05]  # Class split points.
        self.classNum = len(self.splitpoint)  # Number of classes.
        self.timeoutcoeff = 1.01 + self.splitpoint[0]  # Execution-timeout coefficient.
        self.cost_plan_timeoutcoeff = 2.0  # Candidate-planning timeout relative to the baseline cost-plan time.
        self.cost_plan_timeout_min_ms = 60000  # Minimum EXPLAIN timeout in milliseconds to avoid overly short baseline-derived limits.
        self.maxsamples = 5  # Maximum samples.
        self.active_sample_once_per_planner_iter = True  # Consume active-sampling candidates once per PPO iteration.
        self.sample_strategy = 'hybrid'  # Sampling strategy: hybrid, uncertainty, random or hybrid_global.
        self.hybrid_sample_threshold = 0.9  # Higher confidence thresholds make empty sampling batches less likely.
        self.update_scorer = True  # Update the scorer.
        # Optional scorer-update override.
        self.left_deep_restriction = True  # Restrict join trees to left-deep structures.

        # ====== Basic ========
        self.operator_pg2hint = {'Hash Join':'HASHJOIN','Merge Join': 'MERGEJOIN','Nested Loop':'NESTLOOP'} # Map PostgreSQL operators to hint operators.
        self.OperatorDict = {'NESTLOOP':0,'HASHJOIN':1,'MERGEJOIN':2}   # Map hint operators to numeric IDs.
        self.Operatortype = ['NESTLOOP','HASHJOIN','MERGEJOIN']  # Hint-operator names.

    @property
    def adaptive_gate_enable(self):
        return self.adaptive_gate_mode != 'off'

    @adaptive_gate_enable.setter
    def adaptive_gate_enable(self, enabled):
        # Compatibility for existing experiment overrides and old checkpoints.
        if not enabled:
            self.adaptive_gate_mode = 'off'
        elif getattr(self, 'adaptive_gate_mode', 'off') == 'off':
            self.adaptive_gate_mode = 'baseline'

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

    def ResolveBestCheckpointPaths(self, model_path, metric='train_wrl'):
        aliases = {
            'wrl': 'train_wrl',
            'gmrl': 'train_gmrl',
            'train_wrl_noPlanningTime': 'train_wrl',
            'train_wrl_total': 'train_wrl_withPlanning',
            'test_wrl_total': 'test_wrl_withPlanning',
        }
        metric = aliases.get(metric, metric)
        record = None
        train_info_path = os.path.join(model_path, 'trainfo.json')
        if os.path.exists(train_info_path):
            with open(train_info_path, 'r', encoding='utf-8') as file_obj:
                train_info = json.load(file_obj)
            self._require_current_train_info(train_info, train_info_path)
            best_checkpoints = train_info.get('best_checkpoints', {})
            record = best_checkpoints.get(metric)
            if record is None and metric == 'train_wrl':
                record = best_checkpoints.get('train_wrl_noPlanningTime')

        if record:
            if 'scorer_path' not in record:
                raise ValueError(
                    f'旧 checkpoint 元数据不兼容: {train_info_path} 缺少 scorer_path；'
                    '需要新格式的 trainfo.json 和 scorer.pt。'
                )
            planner_path = record['planner_path']
            scorer_path = record['scorer_path']
            if not os.path.isabs(planner_path):
                planner_path = os.path.join(model_path, planner_path)
            if not os.path.isabs(scorer_path):
                scorer_path = os.path.join(model_path, scorer_path)
            return os.path.normpath(planner_path), os.path.normpath(scorer_path)
        return os.path.join(model_path, 'planner'), os.path.join(model_path, 'scorer.pt')

    @staticmethod
    def _require_current_train_info(train_info, path):
        legacy_fields = sorted(
            key for key in train_info
            if key.startswith('evaluator_') or key == 'EvaluatorworkersNum'
        )
        if legacy_fields:
            raise ValueError(
                f'旧训练元数据不兼容: {path} 使用 {legacy_fields}；'
                '当前代码要求 scorer_epochs/scorer_path，且旧模型路径 evaluator.pt 不再自动识别。'
            )
        for record in train_info.get('best_checkpoints', {}).values():
            if isinstance(record, dict) and 'evaluator_path' in record:
                raise ValueError(
                    f'旧 checkpoint 元数据不兼容: {path} 使用 evaluator_path；'
                    '当前代码要求 scorer_path。'
                )

    def ConfirmPath(self):
        self.model_root = './model/{}_{}/'.format(self.DBMS, self.mode)
        self.model_run_id = self.BuildModelRunId()
        self.model_dir = os.path.join(self.model_root, self.model_run_id)
        # Training-workload directory.
        self.train_workload_path = './experiment/{}/train/'.format(self.mode)
        # Test-workload directory.
        self.test_workload_path = './experiment/{}/test/'.format(self.mode)
        # Persistent encoding dictionaries for columns, tables and operators.
        self.encoding_path = os.path.join(self.model_root, 'encoding.json')
        # Scorer weights.
        self.scorer_path = os.path.join(self.model_dir, 'scorer.pt')
        # Planner/RLlib checkpoint directory.
        self.agent_checkpoint = os.path.join(self.model_dir, 'planner/')
        # One metadata NPZ per database to avoid repeated catalog queries.
        self.db_meta_cache_dir = os.path.join(self.model_root, 'dbmeta/')
        # Latency and cardinality buffer for this run.
        self.latency_buffer_path = './latencybuffer/{}/{}_buffer_{}.json'.format(self.time,self.mode,self.expname)
        # Historical PostgreSQL latency file used to initialize a missing run buffer.
        safe_train_mode = self._safe_path_component(self.train_mode)
        latency_name = '{}_{}'.format(self.mode, safe_train_mode)
        latency_dir = './latency/{}'.format(self.mode)
        self.pg_latency = os.path.join(latency_dir, '{}.json'.format(latency_name))
        self.cardinality_cache_path = os.path.join(latency_dir, '{}_cardinality.jsonl'.format(self.mode))
        # Persist one best-plan record per database and query for reuse across runs.
        self.best_plan_path = os.path.join(latency_dir, '{}_{}_best_plan.json'.format(self.mode, safe_train_mode))
        # Per-iteration workload metrics output.
        self.eval_output_path = './timely_result/{}.json'.format(self.mode)
        # Execution and planning times for baseline and validation phases.
        self.outfile_path = './timely_result/{}_{}.json'.format(self.mode,self.expname)
        # TensorBoard and runtime-state directory.
        self.runstate = './runstate/{}_{}_{}'.format(self.DBMS,self.mode,self.expname)
        # Automatically discovered structural dimensions.
        self.auto_config = os.path.join(self.model_dir, 'auto_config.json')
        # Training metadata used when resuming.
        self.train_info_path = os.path.join(self.model_dir, 'trainfo.json')
        # Inference-result export directory, typically containing CSV files.
        self.opti_result_path = './result/{}_{}/'.format(self.DBMS,self.mode)

    def get_current_train_eval(self, train_times=0):
        # Choose the next retraining sample threshold using the scorer-update count.
        current_train_eval = self.train_eval
        for min_train_times, scheduled_train_eval in self.train_eval_schedule:
            if train_times >= min_train_times:
                current_train_eval = scheduled_train_eval
            else:
                break
        return current_train_eval

    def CpuDeployment(self,print_info = True, type = None):
        """Allocate CPU resources and optionally print the worker allocation."""
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
                print(f"CPU - allocated: {self.use_resources:.1f} (Planner: {self.PlannerworkersNum:.1f} + Learner: {self.LearnerworkersNum:.1f} + Scorer: {self.ScorerworkersNum*2:.1f}) / "
                f"total: {self.cpu_resources:.1f}")
        else:
            scorer_rate = 0. if type == 'test' else 0.2
            if type == 'test':
                self.ScorerworkersNum = 0
            if self.PlannerworkersNum + self.LearnerworkersNum + self.ScorerworkersNum*2 > self.cpu_resources:
                self.LearnerworkersNum = int(self.cpu_resources*0.2)
                self.ScorerworkersNum = int(self.cpu_resources*scorer_rate)
                self.PlannerworkersNum = int(self.cpu_resources - self.LearnerworkersNum - self.ScorerworkersNum*2)
                if print_info is True:
                    print("Insufficient CPU resources; worker allocation was adjusted to available capacity")
                    print(f"CPU - allocated: {self.cpu_resources:.1f} (Planner: {self.PlannerworkersNum:.1f} + Learner: {self.LearnerworkersNum:.1f} + Scorer: {self.ScorerworkersNum*2:.1f}) / "
                    f"total: {self.cpu_resources:.1f}")

    @staticmethod
    def _normalize_model_metrics(model_metrics):
        normalized = {}
        for phase in ('Train', 'Test', 'TestDataDrift'):
            phase_metrics = model_metrics.get(phase, {}) if isinstance(model_metrics, dict) else {}
            legacy_metrics = 'WRL_noPlanningTime' in phase_metrics and 'WRL_withPlanning' not in phase_metrics
            wrl = phase_metrics.get('WRL_noPlanningTime') if legacy_metrics else phase_metrics.get('WRL')
            wrl_with_planning = phase_metrics.get('WRL') if legacy_metrics else phase_metrics.get('WRL_withPlanning')
            gmrl = phase_metrics.get('GMRL')
            normalized[phase] = {
                'WRL': float(wrl) if wrl is not None else None,
                'GMRL': float(gmrl) if gmrl is not None else None,
                'WRL_withPlanning': float(wrl_with_planning) if wrl_with_planning is not None else None,
            }
        return normalized

    @staticmethod
    def _normalize_validation_time(validation_time):
        source = validation_time if isinstance(validation_time, dict) else {}
        normalized = {
            phase: max(float(source.get(phase, 0.0)), 0.0)
            for phase in ('Train', 'Test', 'TestDataDrift')
        }
        normalized['Total'] = sum(normalized.values())
        return {
            'Total': normalized['Total'],
            'Train': normalized['Train'],
            'Test': normalized['Test'],
            'TestDataDrift': normalized['TestDataDrift'],
        }

    def RecordValidationTime(self, phase, elapsed_seconds):
        if phase not in ('Train', 'Test', 'TestDataDrift'):
            raise ValueError(f'Unsupported validation phase: {phase}')
        self.validation_time_seconds = self._normalize_validation_time(self.validation_time_seconds)
        self.validation_time_seconds[phase] += max(float(elapsed_seconds), 0.0)
        self.validation_time_seconds['Total'] = sum(
            self.validation_time_seconds[name]
            for name in ('Train', 'Test', 'TestDataDrift')
        )
        return dict(self.validation_time_seconds)

    def SaveTrainInfo(self, train_iter=0, start_time=None, first_save=False, best_checkpoints=None):
        """Save training metadata, elapsed time and best-checkpoint paths and metrics."""
        self.train_info = {}
        end_time = datetime.now()
        if train_iter==1 or not os.path.exists(self.train_info_path):
            self.train_info = {
                'start_time': [start_time.strftime('%Y/%m/%d %H.%M.%S')],  # Start time.
                'end_time': [end_time.strftime('%Y/%m/%d %H.%M.%S')],  # End time.
                'time': self.time,  # Time.
                'expname': self.expname,  # Experiment name.
                'train_mode': self.train_mode,  # Train mode.
                'model_root': self.model_root,  # Model root.
                'model_run_id': self.model_run_id,  # Model run id.
                'model_dir': self.model_dir,  # Model dir.
                'train_iter': 1,  # Train iter.
                'best_checkpoints': best_checkpoints or {},
                'validate_test_only_on_train_improvement': self.validate_test_only_on_train_improvement,
                'full_validation_after_iter': self.full_validation_after_iter,
                'validation_time_seconds': self._normalize_validation_time(self.validation_time_seconds),
                'scorer_epochs': self.scorer_epochs,  # Initial scorer-training epochs.
                'scorer_incremental_epochs': self.scorer_incremental_epochs,
                'scorer_reinitialize_each_train': self.scorer_reinitialize_each_train,
                'train_eval': self.train_eval,  # Train eval.
                'train_eval_schedule': self.train_eval_schedule,  # Train eval schedule.
                'hybrid_sample_threshold': self.hybrid_sample_threshold,  # Hybrid sample threshold.
                'planner_batch_size': self.planner_batch_size,  # Planner-training batch size.
                'planner_num_sgd_iter': self.planner_num_sgd_iter,  # Planner num sgd iter.
                'maxsteps': self.maxsteps,  # Maximum search steps.
                'action_mode': self.action_mode,
                'set_hint_gucs': self.set_hint_gucs,
                'set_size': self.set_size,
                'hint_sets_switch': self.hint_sets_switch,
                'num_agents': self.num_agents,  # Number of agents.
                'num_policies': self.num_policies,  # Number of policies.
                'stop_warmup_iters': self.stop_warmup_iters,  # Stop warmup iters.
                'stop_min_steps': self.stop_min_steps,  # Stop min steps.
                'test_stop_min_steps': self.test_stop_min_steps,  # Test stop min steps.
                'selection_strategy_diagnostics': self.selection_strategy_diagnostics,
                'reward_use_total_time': self.reward_use_total_time,  # Reward use total time.
                'balanced_db_sampling': self.balanced_db_sampling,  # Balanced db sampling.
                'checkpoint_metric': self.checkpoint_metric,  # Checkpoint metric.
                'tmp_checkpoint_freq': self.tmp_checkpoint_freq,
                'active_sample_once_per_planner_iter': self.active_sample_once_per_planner_iter,
                'cost_plan_timeoutcoeff': self.cost_plan_timeoutcoeff,
                'cost_plan_timeout_min_ms': self.cost_plan_timeout_min_ms,
                'scorer_data_fusion_mode': self.scorer_data_fusion_mode,  # Scorer data fusion mode.
                'scorer_cross_attn_layers': self.scorer_cross_attn_layers,  # Scorer cross attn layers.
                'scorer_cross_attn_heads': self.scorer_cross_attn_heads,  # Scorer cross attn heads.
                'planner_data_fusion_mode': self.planner_data_fusion_mode,  # Planner data fusion mode.
                'planner_cross_attn_actor_only': self.planner_cross_attn_actor_only,  # Planner cross attn actor only.
                'cross_attn_use_local_plan_q': self.cross_attn_use_local_plan_q,  # Cross attn use local plan q.
                'ablation_disable_global_context': self.ablation_disable_global_context,
                'cross_attn_q_feature_mask': self.cross_attn_q_feature_mask,  # Cross attn q feature mask.
                'cross_attn_kv_feature_mask': self.cross_attn_kv_feature_mask,  # Cross attn kv feature mask.
                'cross_attn_q_use_query_feat': self.cross_attn_q_use_query_feat,  # Cross attn q use query feat.
                'cross_attn_q_use_plan_nodes': self.cross_attn_q_use_plan_nodes,  # Cross attn q use plan nodes.
                'use_db_features': self.use_db_features,  # Use db features.
                'use_db_bitmap_features': self.use_db_bitmap_features,
                'enable_db_meta': self.enable_db_meta,  # Enable db meta.
                'scorer_use_risk_head': self.scorer_use_risk_head,  # Scorer use risk head.
                'scorer_label_clip': self.scorer_label_clip,
                'scorer_risk_loss_weight': self.scorer_risk_loss_weight,  # Scorer risk loss weight.
                'scorer_risk_inference_weight': self.scorer_risk_inference_weight,  # Scorer risk inference weight.
                'planner_surrogate_risk_weight': self.planner_surrogate_risk_weight,  # Planner surrogate risk weight.
                'scorer_rollout_update_mode': self.scorer_rollout_update_mode,  # Scorer rollout update mode.
                'planner_reward_diagnostics': self.planner_reward_diagnostics,  # Planner reward diagnostics.
                'risk_negative_margin': self.risk_negative_margin,  # Risk negative margin.
                'negative_sample_weight': self.negative_sample_weight,  # Negative sample weight.
                'high_cost_weight_log_base': self.high_cost_weight_log_base,  # High cost weight log base.
                'high_cost_weight_cap': self.high_cost_weight_cap,  # High cost weight cap.
                'safe_select_enable': self.safe_select_enable,  # Safe select enable.
                'safe_select_enable_train': self.safe_select_enable_train,  # Safe select enable train.
                'safe_select_enable_test': self.safe_select_enable_test,  # Safe select enable test.
                'safe_select_risk_threshold': self.safe_select_risk_threshold,  # Safe select risk threshold.
                'safe_select_min_adjusted_margin': self.safe_select_min_adjusted_margin,  # Safe select min adjusted margin.
                'safe_select_baseline_score_eps': self.safe_select_baseline_score_eps,  # Safe select baseline score eps.
                'data_drift_workload_source': self.data_drift_workload_source,  # Data drift workload source.
                'distill_best_plan_topk': self.distill_best_plan_topk,  # Distill best plan topk.
                'distill_best_plan_weight': self.distill_best_plan_weight,  # Distill best plan weight.
                'dynamic_hard_weight': self.dynamic_hard_weight,  # Dynamic hard weight.
                'dynamic_hard_weight_cap': self.dynamic_hard_weight_cap,  # Dynamic hard weight cap.
                'use_best_plan_pool': self.use_best_plan_pool,  # Use best plan pool.
                'ablation_name': self.ablation_name,  # Ablation name.
                'ablation_disable_cardinality_aux_loss': self.ablation_disable_cardinality_aux_loss,
                'ablation_disable_risk_head': self.ablation_disable_risk_head,
                'ablation_disable_safe_select': self.ablation_disable_safe_select,
                'ablation_disable_db_state_features': self.ablation_disable_db_state_features,
                'ablation_disable_topk_best_weighting': self.ablation_disable_topk_best_weighting,
                'ablation_disable_negative_risk_weighting': self.ablation_disable_negative_risk_weighting,
                'ablation_disable_hard_sample_weighting': self.ablation_disable_hard_sample_weighting,
                'ablation_disable_db_balancing': self.ablation_disable_db_balancing,
                'use_cardinality_aux_loss': self.use_cardinality_aux_loss,
                'maxbounty': self.maxbounty,  # Maximum bounty.
                'penalty_coeff': self.penalty_coeff,  # Penalty coefficient.
                'planner_config_even': self.planner_config_even,  # Planner settings for even-numbered policies.
                'planner_config_odd': self.planner_config_odd,  # Planner settings for odd-numbered policies.
                'entropy_coeff': self.entropy_coeff,  # Planner entropy coefficient.
                'kl_coeff': self.kl_coeff,  # Planner KL-divergence coefficient.
                'lambda_': self.lambda_,  # Planner GAE lambda.
                'vf_share_layers': self.vf_share_layers,  # Share planner value-function layers.
                'fcnet_hiddens': self.fcnet_hiddens,  # Planner hidden-layer dimensions.
                'fcnet_activation': self.fcnet_activation,  # Planner activation function.
                'encoding_path': self.encoding_path,  # Encoding path.
                'auto_config': self.auto_config,  # Auto config.
                'latency_buffer_path': self.latency_buffer_path,  # Latency buffer path.
                'best_plan_path': self.best_plan_path,  # Best plan path.
                'scorer_path': self.scorer_path,  # Scorer model path.
                'agent_checkpoint': self.agent_checkpoint,  # Agent checkpoint.
                'train_info_path': self.train_info_path,  # Train info path.
                'PlannerworkersNum': self.PlannerworkersNum,  # Planner worker count.
                'LearnerworkersNum': self.LearnerworkersNum,  # Learner scheduling worker count; zero disables scheduling resources.
                'ScorerworkersNum': self.ScorerworkersNum,  # Scorer scheduling worker count; two scorer actors are created.
                'use_cpu_rate': self.use_cpu_rate,  # Allocate workers by a fraction of available CPU threads.
                'CPU_thread_utilization_rate': self.CPU_thread_utilization_rate,  # Cpu thread utilization rate.
                'total_time': [(end_time-start_time).total_seconds()],  # Total time.
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
            self.train_info['scorer_incremental_epochs'] = self.scorer_incremental_epochs
            self.train_info['scorer_reinitialize_each_train'] = self.scorer_reinitialize_each_train
            self.train_info['planner_batch_size'] = self.planner_batch_size
            self.train_info['planner_num_sgd_iter'] = self.planner_num_sgd_iter
            self.train_info['maxsteps'] = self.maxsteps
            self.train_info['action_mode'] = self.action_mode
            self.train_info['set_hint_gucs'] = self.set_hint_gucs
            self.train_info['set_size'] = self.set_size
            self.train_info['hint_sets_switch'] = self.hint_sets_switch
            self.train_info['stop_warmup_iters'] = self.stop_warmup_iters
            self.train_info['stop_min_steps'] = self.stop_min_steps
            self.train_info['test_stop_min_steps'] = self.test_stop_min_steps
            self.train_info['selection_strategy_diagnostics'] = self.selection_strategy_diagnostics
            self.train_info['reward_use_total_time'] = self.reward_use_total_time
            self.train_info['balanced_db_sampling'] = self.balanced_db_sampling
            self.train_info['checkpoint_metric'] = self.checkpoint_metric
            self.train_info['validate_test_only_on_train_improvement'] = self.validate_test_only_on_train_improvement
            self.train_info['full_validation_after_iter'] = self.full_validation_after_iter
            self.train_info['tmp_checkpoint_freq'] = self.tmp_checkpoint_freq
            self.train_info['active_sample_once_per_planner_iter'] = self.active_sample_once_per_planner_iter
            self.train_info['cost_plan_timeoutcoeff'] = self.cost_plan_timeoutcoeff
            self.train_info['cost_plan_timeout_min_ms'] = self.cost_plan_timeout_min_ms
            self.train_info['scorer_data_fusion_mode'] = self.scorer_data_fusion_mode
            self.train_info['scorer_cross_attn_layers'] = self.scorer_cross_attn_layers
            self.train_info['scorer_cross_attn_heads'] = self.scorer_cross_attn_heads
            self.train_info['planner_data_fusion_mode'] = self.planner_data_fusion_mode
            self.train_info['planner_cross_attn_actor_only'] = self.planner_cross_attn_actor_only
            self.train_info['cross_attn_use_local_plan_q'] = self.cross_attn_use_local_plan_q
            self.train_info['ablation_disable_global_context'] = self.ablation_disable_global_context
            self.train_info['cross_attn_q_feature_mask'] = self.cross_attn_q_feature_mask
            self.train_info['cross_attn_kv_feature_mask'] = self.cross_attn_kv_feature_mask
            self.train_info['cross_attn_q_use_query_feat'] = self.cross_attn_q_use_query_feat
            self.train_info['cross_attn_q_use_plan_nodes'] = self.cross_attn_q_use_plan_nodes
            self.train_info['use_db_features'] = self.use_db_features
            self.train_info['use_db_bitmap_features'] = self.use_db_bitmap_features
            self.train_info['enable_db_meta'] = self.enable_db_meta
            self.train_info['scorer_use_risk_head'] = self.scorer_use_risk_head
            self.train_info['scorer_label_clip'] = self.scorer_label_clip
            self.train_info['scorer_risk_loss_weight'] = self.scorer_risk_loss_weight
            self.train_info['scorer_risk_inference_weight'] = self.scorer_risk_inference_weight
            self.train_info['planner_surrogate_risk_weight'] = self.planner_surrogate_risk_weight
            self.train_info['scorer_rollout_update_mode'] = self.scorer_rollout_update_mode
            self.train_info['planner_reward_diagnostics'] = self.planner_reward_diagnostics
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
            self.train_info['data_drift_workload_source'] = self.data_drift_workload_source
            self.train_info['distill_best_plan_topk'] = self.distill_best_plan_topk
            self.train_info['distill_best_plan_weight'] = self.distill_best_plan_weight
            self.train_info['dynamic_hard_weight'] = self.dynamic_hard_weight
            self.train_info['dynamic_hard_weight_cap'] = self.dynamic_hard_weight_cap
            self.train_info['use_best_plan_pool'] = self.use_best_plan_pool
            self.train_info['ablation_name'] = self.ablation_name
            self.train_info['ablation_disable_cardinality_aux_loss'] = self.ablation_disable_cardinality_aux_loss
            self.train_info['ablation_disable_risk_head'] = self.ablation_disable_risk_head
            self.train_info['ablation_disable_safe_select'] = self.ablation_disable_safe_select
            self.train_info['ablation_disable_db_state_features'] = self.ablation_disable_db_state_features
            self.train_info['ablation_disable_topk_best_weighting'] = self.ablation_disable_topk_best_weighting
            self.train_info['ablation_disable_negative_risk_weighting'] = self.ablation_disable_negative_risk_weighting
            self.train_info['ablation_disable_hard_sample_weighting'] = self.ablation_disable_hard_sample_weighting
            self.train_info['ablation_disable_db_balancing'] = self.ablation_disable_db_balancing
            self.train_info['use_cardinality_aux_loss'] = self.use_cardinality_aux_loss
            self.train_info['encoding_path'] = self.encoding_path
            self.train_info['auto_config'] = self.auto_config
            self.train_info['latency_buffer_path'] = self.latency_buffer_path
            self.train_info['best_plan_path'] = self.best_plan_path
            self.train_info['scorer_path'] = self.scorer_path
            self.train_info['agent_checkpoint'] = self.agent_checkpoint
            self.train_info['train_info_path'] = self.train_info_path
            self.train_info['validation_time_seconds'] = self._normalize_validation_time(
                self.validation_time_seconds
            )
            if first_save is True:
                self.train_info['start_time'].append(start_time.strftime('%Y/%m/%d %H.%M.%S'))
                self.train_info['end_time'].append(end_time.strftime('%Y/%m/%d %H.%M.%S'))
                self.train_info['total_time'].append((end_time-start_time).total_seconds())
            else:
                self.train_info['start_time'][-1] = start_time.strftime('%Y/%m/%d %H.%M.%S')
                self.train_info['end_time'][-1] = end_time.strftime('%Y/%m/%d %H.%M.%S')
                self.train_info['total_time'][-1] = (end_time-start_time).total_seconds()
            if best_checkpoints is not None:
                self.train_info['best_checkpoints'] = best_checkpoints
            checkpoints = self.train_info.get('best_checkpoints', {})
            if ('train_wrl' in checkpoints or 'train_wrl_noPlanningTime' in checkpoints) and 'train_gmrl' in checkpoints:
                for key in (
                    'model_saved_iter', 'model_wrl_train', 'model_wrl_test',
                    'model_wrl_withPlanning_train', 'model_wrl_withPlanning_test',
                    'model_gmrl_train', 'model_gmrl_test', 'checkpoint_best_wrl',
                    'checkpoint_best_gmrl', 'model_metrics', 'metric_naming_version',
                ):
                    self.train_info.pop(key, None)
        self.train_info['adaptive_gate_enable'] = self.adaptive_gate_enable
        self.train_info['adaptive_gate_mode'] = self.adaptive_gate_mode
        self.train_info['nearest_tag_model_path'] = self.nearest_tag_model_path
        self.train_info['adaptive_gate_min_records'] = self.adaptive_gate_min_records
        self.train_info['latency_repeat_phases'] = list(self.latency_repeat_phases)
        self.train_info['latency_warmup_runs'] = self.latency_warmup_runs
        self.train_info['latency_measure_runs'] = self.latency_measure_runs
        self.train_info['latency_aggregate'] = self.latency_aggregate
        if not os.path.exists(os.path.dirname(self.train_info_path)):
            os.makedirs(os.path.dirname(self.train_info_path))
        with open(self.train_info_path, 'w') as json_file:
            json.dump(self.train_info, json_file, ensure_ascii=False, indent=4)  # Save as JSON.
        return self.train_info['start_time'][0],end_time

    @staticmethod
    def _checkpoint_metrics_from_train_info(train_info, checkpoint_metric):
        best_checkpoints = train_info.get('best_checkpoints', {})
        wrl_record = best_checkpoints.get('train_wrl') or best_checkpoints.get('train_wrl_noPlanningTime')
        gmrl_record = best_checkpoints.get('train_gmrl')
        if wrl_record or gmrl_record:
            return (
                float(wrl_record['value']) if wrl_record else float(train_info.get('checkpoint_best_wrl', 1.0)),
                float(gmrl_record['value']) if gmrl_record else float(train_info.get('checkpoint_best_gmrl', 1.0)),
            )
        checkpoint_wrl = train_info.get('checkpoint_best_wrl')
        checkpoint_gmrl = train_info.get('checkpoint_best_gmrl')
        if checkpoint_wrl is not None and checkpoint_gmrl is not None:
            return checkpoint_wrl, checkpoint_gmrl
        if 'model_wrl_train' not in train_info:
            return float(checkpoint_wrl if checkpoint_wrl is not None else 1.0), float(
                checkpoint_gmrl if checkpoint_gmrl is not None else 1.0
            )

        checkpoint_metric = {
            'train_wrl_noPlanningTime': 'train_wrl',
            'train_wrl_total': 'train_wrl_withPlanning',
            'test_wrl_total': 'test_wrl_withPlanning',
        }.get(checkpoint_metric, checkpoint_metric)
        if checkpoint_metric == 'train_wrl':
            if train_info.get('metric_naming_version', 1) >= 2:
                checkpoint_wrl = train_info['model_wrl_train']
            else:
                checkpoint_wrl = (
                    train_info.get('model_metrics', {})
                    .get('Train', {})
                    .get('WRL_noPlanningTime', train_info['model_wrl_train'])
                )
            return checkpoint_wrl, train_info['model_gmrl_train']
        if checkpoint_metric == 'train_wrl_withPlanning':
            checkpoint_wrl = train_info.get('model_wrl_withPlanning_train', train_info['model_wrl_train'])
            return checkpoint_wrl, train_info['model_gmrl_train']
        if checkpoint_metric == 'test_wrl_withPlanning':
            checkpoint_wrl = train_info.get('model_wrl_withPlanning_test', train_info['model_wrl_test'])
            return checkpoint_wrl, train_info['model_gmrl_test']
        return (
            train_info.get('model_wrl_test', train_info['model_wrl_train']),
            train_info.get('model_gmrl_test', train_info['model_gmrl_train']),
        )

    def LoadTrainInfo(self,train_info_path):
        """Restore training metadata and return the iteration and best Train WRL/GMRL."""
        train_iter = 0
        if os.path.exists(train_info_path):
            with open(train_info_path, 'r') as f:
                self.train_info = json.load(f)
            self._require_current_train_info(self.train_info, train_info_path)
            self.latency_repeat_phases = list(self.train_info.get(
                'latency_repeat_phases', self.latency_repeat_phases
            ))
            self.latency_warmup_runs = self.train_info.get('latency_warmup_runs', self.latency_warmup_runs)
            self.latency_measure_runs = self.train_info.get('latency_measure_runs', self.latency_measure_runs)
            self.latency_aggregate = self.train_info.get('latency_aggregate', self.latency_aggregate)
            if 'adaptive_gate_mode' in self.train_info:
                self.adaptive_gate_mode = self.train_info['adaptive_gate_mode']
            elif 'adaptive_gate_enable' in self.train_info:
                self.adaptive_gate_mode = 'baseline' if self.train_info['adaptive_gate_enable'] else 'off'
            self.nearest_tag_model_path = self.train_info.get('nearest_tag_model_path', self.nearest_tag_model_path)
            self.adaptive_gate_min_records = self.train_info.get(
                'adaptive_gate_min_records', self.adaptive_gate_min_records
            )
            self.encoding_frozen = True
            self.time = self.train_info['time']
            self.expname = self.train_info.get('expname', self.expname)
            self.train_mode = self.train_info.get('train_mode', self.train_mode)
            self.model_root = self.train_info.get('model_root', './model/{}_{}/'.format(self.DBMS, self.mode))
            self.model_dir = self.train_info.get('model_dir', os.path.dirname(train_info_path))
            self.model_run_id = self.train_info.get('model_run_id', os.path.basename(self.model_dir))
            self.scorer_epochs = self.train_info['scorer_epochs']
            self.scorer_incremental_epochs = self.train_info.get(
                'scorer_incremental_epochs', self.scorer_incremental_epochs
            )
            self.scorer_reinitialize_each_train = self.train_info.get(
                'scorer_reinitialize_each_train', self.scorer_reinitialize_each_train
            )
            self.train_eval = self.train_info.get('train_eval', self.train_eval)
            self.train_eval_schedule = self.train_info.get('train_eval_schedule', self.train_eval_schedule)
            self.hybrid_sample_threshold = self.train_info.get('hybrid_sample_threshold', self.hybrid_sample_threshold)
            self.planner_batch_size = self.train_info['planner_batch_size']
            self.planner_num_sgd_iter = self.train_info.get(
                'planner_num_sgd_iter', self.planner_num_sgd_iter
            )
            self.maxsteps = self.train_info['maxsteps']
            loaded_action_mode = self.train_info.get('action_mode', self.action_mode)
            self.action_mode = loaded_action_mode
            self.set_hint_gucs = self.train_info.get('set_hint_gucs', self.set_hint_gucs)
            self.set_size = self.train_info.get('set_size', self.set_size)
            self.hint_sets_switch = self.train_info.get('hint_sets_switch', self.hint_sets_switch)
            self.num_agents = self.train_info['num_agents']
            self.num_policies = self.train_info['num_policies']
            self.stop_warmup_iters = self.train_info.get('stop_warmup_iters', self.stop_warmup_iters)
            self.stop_min_steps = self.train_info.get('stop_min_steps', self.stop_min_steps)
            self.test_stop_min_steps = max(
                2, int(self.train_info.get('test_stop_min_steps', self.test_stop_min_steps))
            )
            self.selection_strategy_diagnostics = self.train_info.get(
                'selection_strategy_diagnostics', self.selection_strategy_diagnostics
            )
            self.reward_use_total_time = self.train_info.get('reward_use_total_time', self.reward_use_total_time)
            self.balanced_db_sampling = self.train_info.get('balanced_db_sampling', self.balanced_db_sampling)
            self.checkpoint_metric = {
                'train_wrl_noPlanningTime': 'train_wrl',
                'train_wrl_total': 'train_wrl_withPlanning',
                'test_wrl_total': 'test_wrl_withPlanning',
            }.get(self.train_info.get('checkpoint_metric', self.checkpoint_metric), self.checkpoint_metric)
            self.validate_test_only_on_train_improvement = self.train_info.get(
                'validate_test_only_on_train_improvement',
                self.validate_test_only_on_train_improvement,
            )
            self.full_validation_after_iter = self.train_info.get(
                'full_validation_after_iter', self.full_validation_after_iter
            )
            self.tmp_checkpoint_freq = self.train_info.get('tmp_checkpoint_freq', self.tmp_checkpoint_freq)
            self.active_sample_once_per_planner_iter = self.train_info.get(
                'active_sample_once_per_planner_iter', self.active_sample_once_per_planner_iter
            )
            self.cost_plan_timeoutcoeff = self.train_info.get(
                'cost_plan_timeoutcoeff', self.cost_plan_timeoutcoeff
            )
            self.cost_plan_timeout_min_ms = self.train_info.get(
                'cost_plan_timeout_min_ms', self.cost_plan_timeout_min_ms
            )
            self.scorer_data_fusion_mode = self.train_info.get('scorer_data_fusion_mode', self.scorer_data_fusion_mode)
            self.scorer_cross_attn_layers = self.train_info.get('scorer_cross_attn_layers', self.scorer_cross_attn_layers)
            self.scorer_cross_attn_heads = self.train_info.get('scorer_cross_attn_heads', self.scorer_cross_attn_heads)
            self.planner_data_fusion_mode = self.train_info.get('planner_data_fusion_mode', self.planner_data_fusion_mode)
            self.planner_cross_attn_actor_only = self.train_info.get('planner_cross_attn_actor_only', self.planner_cross_attn_actor_only)
            self.cross_attn_use_local_plan_q = self.train_info.get('cross_attn_use_local_plan_q', self.cross_attn_use_local_plan_q)
            self.ablation_disable_global_context = self.train_info.get('ablation_disable_global_context', False)
            self.cross_attn_q_feature_mask = self.train_info.get('cross_attn_q_feature_mask', self.cross_attn_q_feature_mask)
            self.cross_attn_kv_feature_mask = self.train_info.get('cross_attn_kv_feature_mask', self.cross_attn_kv_feature_mask)
            self.cross_attn_q_use_query_feat = self.train_info.get('cross_attn_q_use_query_feat', self.cross_attn_q_use_query_feat)
            self.cross_attn_q_use_plan_nodes = self.train_info.get('cross_attn_q_use_plan_nodes', self.cross_attn_q_use_plan_nodes)
            self.use_db_features = self.train_info.get('use_db_features', self.use_db_features)
            self.use_db_bitmap_features = self.train_info.get('use_db_bitmap_features', self.use_db_bitmap_features)
            self.enable_db_meta = self.train_info.get('enable_db_meta', self.enable_db_meta)
            self.scorer_use_risk_head = self.train_info.get('scorer_use_risk_head', self.scorer_use_risk_head)
            self.scorer_label_clip = self.train_info.get('scorer_label_clip', self.scorer_label_clip)
            self.scorer_risk_loss_weight = self.train_info.get('scorer_risk_loss_weight', self.scorer_risk_loss_weight)
            self.scorer_risk_inference_weight = self.train_info.get('scorer_risk_inference_weight', self.scorer_risk_inference_weight)
            self.planner_surrogate_risk_weight = self.train_info.get(
                'planner_surrogate_risk_weight', self.planner_surrogate_risk_weight
            )
            self.scorer_rollout_update_mode = self.train_info.get(
                'scorer_rollout_update_mode', self.scorer_rollout_update_mode
            )
            self.planner_reward_diagnostics = self.train_info.get(
                'planner_reward_diagnostics', self.planner_reward_diagnostics
            )
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
            self.data_drift_workload_source = self.train_info.get('data_drift_workload_source', self.data_drift_workload_source)
            self.distill_best_plan_topk = self.train_info.get('distill_best_plan_topk', self.distill_best_plan_topk)
            self.distill_best_plan_weight = self.train_info.get('distill_best_plan_weight', self.distill_best_plan_weight)
            self.dynamic_hard_weight = self.train_info.get('dynamic_hard_weight', self.dynamic_hard_weight)
            self.dynamic_hard_weight_cap = self.train_info.get('dynamic_hard_weight_cap', self.dynamic_hard_weight_cap)
            self.use_best_plan_pool = self.train_info.get('use_best_plan_pool', self.use_best_plan_pool)
            self.ablation_name = self.train_info.get('ablation_name', self.ablation_name)
            self.ablation_disable_cardinality_aux_loss = self.train_info.get('ablation_disable_cardinality_aux_loss', self.ablation_disable_cardinality_aux_loss)
            self.ablation_disable_risk_head = self.train_info.get('ablation_disable_risk_head', self.ablation_disable_risk_head)
            self.ablation_disable_safe_select = self.train_info.get('ablation_disable_safe_select', self.ablation_disable_safe_select)
            self.ablation_disable_db_state_features = self.train_info.get('ablation_disable_db_state_features', self.ablation_disable_db_state_features)
            self.ablation_disable_topk_best_weighting = self.train_info.get('ablation_disable_topk_best_weighting', self.ablation_disable_topk_best_weighting)
            self.ablation_disable_negative_risk_weighting = self.train_info.get('ablation_disable_negative_risk_weighting', self.ablation_disable_negative_risk_weighting)
            self.ablation_disable_hard_sample_weighting = self.train_info.get('ablation_disable_hard_sample_weighting', self.ablation_disable_hard_sample_weighting)
            self.ablation_disable_db_balancing = self.train_info.get('ablation_disable_db_balancing', self.ablation_disable_db_balancing)
            self.use_cardinality_aux_loss = self.train_info.get('use_cardinality_aux_loss', self.use_cardinality_aux_loss)
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
            self.best_checkpoints = self.train_info.get('best_checkpoints', {})
            if 'train_wrl' not in self.best_checkpoints and 'train_wrl_noPlanningTime' in self.best_checkpoints:
                self.best_checkpoints['train_wrl'] = self.best_checkpoints.pop('train_wrl_noPlanningTime')
            self.train_info_path = self.train_info.get('train_info_path', train_info_path)
            self.validation_time_seconds = self._normalize_validation_time(
                self.train_info.get('validation_time_seconds')
            )
            train_iter = self.train_info['train_iter']
            model_wrl_train, model_gmrl_train = self._checkpoint_metrics_from_train_info(
                self.train_info,
                self.checkpoint_metric,
            )
            if self.train_info['PlannerworkersNum'] + self.LearnerworkersNum + self.ScorerworkersNum*2 > self.cpu_resources:
                print(f"Insufficient CPU resources to resume training; update the configuration:")
                if self.train_info['use_cpu_rate'] is True:
                    if self.train_info['CPU_thread_utilization_rate'] != self.CPU_thread_utilization_rate:
                        print(f"Match the CPU-thread utilization fraction to the original run:")
                        print(f"CPU_thread_utilization_rate: { self.CPU_thread_utilization_rate } -> original: { self.train_info['CPU_thread_utilization_rate'] }")
                else:
                    if self.train_info['PlannerworkersNum'] != self.PlannerworkersNum or self.train_info['LearnerworkersNum'] != self.LearnerworkersNum or self.train_info['ScorerworkersNum'] != self.ScorerworkersNum:
                        print(f"Match worker counts to the original run:")
                        print(f"PlannerworkersNum: { self.PlannerworkersNum } -> original: { self.train_info['PlannerworkersNum'] }")
                        print(f"LearnerworkersNum: { self.LearnerworkersNum } -> original: { self.train_info['LearnerworkersNum'] }")
                        print(f"ScorerworkersNum: { self.ScorerworkersNum } -> original: { self.train_info['ScorerworkersNum'] }")
                raise ValueError(f"ERROR: 继续训练的CPU资源不足!")
            else:
                print("Compute resource allocation for resumed training:")
                use_resources = self.train_info['PlannerworkersNum'] + self.LearnerworkersNum + self.ScorerworkersNum*2
                print(f"CPU - allocated: {use_resources:.1f} (Planner: {self.train_info['PlannerworkersNum']:.1f} + Learner: {self.LearnerworkersNum:.1f} + Scorer: {self.ScorerworkersNum*2:.1f}) / "
                f"total: {self.cpu_resources:.1f}")
        else:
            raise FileNotFoundError(f"训练信息文件【{train_info_path}】不存在")  # Report a missing training-metadata file.
        return train_iter, model_wrl_train, model_gmrl_train
    
    def GetTotalTime(self):
        """Return formatted total and current-session training durations."""
        # Sum recorded training durations in seconds.
        total_seconds = sum(self.train_info['total_time'])
        # Format seconds as hours, minutes and seconds.
        hours = int(total_seconds // 3600)  # Hours.
        minutes = int((total_seconds % 3600) // 60)  # Minutes.
        seconds = int(total_seconds % 60)  # Seconds.

        last_time = self.train_info['total_time'][-1]
        last_hours = int(last_time // 3600)  # Hours.
        last_minutes = int((last_time % 3600) // 60)  # Minutes.
        last_seconds = int(last_time % 60)  # Seconds.
        return [f"{hours}:{minutes:02d}:{seconds:02d}",f"{last_hours}:{last_minutes:02d}:{last_seconds:02d}"]

    
