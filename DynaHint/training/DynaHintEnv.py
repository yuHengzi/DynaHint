"""Search environments for training and evaluation."""
import gymnasium as gym
import numpy as np
from copy import deepcopy
from shared.util import min_steps,min_swap_steps,get_label
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os,random
from training.scorer import ScorerTrainer
from shared.manager import QueryManager
from collection.planhelper import PlanHelper
import ray


def should_reload_scorer(config, version_changed, current_planner_iter, loaded_planner_iter):
    if not version_changed:
        return False
    mode = getattr(config, 'scorer_rollout_update_mode', 'immediate')
    if mode == 'frozen':
        return False
    if mode == 'planner_boundary':
        return current_planner_iter != loaded_planner_iter
    if mode == 'immediate':
        return True
    raise ValueError(
        "scorer_rollout_update_mode must be 'immediate', 'planner_boundary', or 'frozen'"
    )


def build_reward_diagnostic(source, predicted_score=None, true_score=None):
    diagnostic = {'reward_source': source}
    if predicted_score is not None:
        diagnostic['scorer_predicted_score'] = float(predicted_score)
    if true_score is not None:
        true_score = float(true_score)
        diagnostic['scorer_true_score'] = true_score
        if predicted_score is not None:
            predicted_score = float(predicted_score)
            diagnostic['scorer_sign_correct'] = float(
                (predicted_score >= 0.0) == (true_score >= 0.0)
            )
            diagnostic['scorer_abs_error'] = abs(predicted_score - true_score)
    return diagnostic


class DynaHintEnvBase(gym.Env):
    def __init__(self,env_config):
        # Initialize environment configuration and table capacity.
        self.tablenum = env_config['tableNum']
        self.config   = env_config['genConfig']
        if self.config.AutoGetParam and os.path.exists(self.config.auto_config):
            data = self.GetParam()
            # self.tablenum = data["tablenum"]-1
            self.config.maxjoins = data["maxjoins"]
            self.config.max_logical_edges = data["max_logical_edges"]
            self.config.num_node_feature = data["num_node_feature"]
            self.config.heightsize = data["heightsize"]
            self.config.maxnode = data["maxnode"]
            self.config.db_hist_dim = data["db_hist_dim"]
            self.config.db_bitmap_dim = data["db_bitmap_dim"]
            self.config.query_feat_dim = data["query_feat_dim"]
        # Compute action-space size.
        self.action_mode = self.config.action_mode
        if self.action_mode not in ('set_then_join_order', 'join_order_and_operator'):
            raise ValueError(f'Unsupported action_mode: {self.action_mode}')
        self.swap_action_size = int((self.tablenum * (self.tablenum - 1)) / 2)  # Number of join-order swap actions.
        self.set_action_size = int(getattr(self.config, 'set_size', 64)) if self.action_mode == 'set_then_join_order' else 0
        self.operator_action_size = 3 * (self.tablenum - 1) if self.action_mode == 'join_order_and_operator' else 0
        self.set_action_start = self.swap_action_size
        self.set_action_end = self.set_action_start + self.set_action_size
        self.action_space_size = self.swap_action_size + self.set_action_size + self.operator_action_size + 1
        self.stop_action_idx = self.action_space_size - 1
        self.valid_hint_sets = tuple(
            hint_set for hint_set in range(self.set_action_size)
            if self.is_valid_hint_set(hint_set)
        )
        if self.action_mode == 'set_then_join_order' and not self.valid_hint_sets:
            raise ValueError('SET configuration has no valid actions')
        # Define observation spaces.
        self.observation_space = spaces.Dict({
            'x': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                            shape=(self.config.maxnode, self.config.num_node_feature)),
            'attn_bias': spaces.Box(0, 1, dtype=np.float32,
                                    shape=(self.config.maxnode + 1, self.config.maxnode + 1)),
            'heights': spaces.Box(0, self.config.heightsize - 1, dtype=np.int64,
                                  shape=(self.config.maxnode,)),
            'action_mask': spaces.Box(0, 1, dtype=np.int32,
                                      shape=(self.action_space_size,)),
            'steps': spaces.Box(0, 1, dtype=np.float32, shape=(1,)),
            'set_hint_bits': spaces.Box(0, 1, dtype=np.float32,
                                        shape=(len(self.config.set_hint_gucs),)),
            **({
                'physical_join_node_indices': spaces.Box(
                    0, self.config.maxnode, dtype=np.int64,
                    shape=(max(0, self.tablenum - 1),),
                ),
            } if self.action_mode == 'join_order_and_operator' else {}),
            'card_label': spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=(1,)),
            # ===== alias-level logical join graph =====
            # 0 is padding; encoded table and relation IDs are one-based.
            'logical_rel_table_ids': spaces.Box(0, self.tablenum, dtype=np.int64,
                                                 shape=(self.tablenum,)),
            # 0 is padding; real database table positions are one-based.
            'logical_rel_db_positions': spaces.Box(0, self.tablenum, dtype=np.int64,
                                                   shape=(self.tablenum,)),
            'logical_rel_mask': spaces.Box(0, 1, dtype=np.float32, shape=(self.tablenum,)),
            'logical_edge_index': spaces.Box(0, self.tablenum, dtype=np.int64,
                                             shape=(self.config.max_logical_edges, 2)),
            'logical_edge_columns': spaces.Box(0, self.config.columns - 1, dtype=np.int64,
                                               shape=(self.config.max_logical_edges, 2)),
            'logical_edge_ops': spaces.Box(0, self.config.opsnum - 1, dtype=np.int64,
                                           shape=(self.config.max_logical_edges,)),
            'logical_edge_mask': spaces.Box(0, 1, dtype=np.float32,
                                            shape=(self.config.max_logical_edges,)),
            'join_order_rel_ids': spaces.Box(0, self.tablenum, dtype=np.int64,
                                             shape=(self.tablenum,)),
            # Histogram and bitmap features for data drift.
            # PlanHelper.get_feature() injects these fields automatically.
            **({
                'db_hist': spaces.Box(0, 1, dtype=np.float32,
                                      shape=(self.config.db_hist_dim,)),
                'db_bitmap': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                                        shape=(self.config.db_bitmap_dim,)),
                'query_feat': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                                         shape=(self.config.query_feat_dim,)),
            } if getattr(self.config, 'use_db_features', False) else {}),

            # Metadata features for data drift and query drift.
            # PlanHelper.get_feature() injects these fields automatically.
            **({
                'db_table_stats': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                                             shape=(self.tablenum, self.config.db_table_feat_dim)),
                'db_global_stats': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                                              shape=(self.config.db_global_feat_dim,)),
                'query_table_mask': spaces.Box(0, 1, dtype=np.float32,
                                               shape=(self.tablenum,)),
                'query_col_stats': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                                              shape=(self.config.db_col_feat_dim,)),
            } if getattr(self.config, 'enable_db_meta', False) else {})
            })
        # Define the action space.
        self.action_space = spaces.Discrete(self.action_space_size)
        self.action_inteval = [0]
        # Compute action offsets.
        for i in range(1,self.tablenum):
            self.action_inteval.append(self.action_inteval[-1] + self.tablenum - i)

    def is_set_action(self, action):
        return self.set_action_start <= int(action) < self.set_action_end

    def operator_action_index(self, join_idx, operator_idx):
        return self.stop_action_idx - (3 * join_idx + operator_idx + 1)

    def operator_group_bounds(self, join_idx):
        end = self.stop_action_idx - 3 * join_idx
        return end - 3, end

    def hint_set_from_action(self, action):
        if not self.is_set_action(action):
            raise ValueError(f'Action {action} is not a SET action')
        hint_set = int(action) - self.set_action_start
        if not self.is_valid_hint_set(hint_set):
            raise ValueError(f'SET hint {hint_set} disables every join or scan method')
        return hint_set

    def hint_set_bits(self, hint_set):
        width = len(self.config.set_hint_gucs)
        return np.asarray(
            [int(bit) for bit in bin(int(hint_set))[2:].zfill(width)],
            dtype=np.float32,
        )

    def is_valid_hint_set(self, hint_set):
        allowed_hint_sets = getattr(self.config, 'hint_sets_switch', None)
        if allowed_hint_sets is not None and int(hint_set) not in allowed_hint_sets:
            return False
        enabled = {
            guc: bool(bit)
            for guc, bit in zip(self.config.set_hint_gucs, self.hint_set_bits(hint_set))
        }
        has_join = any(enabled.get(guc, False) for guc in (
            'enable_hashjoin', 'enable_mergejoin', 'enable_nestloop',
        ))
        has_scan = any(enabled.get(guc, False) for guc in (
            'enable_indexscan', 'enable_seqscan', 'enable_indexonlyscan',
        ))
        return has_join and has_scan

    def enable_valid_set_actions(self, action_mask):
        indices = [self.set_action_start + hint_set for hint_set in self.valid_hint_sets]
        action_mask[indices] = 1
    
    def GetParam(self):
        import json
        with open(self.config.auto_config, 'r', encoding='utf-8') as json_file:  # Open the file with UTF-8 encoding.
            data = json.load(json_file)  # Load JSON data.
        bins_per_col = data.get("db_hist_bins_per_col", getattr(self.config, "db_hist_bins_per_col", 51))
        data["node_hist_dim"] = data.get("node_hist_dim", 3 * (bins_per_col - 1))
        if "maxjoins" in data and "filtmaxnum" in data:
            data["num_node_feature"] = 7 + data["maxjoins"] + 5 * data["filtmaxnum"] + data["node_hist_dim"]
        return data
    
    def reset(self, seed=None, options=None):
        # Reset environment state.
        return None,None
    
    def step(self,action):
        # Execute an action and return the new state.
        return None,None,None,None,None

class DynaHintEnvTrainBase(gym.Wrapper):
    # Training-environment base wrapper.
    def __init__(self,env_config) -> None:
        unwrapped_env = DynaHintEnvBase(env_config)
        super().__init__(unwrapped_env) 
        self.planhelper  = env_config['planhelper']
        self.querymanger = env_config['querymanger']
        self.scorer_trainer = env_config['scorer_trainer']
        self.config      = env_config['genConfig']
        self.bpm         = env_config['bestplanmanager']
        self.querymanger.creat_trainSet()
        self.isCollectSamples = self.config.update_scorer
        splitpoint = [1.00] + self.config.splitpoint
        self.bouns_weight = [0.00]
        for i in range(len(splitpoint) - 1, 0, -1):
            self.bouns_weight.append((splitpoint[i] + splitpoint[i - 1]) / 2)
        self.episode_count = 0  # Use episode counts to implement STOP warmup.
    # Find observation features outside their declared spaces.
    def print_outside_given_space(self, space, obs):
        for k, sp in space.spaces.items():
            if k not in obs:
                print("[MISSING]", k)
                continue
            v = obs[k]
            if not sp.contains(v):
                arr = np.asarray(v)
                print(f"[OUTSIDE] {k}: shape={arr.shape}, dtype={arr.dtype}, "
                      f"min={np.nanmin(arr)}, max={np.nanmax(arr)}, "
                      f"has_nan={np.isnan(arr).any()}, has_inf={np.isinf(arr).any()}")
    def print_feature_dims(self, base_train_feature):
        """Print the shape of each base_train_feature entry."""
        for k, v in base_train_feature.items():
            arr = np.asarray(v)
            print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")

    def _enable_stop_action(self, action_mask):
        action_mask[self.unwrapped.stop_action_idx] = 1 if self._allow_stop() else 0  # Disable STOP during warmup and early steps to avoid immediate termination.

    def _allow_stop(self):
        return self.episode_count > self.config.stop_warmup_iters and self.stepnum >= self.config.stop_min_steps

    def _enable_swap_actions(self):
        for i in range(self.count_table - 1):
            self.action_mask[self.action_inteval[i]:self.action_inteval[i] + self.count_table - i - 1] = 1  # Enable join-order swaps applicable to this query.

    def _enable_operator_group(self, join_idx):
        if 0 <= join_idx < self.count_table - 1:
            start, end = self.unwrapped.operator_group_bounds(join_idx)
            self.action_mask[start:end] = 1

    def _rebuild_full_action_mask(self):
        self.action_mask.fill(0)
        self._enable_swap_actions()
        for join_idx in range(self.count_table - 1):
            self._enable_operator_group(join_idx)
        for join_idx, operator in enumerate(self.hintdict['join operator']):
            self.action_mask[self.unwrapped.operator_action_index(
                join_idx, self.config.OperatorDict[operator]
            )] = 0
        self._enable_stop_action(self.action_mask)

    def _rebuild_swap_followup_mask(self, left, right):
        self.action_mask.fill(0)
        if left <= 1 or right == 1:
            self._enable_operator_group(0)
        else:
            self._enable_operator_group(left - 1)
            self._enable_operator_group(right - 1)
        self._enable_stop_action(self.action_mask)

    def _rebuild_initial_action_mask(self):
        if self.unwrapped.action_mode == 'join_order_and_operator':
            self._rebuild_full_action_mask()
        else:
            self.action_mask.fill(0)
            self.unwrapped.enable_valid_set_actions(self.action_mask)

    def _rebuild_join_order_action_mask(self):
        self.action_mask.fill(0)
        self._enable_swap_actions()
        self._enable_stop_action(self.action_mask)

    @staticmethod
    def _same_candidate_hint(left, right):
        return (
            left.get('hint set') == right.get('hint set')
            and left['join order'] == right['join order']
            and left['join operator'] == right['join operator']
        )

    def _clip_reward(self, value):
        # Clip rewards to keep occasional large ratios from destabilizing PPO.
        return float(np.clip(value, -self.config.maxbounty, self.config.maxbounty))

    def _reward_objective_value(self, execution_latency, planning_time_cum_ms):
        # Use total-time objectives only when reward_use_total_time is enabled.
        execution_latency = float(execution_latency)
        if getattr(self.config, 'reward_use_total_time', False):
            return execution_latency + float(planning_time_cum_ms)
        return execution_latency

    def _score_to_objective_ms(self, score_value):
        # Convert pointwise scores into baseline-relative surrogate objectives when latency is unavailable.
        score_clip = float(getattr(self.config, 'scorer_label_clip', 8.0))
        clipped_score = float(np.clip(score_value, -score_clip, score_clip))
        return float(self.baseline_reward_objective / math.exp(clipped_score))

    def _scorer_feature(self, plan_feature):
        # Use the original SQL baseline consistently for scorer training and online predictions.
        return {'candidate': plan_feature, 'baseline': self.baseplan}

    def _finalize_episode_bounty(self):
        # Terminal rewards measure net gain of the best objective relative to baseline.
        eps = 1e-6
        bounty = math.log((self.baseline_reward_objective + eps) / (self.best_reward_objective_so_far + eps))
        isvalidate = self.best_reward_objective_so_far + eps < self.baseline_reward_objective
        return self._clip_reward(bounty), isvalidate

    def reset(self, seed=None, options=None):
        # Reset environment state.
        self.episode_count += 1
        self.stepnum = 0
        self.scorer_update_times = options['scorer_times']
        self.candidatehint = []
        self.sql, base_train_feature,self.db,self.query_id,self.best_executed,self.RL_best = self.querymanger.get2train()
        self.baselatency  = self.planhelper.tryGetLatency('',self.db,self.query_id)
        self.optimlatency = self.baselatency
        feature_dict,self.hintdict, cost_plan_json = deepcopy(base_train_feature)
        # self.print_feature_dims(feature_dict)
        # self.debug_obs(self.observation_space, feature_dict)
        self.count_table = len(self.hintdict['join order'])
        if self.count_table != len(self.hintdict['join operator']) + 1:
            print(self.query_id + ' count_table:' + str(self.count_table)+', join operator:'+str(len(self.hintdict['join operator']))+' ,False !')
        assert self.count_table == len(self.hintdict['join operator']) + 1
        #=========== init action and init action mask==========
        self.action_mask = np.zeros(self.action_space_size, dtype=np.int32)
        self._rebuild_initial_action_mask()
        # process state
        feature_dict['steps'] = np.array(
            [self.stepnum * 1.0 / self.config.maxsteps], dtype=np.float32
        )
        feature_dict['set_hint_bits'] = np.ones(
            len(self.config.set_hint_gucs), dtype=np.float32
        )
        self.baseplan = deepcopy(feature_dict)
        self.currplan = deepcopy(feature_dict)
        
        feature_dict['action_mask'] = self.action_mask
        self.curr_obs = deepcopy(feature_dict)
        # self.best_executed[0]['action_mask'] = self.action_mask # 
        # self.RL_best[0]['action_mask'] = self.action_mask # 
        self.esbestplan = self.baseplan
        self.esbesthint = ''
        self.beststeps = 0
        self.isswapL = False
        if self.baselatency == None:
            self.baselatency = self.config.max_time_out
        self.baseline_planning_time_ms = float(cost_plan_json.get('Planning Time', 0.0)) * 1000.0
        self.plan_time_cum_ms = self.baseline_planning_time_ms
        self.baseline_reward_objective = self._reward_objective_value(self.baselatency, self.baseline_planning_time_ms)
        self.best_reward_objective_so_far = self.baseline_reward_objective
        self.best_candidate_plan = deepcopy(self.baseplan)

        self.candidatehint.append(deepcopy(self.hintdict))
        self.set_base_join_order = None
        self.is_done = False
        self.truncated = False
        return feature_dict,{}
    def step(self,action):
         # Execute an action and update state.
        self.stepnum += 1
        if action == self.unwrapped.stop_action_idx:
            self.is_done = True
            self.truncated = False
            bounty, _ = self._finalize_episode_bounty()
            stop_feature = deepcopy(self.curr_obs)
            stop_feature['steps'] = np.array(
                [self.stepnum * 1.0 / self.config.maxsteps], dtype=np.float32
            )
            stop_feature['action_mask'] = self.action_mask
            return stop_feature, bounty, self.is_done, self.truncated, build_reward_diagnostic('terminal')
        try:
            if self.unwrapped.action_mode == 'join_order_and_operator' and action >= self.unwrapped.swap_action_size:
                idx = self.unwrapped.stop_action_idx - 1 - action
                self.hintdict['join operator'][idx // 3] = self.config.Operatortype[idx % 3]
                self._rebuild_full_action_mask()
                exechint = self.planhelper.to_exechint(self.hintdict)
                self.currlatency = self.planhelper.tryGetLatency(exechint, self.db, self.query_id)
                feature_dict, _, _, cost_plan_json = self.planhelper.get_feature(
                    exechint, self.sql, False, self.db, query_id=self.query_id
                )
            elif self.unwrapped.is_set_action(action):
                if self.stepnum != 1 or self.set_base_join_order is not None:
                    raise ValueError(f'SET action is only valid at step 1: {action}')
                hint_set = self.unwrapped.hint_set_from_action(action)
                set_only_state = {'hint set': hint_set, 'force join order': False}
                exechint = self.planhelper.to_exechint(set_only_state)
                self.currlatency = self.planhelper.tryGetLatency(exechint, self.db, self.query_id)
                feature_dict, replanned_hintdict, _, cost_plan_json = self.planhelper.get_feature(
                    exechint, self.sql, True, self.db, query_id=self.query_id
                )
                replanned_hintdict['hint set'] = hint_set
                replanned_hintdict['force join order'] = True
                self.hintdict = replanned_hintdict
                self.set_base_join_order = deepcopy(self.hintdict['join order'])
                self._rebuild_join_order_action_mask()
            else:
                if action >= self.unwrapped.swap_action_size or (
                    self.unwrapped.action_mode == 'set_then_join_order' and self.set_base_join_order is None
                ):
                    raise ValueError(f'Join-order action is invalid before SET selection: {action}')
                tag = -1
                for i in range(len(self.unwrapped.action_inteval)):
                    if action < self.unwrapped.action_inteval[i]:
                        tag = i
                        break
                if tag != -1:
                    t1 = tag - 1
                    t2 = action - self.unwrapped.action_inteval[t1] + tag
                    # Swap operation.
                    temp = self.hintdict['join order'][t1]
                    self.hintdict['join order'][t1] = self.hintdict['join order'][t2]
                    self.hintdict['join order'][t2] = temp 
                    if self.unwrapped.action_mode == 'join_order_and_operator':
                        self._rebuild_swap_followup_mask(t1, t2)
                    else:
                        self._rebuild_join_order_action_mask()
                exechint = self.planhelper.to_exechint(self.hintdict)
                self.currlatency = self.planhelper.tryGetLatency(exechint, self.db, self.query_id)
                feature_dict, _, _, cost_plan_json = self.planhelper.get_feature(
                    exechint, self.sql, False, self.db, query_id=self.query_id
                )
        except Exception:
            # Catch exceptions and print diagnostic information.
            print(self.query_id, self.hintdict, self.action_mask)
            raise
        # ========Determine if there are duplicates=====
        isloop = False
        for hint in self.candidatehint:
            if self._same_candidate_hint(hint, self.hintdict):
                isloop = True
                break
        if not isloop:
            # Add the current hint only if it is not already a candidate.
            self.candidatehint.append(deepcopy(self.hintdict))
        self.plan_time_cum_ms += float(cost_plan_json.get('Planning Time', 0.0)) * 1000.0
        feature_dict['steps'] = np.array(
            [self.stepnum * 1.0 / self.config.maxsteps], dtype=np.float32
        )
        if self.unwrapped.action_mode == 'set_then_join_order':
            feature_dict['set_hint_bits'] = self.unwrapped.hint_set_bits(self.hintdict['hint set'])
        else:
            feature_dict['set_hint_bits'] = np.ones(len(self.config.set_hint_gucs), dtype=np.float32)
        currplan = deepcopy(feature_dict)
        feature_dict['action_mask'] = self.action_mask
        self.currplan = deepcopy(currplan)
        self.curr_obs = deepcopy(feature_dict)
        #=========calculate penalty=======
        minsteps = (
            min_steps(self.candidatehint[0], self.hintdict)
            if self.unwrapped.action_mode == 'join_order_and_operator'
            else 1 + min_swap_steps(self.set_base_join_order, self.hintdict['join order'])
        )
        penalty = (minsteps - self.stepnum) * self.config.penalty_coeff  # Control the explicit step penalty independently of the reward objective.

        bounty = 0
        reward_info = build_reward_diagnostic('duplicate')
        if not isloop:
            score_margin = None
            curr_score = None
            if self.currlatency is None or getattr(self.config, 'planner_reward_diagnostics', False):
                _, _, curr_score, score_margin = self.scorer_trainer.compare_features(
                    self._scorer_feature(self.esbestplan),
                    self._scorer_feature(currplan),
                    risk_weight=float(getattr(self.config, 'planner_surrogate_risk_weight', 0.0)),
                )
            if self.currlatency is None:
                curr_objective = self._score_to_objective_ms(curr_score)
                reward_info = build_reward_diagnostic('scorer_surrogate', predicted_score=curr_score)
            else:
                curr_objective = self._reward_objective_value(
                    self.currlatency, self.plan_time_cum_ms
                )
                true_score = math.log(
                    (self.baseline_reward_objective + 1e-6) / (curr_objective + 1e-6)
                )
                reward_info = build_reward_diagnostic(
                    'real_latency',
                    predicted_score=curr_score,
                    true_score=true_score,
                )
            bounty = self._clip_reward(math.log((self.best_reward_objective_so_far + 1e-6) / (curr_objective + 1e-6)))
            if curr_objective + 1e-6 < self.best_reward_objective_so_far:
                self.optimlatency = self.currlatency
                self.esbesthint   = exechint
                self.beststeps    = minsteps
                self.esbestplan   = currplan
                self.best_candidate_plan = deepcopy(currplan)
                self.best_reward_objective_so_far = curr_objective
            if self.currlatency == None and self.isCollectSamples:
                prob_right = 1.0 / (1.0 + math.exp(-float(np.clip(score_margin, -20.0, 20.0))))
                self.bpm.add_iterCandidate.remote(
                    self.db,
                    self.query_id,
                    exechint,
                    currplan,
                    self.sql,
                    [1.0 - prob_right, prob_right],
                    planning_time_cum=self.plan_time_cum_ms,
                )
        if self.stepnum >= self.config.maxsteps:
            self.is_done = True
            self.truncated = True
            terminal_bounty, _ = self._finalize_episode_bounty()
            bounty = bounty + terminal_bounty

        reward = penalty + bounty #+ self.basebonus
        # print("Feature:", feature_dict)
        return feature_dict, reward, self.is_done, self.truncated, reward_info


class DynaHintEnvTrain(MultiAgentEnv):

    def __init__(self,out_config):
        super().__init__()
        self.config      = out_config['genConfig']
        self.scorer_trainer = ScorerTrainer(self.config, device=self.config.device)
        self.planhelper  = PlanHelper(self.config)
        self.querymanger = QueryManager(self.config, planhelper = self.planhelper,isremote=False)
        self.bpm         = ray.get_actor('bpm')
        best_executed_features = ray.get(self.bpm.get_best_executed_plan.remote())
        self.querymanger.update_best_executed_plan(best_executed_features)
        median_hint_latency = ray.get(self.bpm.get_median_plan.remote())
        self.querymanger.update_Median(median_hint_latency)
        self.tmp_scorer_dir = os.path.dirname(self.config.scorer_path)
        self.tmp_scorer_path = os.path.join(self.tmp_scorer_dir, 'tmp_scorer.pt')
        # RL_esbest_feature = ray.get(self.bpm.get_RL_esbest.remote())
        # self.querymanger.update_RL_esbest(RL_esbest_feature)
        self.scorer_version = None
        _, _, self.scorer_planner_iteration = ray.get(self.bpm.get_stateNo.remote())
        current_signature = self._get_scorer_signature(self.tmp_scorer_path)
        if current_signature is not None:
            if self.scorer_trainer.try_load_model(self.tmp_scorer_path, load_tag='env_init_tmp_scorer'):
                self.scorer_version = current_signature
        tableNum = self.planhelper.get_table_num()
        env_config = {'scorer_trainer':self.scorer_trainer,'querymanger':self.querymanger,
                      'bestplanmanager':self.bpm,'planhelper':self.planhelper,'tableNum':tableNum, 'genConfig':self.config}
        self.agents = [DynaHintEnvTrainBase(env_config) for _ in range(self.config.num_agents)]
        self._agent_ids = set(range(self.config.num_agents))
        self.observation_space = self.agents[0].observation_space
        self.action_space = self.agents[0].action_space
        self.scorer_update_times = 0
        self.resetted = False

    def _get_scorer_signature(self, model_path):
        if not os.path.exists(model_path):
            return None
        try:
            return (os.path.getmtime(model_path), os.path.getsize(model_path))  # Use modification time and size together to detect checkpoint changes.
        except OSError:
            return None

    def reset(self, *, seed=None, options=None):
        if not self.resetted:
            self.resetted = True
            return {},{}
        super().reset(seed=seed)
        _, _, current_planner_iteration = ray.get(self.bpm.get_stateNo.remote())
        now_signature = self._get_scorer_signature(self.tmp_scorer_path)
        version_changed = now_signature is not None and now_signature != self.scorer_version
        if should_reload_scorer(
            self.config,
            version_changed,
            current_planner_iteration,
            self.scorer_planner_iteration,
        ):
            queryImportance = ray.get(self.bpm.update_weightsByRLesbest.remote())
            self.querymanger.updateBuffer(queryImportance)
            latencyBuffer = ray.get(self.bpm.get_latencyBuffer.remote())
            self.planhelper.updatePGLatencyBuffer(latencyBuffer)
            best_executed_features = ray.get(self.bpm.get_best_executed_plan.remote())
            self.querymanger.update_best_executed_plan(best_executed_features)
            median_hint_latency = ray.get(self.bpm.get_median_plan.remote())
            self.querymanger.update_Median(median_hint_latency)
            if self.scorer_trainer.try_load_model(self.tmp_scorer_path, load_tag='env_reset_tmp_scorer'):
                self.scorer_version = now_signature
                self.scorer_update_times += 1
        self.scorer_planner_iteration = current_planner_iteration
        self.resetted = True
        self.terminateds = set()
        self.truncateds = set()
        reset_results = [a.reset(options={'scorer_times':self.scorer_update_times}) for a in self.agents]
        # self.epi += 1
        return (
            {i: oi[0] for i, oi in enumerate(reset_results)},
            {i: oi[1] for i, oi in enumerate(reset_results)},
        )

    def step(self, action_dict):
        obs, rew, terminated, truncated, info = {}, {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], terminated[i], truncated[i], info[i] = self.agents[i].step(action)
            if terminated[i]:
                self.terminateds.add(i)
            if truncated[i]:
                self.truncateds.add(i)
        terminated["__all__"] = len(self.terminateds) == len(self.agents)
        truncated["__all__"] = len(self.truncateds) == len(self.agents)
        return obs, rew, terminated, truncated, info


class DynaHintEnvTest(MultiAgentEnv):
    def __init__(self,env_config):
        super().__init__()
        self.planhelper = env_config['planhelper']
        self.config     = env_config['genConfig']
        self.numagents  = self.config.num_agents
        
        tableNum = ray.get(self.planhelper.GetTableNum.remote())
        self.unwrapped_env = DynaHintEnvBase({'tableNum':tableNum, 'genConfig':self.config})
        self._agent_ids = set(range(self.numagents))
        self.observation_space = self.unwrapped_env.observation_space
        self.action_space = self.unwrapped_env.action_space
        self.resetted = False

    def _allow_stop(self):
        return self.stepnum >= int(getattr(self.config, 'test_stop_min_steps', 0))

    def _enable_stop_action(self, action_mask):
        action_mask[self.unwrapped_env.stop_action_idx] = 1 if self._allow_stop() else 0  # Restrict early STOP in evaluation and inference to preserve candidate diversity.

    def _enable_swap_actions(self, action_mask):
        for i in range(self.count_table - 1):
            action_mask[self.unwrapped_env.action_inteval[i]: self.unwrapped_env.action_inteval[i] + self.count_table - i - 1] = 1  # Enable join-order swaps applicable to this query.

    def _enable_operator_group(self, action_mask, join_idx):
        if 0 <= join_idx < self.count_table - 1:
            start, end = self.unwrapped_env.operator_group_bounds(join_idx)
            action_mask[start:end] = 1

    def _rebuild_full_action_mask(self, action_mask, hintdict):
        action_mask.fill(0)
        self._enable_swap_actions(action_mask)
        for join_idx in range(self.count_table - 1):
            self._enable_operator_group(action_mask, join_idx)
        for join_idx, operator in enumerate(hintdict['join operator']):
            action_mask[self.unwrapped_env.operator_action_index(
                join_idx, self.config.OperatorDict[operator]
            )] = 0
        self._enable_stop_action(action_mask)

    def _rebuild_swap_followup_mask(self, action_mask, left, right):
        action_mask.fill(0)
        if left <= 1 or right == 1:
            self._enable_operator_group(action_mask, 0)
        else:
            self._enable_operator_group(action_mask, left - 1)
            self._enable_operator_group(action_mask, right - 1)
        self._enable_stop_action(action_mask)

    def _rebuild_initial_action_mask(self, action_mask):
        if self.unwrapped_env.action_mode == 'join_order_and_operator':
            self._rebuild_full_action_mask(action_mask, self.hintdict)
        else:
            action_mask.fill(0)
            self.unwrapped_env.enable_valid_set_actions(action_mask)

    def _rebuild_join_order_action_mask(self, action_mask):
        action_mask.fill(0)
        self._enable_swap_actions(action_mask)
        self._enable_stop_action(action_mask)
        
    def reset(self, *, seed=None, options=None):
        # Reset the environment.
        
        super().reset(seed=seed)
        self.stepnum = 0
        self.db = options['database']
        self.sql = options['sql']
        self.query_id = options['query_id']
        self.cost_plan_timeout = options.get('cost_plan_timeout')

        prepared = options.get('initial_plan_feature')
        if prepared is None:
            prepared = ray.get(
                self.planhelper.GetFeature.remote(
                    '', self.sql, True, self.db, query_id=self.query_id,
                    need_card_label=False, cost_plan_timeout=self.cost_plan_timeout,
                )
            )
        feature_dict,self.hintdict,left_deep,cost_plan_json = deepcopy(prepared)
        self.plantime = float(cost_plan_json.get('Planning Time', 0.0)) * 1000.0

        self.count_table = len(self.hintdict['join order'])
        
        if self.count_table > 1:
            assert self.count_table == len(self.hintdict['join operator']) + 1
            self.can_optimize = True
        else:
            self.can_optimize = False
            print(self.query_id + ': no optimization opportunities; skipping DynaHint')
        
        # init action
        self.action_mask = np.zeros(
            self.unwrapped_env.action_space_size, dtype=np.int32
        )
        self._rebuild_initial_action_mask(self.action_mask)
        # process state
        feature_dict['steps'] = np.array(
            [self.stepnum * 1.0 / self.config.maxsteps], dtype=np.float32
        )
        feature_dict['set_hint_bits'] = np.ones(
            len(self.config.set_hint_gucs), dtype=np.float32
        )
        state = feature_dict.copy()
        state['action_mask'] = self.action_mask
        state_all = {}
        self.action_mask_tatal = []
        self.hintdict_total = []
        self.es_hint = []
        
        for i in range(self.numagents):
            self.hintdict_total.append(deepcopy(self.hintdict))
            self.action_mask_tatal.append(deepcopy(self.action_mask))
            state_all.update({i:deepcopy(state)})
            self.es_hint.append('')
        self.last_state_all = deepcopy(state_all)
        info = {
            'hint':'',
            'can_optimize':self.can_optimize,
            'planning_time_cum_ms': self.plantime,
        }
        return (state_all,info)
    
    def step(self, action_dict):
        # Execute an action and update state.
        self.stepnum += 1
        state = {}
        rew, terminated, truncated, info =  {}, {'__all__':False}, {'__all__':False},{}
        stop_taken = self._allow_stop() and any(action == self.unwrapped_env.stop_action_idx for action in action_dict.values())
        if stop_taken:
            terminated['__all__'] = True
            for agent_id in action_dict:
                rew[agent_id] = 0
                info[agent_id] = {
                    'record_candidate': False,
                    'stop_taken': True,
                    'stop_step': self.stepnum,
                    'terminate_reason': 'stop',
                }
            return deepcopy(self.last_state_all), rew, terminated, truncated, info
        for agent_id, action in action_dict.items():
            if self.unwrapped_env.action_mode == 'join_order_and_operator' and action >= self.unwrapped_env.swap_action_size:
                idx = self.unwrapped_env.stop_action_idx - 1 - action
                self.hintdict_total[agent_id]['join operator'][idx // 3] = self.config.Operatortype[idx % 3]
                self._rebuild_full_action_mask(
                    self.action_mask_tatal[agent_id], self.hintdict_total[agent_id]
                )
                exechint = ray.get(self.planhelper.GetExechint.remote(self.hintdict_total[agent_id]))
                feature_dict, _, _, cost_plan_json = ray.get(
                    self.planhelper.GetFeature.remote(
                        exechint, self.sql, False, self.db, query_id=self.query_id,
                        need_card_label=False, cost_plan_timeout=self.cost_plan_timeout,
                    )
                )
            elif self.unwrapped_env.is_set_action(action):
                if self.stepnum != 1:
                    raise ValueError(f'SET action is only valid at step 1: {action}')
                hint_set = self.unwrapped_env.hint_set_from_action(action)
                set_only_state = {'hint set': hint_set, 'force join order': False}
                exechint = ray.get(self.planhelper.GetExechint.remote(set_only_state))
                feature_dict, replanned_hintdict, _, cost_plan_json = ray.get(
                    self.planhelper.GetFeature.remote(
                        exechint, self.sql, True, self.db, query_id=self.query_id,
                        need_card_label=False, cost_plan_timeout=self.cost_plan_timeout,
                    )
                )
                replanned_hintdict['hint set'] = hint_set
                replanned_hintdict['force join order'] = True
                self.hintdict_total[agent_id] = replanned_hintdict
                self._rebuild_join_order_action_mask(self.action_mask_tatal[agent_id])
            else:
                if action >= self.unwrapped_env.swap_action_size or (
                    self.unwrapped_env.action_mode == 'set_then_join_order'
                    and 'hint set' not in self.hintdict_total[agent_id]
                ):
                    raise ValueError(f'Join-order action is invalid before SET selection: {action}')
                tag = -1
                for i in range(len(self.unwrapped_env.action_inteval)):
                    if action < self.unwrapped_env.action_inteval[i]:
                        tag = i
                        break
                if tag != -1:
                    t1 = tag - 1
                    t2 = action - self.unwrapped_env.action_inteval[t1] + tag
                    temp = self.hintdict_total[agent_id]['join order'][t1]
                    self.hintdict_total[agent_id]['join order'][t1] = self.hintdict_total[agent_id]['join order'][t2]
                    self.hintdict_total[agent_id]['join order'][t2] = temp 
                    if self.unwrapped_env.action_mode == 'join_order_and_operator':
                        self._rebuild_swap_followup_mask(self.action_mask_tatal[agent_id], t1, t2)
                    else:
                        self._rebuild_join_order_action_mask(self.action_mask_tatal[agent_id])
                exechint = ray.get(self.planhelper.GetExechint.remote(self.hintdict_total[agent_id]))
                feature_dict, _, _, cost_plan_json = ray.get(
                    self.planhelper.GetFeature.remote(
                        exechint, self.sql, False, self.db, query_id=self.query_id,
                        need_card_label=False, cost_plan_timeout=self.cost_plan_timeout,
                    )
                )
            # Update planning time.
            self.plantime += float(cost_plan_json.get('Planning Time', 0.0)) * 1000.0
            # Update the step count.
            feature_dict['steps'] = np.array(
                [self.stepnum * 1.0 / self.config.maxsteps], dtype=np.float32
            )
            if self.unwrapped_env.action_mode == 'set_then_join_order':
                feature_dict['set_hint_bits'] = self.unwrapped_env.hint_set_bits(
                    self.hintdict_total[agent_id]['hint set']
                )
            else:
                feature_dict['set_hint_bits'] = np.ones(len(self.config.set_hint_gucs), dtype=np.float32)
            # Update state.
            state[agent_id] = feature_dict.copy()
            state[agent_id]['action_mask'] = self.action_mask_tatal[agent_id]
            # Update auxiliary information.
            info[agent_id] = {
                'hint':exechint,
                'record_candidate': True,
                'stop_taken': False,
                'stop_step': self.stepnum,
                'terminate_reason': '',
                'planning_time_cum_ms': self.plantime,
            }
            rew[agent_id] = 0
        if self.stepnum >= self.config.maxsteps:
            terminated['__all__'] = True
            truncated['__all__'] = True
            for agent_id in info:
                info[agent_id]['terminate_reason'] = 'maxsteps'
        self.last_state_all = deepcopy(state)
        return state, rew, terminated, truncated, info
