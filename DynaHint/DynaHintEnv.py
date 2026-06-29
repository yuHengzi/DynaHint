import gymnasium as gym
import numpy as np
from copy import deepcopy
from util import min_steps,get_label
import math
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import os,random
from pointestimator import PointTrainer
from manager import QueryManager
from planhelper import PlanHelper
import ray
class DynaHintEnvBase(gym.Env):
    def __init__(self,env_config):
        self.tablenum = env_config['tableNum']
        self.config   = env_config['genConfig']
        if self.config.AutoGetParam and os.path.exists(self.config.auto_config):
            data = self.GetParam()
            # self.tablenum = data["tablenum"]-1
            self.config.maxjoins = data["maxjoins"]
            self.config.num_node_feature = data["num_node_feature"]
            self.config.heightsize = data["heightsize"]
            self.config.maxnode = data["maxnode"]
            self.config.db_hist_dim = data["db_hist_dim"]
            self.config.query_feat_dim = data["query_feat_dim"]
        self.swap_action_size = int((self.tablenum * (self.tablenum - 1)) / 2)
        self.operator_action_size = int(3 * (self.tablenum - 1))
        self.action_space_size = self.swap_action_size + self.operator_action_size + 1
        self.stop_action_idx = self.action_space_size - 1
        self.observation_space = spaces.Dict({
            'x': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                            shape=(self.config.maxnode, self.config.num_node_feature)),
            'attn_bias': spaces.Box(0, 1, dtype=np.float32,
                                    shape=(self.config.maxnode + 1, self.config.maxnode + 1)),
            'heights': spaces.Box(0, self.config.heightsize, dtype=np.int64,
                                  shape=(self.config.maxnode,)),
            'action_mask': spaces.Box(0, 1, dtype=np.int32,
                                      shape=(self.action_space_size,)),
            'steps': spaces.Box(0, 1, dtype=np.float32, shape=(1,)),
            'card_label': spaces.Box(-np.inf, np.inf, dtype=np.float32, shape=(1,)),

            **({
                'db_hist': spaces.Box(0, 1, dtype=np.float32,
                                      shape=(self.config.db_hist_dim,)),
                'query_feat': spaces.Box(-np.inf, np.inf, dtype=np.float32,
                                         shape=(self.config.query_feat_dim,)),
            } if getattr(self.config, 'use_db_features', False) else {}),

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
        self.action_space = spaces.Discrete(self.action_space_size)
        self.action_inteval = [0]
        for i in range(1,self.tablenum):
            self.action_inteval.append(self.action_inteval[-1] + self.tablenum - i)

    def operator_action_index(self, join_idx, operator_idx):
        return self.stop_action_idx - (3 * join_idx + operator_idx + 1)

    def operator_group_bounds(self, join_idx):
        end = self.stop_action_idx - 3 * join_idx
        start = end - 3
        return start, end
    
    def GetParam(self):
        import json
        with open(self.config.auto_config, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        bins_per_col = data.get("db_hist_bins_per_col", getattr(self.config, "db_hist_bins_per_col", 51))
        data["node_hist_dim"] = data.get("node_hist_dim", 3 * (bins_per_col - 1))
        if "maxjoins" in data and "filtmaxnum" in data:
            data["num_node_feature"] = 7 + data["maxjoins"] + 5 * data["filtmaxnum"] + data["node_hist_dim"]
        return data
    
    def reset(self, seed=None, options=None):
        return None,None
    
    def step(self,action):
        return None,None,None,None,None

class DynaHintEnvTrainBase(gym.Wrapper):
    def __init__(self,env_config) -> None:
        unwrapped_env = DynaHintEnvBase(env_config)
        super().__init__(unwrapped_env) 
        self.planhelper  = env_config['planhelper']
        self.querymanger = env_config['querymanger']
        self.pointtrainer = env_config['pointtrainer']
        self.config      = env_config['genConfig']
        self.bpm         = env_config['bestplanmanager']
        self.querymanger.creat_trainSet()
        self.isCollectSamples = self.config.update_scorer
        splitpoint = [1.00] + self.config.splitpoint
        self.bouns_weight = [0.00]
        for i in range(len(splitpoint) - 1, 0, -1):
            self.bouns_weight.append((splitpoint[i] + splitpoint[i - 1]) / 2)
        self.episode_count = 0
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
        for k, v in base_train_feature.items():
            arr = np.asarray(v)
            print(f"{k}: shape={arr.shape}, dtype={arr.dtype}")

    def _enable_stop_action(self, action_mask):
        action_mask[self.unwrapped.stop_action_idx] = 1 if self._allow_stop() else 0

    def _allow_stop(self):
        return self.episode_count > self.config.stop_warmup_iters and self.stepnum >= self.config.stop_min_steps

    def _enable_operator_actions(self):
        if self.count_table <= 1:
            return
        for join_idx in range(self.count_table - 1):
            start, end = self.unwrapped.operator_group_bounds(join_idx)
            self.action_mask[start:end] = 1

    def _mask_current_operator_actions(self):
        for i, jo in enumerate(self.hintdict['join operator']):
            action_idx = self.unwrapped.operator_action_index(i, self.config.OperatorDict[jo])
            self.action_mask[action_idx] = 0

    def _enable_swap_actions(self):
        for i in range(self.count_table - 1):
            self.action_mask[self.action_inteval[i]:self.action_inteval[i] + self.count_table - i - 1] = 1

    def _rebuild_full_action_mask(self):
        self.action_mask.fill(0)
        self._enable_swap_actions()
        self._enable_operator_actions()
        self._mask_current_operator_actions()
        self._enable_stop_action(self.action_mask)

    def _enable_operator_group(self, join_idx):
        if join_idx < 0 or join_idx >= self.count_table - 1:
            return
        start, end = self.unwrapped.operator_group_bounds(join_idx)
        self.action_mask[start:end] = 1

    def _rebuild_swap_followup_mask(self, t1, t2):
        self.action_mask.fill(0)
        if t1 == 0 or t1 == 1 or t2 == 1:
            self._enable_operator_group(0)
        else:
            self._enable_operator_group(t1 - 1)
            self._enable_operator_group(t2 - 1)
        self._enable_stop_action(self.action_mask)

    def _clip_reward(self, value):
        return float(np.clip(value, -self.config.maxbounty, self.config.maxbounty))

    def _reward_objective_value(self, execution_latency, planning_time_cum_ms):
        execution_latency = float(execution_latency)
        if getattr(self.config, 'reward_use_total_time', False):
            return execution_latency + float(planning_time_cum_ms)
        return execution_latency

    def _score_to_objective_ms(self, score_value):
        clipped_score = float(np.clip(score_value, -6.0, 6.0))
        return float(self.baseline_reward_objective / math.exp(clipped_score))

    def _estimate_plan_objective_ms(self, plan_feature, observed_latency, planning_time_cum_ms):
        if observed_latency is not None:
            return self._reward_objective_value(observed_latency, planning_time_cum_ms)
        score_value, _ = self.pointtrainer.predict_score(plan_feature)
        return self._score_to_objective_ms(score_value)

    def _finalize_episode_bounty(self):
        eps = 1e-6
        bounty = math.log((self.baseline_reward_objective + eps) / (self.best_reward_objective_so_far + eps))
        isvalidate = self.best_reward_objective_so_far + eps < self.baseline_reward_objective
        return self._clip_reward(bounty), isvalidate

    def reset(self, seed=None, options=None):
        self.episode_count += 1
        self.stepnum = 0
        self.scorer_update_times = options['scorer_times']
        self.candidatehint = []
        self.sql, base_train_feature,self.db,self.query_id,self.scorer_best,self.RL_best = self.querymanger.get2train()
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
        self.action_mask = np.zeros(self.action_space_size)
        self._rebuild_full_action_mask()
        # process state
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
        self.baseplan = deepcopy(feature_dict)
        self.currplan = deepcopy(feature_dict)
        
        feature_dict['action_mask'] = self.action_mask
        self.curr_obs = deepcopy(feature_dict)
        # self.scorer_best[0]['action_mask'] = self.action_mask # 
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
        self.is_done = False
        self.truncated = False
        return feature_dict,{}
    def step(self,action):
        self.stepnum += 1
        if action == self.unwrapped.stop_action_idx:
            self.is_done = True
            self.truncated = False
            bounty, _ = self._finalize_episode_bounty()
            stop_feature = deepcopy(self.curr_obs)
            stop_feature['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
            stop_feature['action_mask'] = self.action_mask
            return stop_feature, bounty, self.is_done, self.truncated, {}
        # =============act on ICP and update action mask===========
        try:
            if action >= self.unwrapped.swap_action_size:
                idx = (self.unwrapped.stop_action_idx - 1) - action
                self.hintdict['join operator'][int(idx / 3)] = self.config.Operatortype[idx % 3]
                self._rebuild_full_action_mask()
                # self.isswapC = False
            else:
                tag = -1
                # self.isswapC = True
                for i in range(len(self.unwrapped.action_inteval)):
                    if action < self.unwrapped.action_inteval[i]:
                        tag = i
                        break
                if tag != -1:
                    t1 = tag - 1
                    t2 = action - self.unwrapped.action_inteval[t1] + tag
                    temp = self.hintdict['join order'][t1]
                    self.hintdict['join order'][t1] = self.hintdict['join order'][t2]
                    self.hintdict['join order'][t2] = temp 
                    self._rebuild_swap_followup_mask(t1, t2)
        except:
            print(self.query_id, self.hintdict, self.action_mask)
            raise ValueError('Action Invalid')
        # ========Determine if there are duplicates=====
        isloop = False
        for hint in self.candidatehint:
            if hint['join order'] == self.hintdict['join order'] and hint['join operator'] == self.hintdict['join operator']:
                isloop = True
                break
        if not isloop:
            self.candidatehint.append(deepcopy(self.hintdict))
        #=====get CP from ICP=========
        exechint = self.planhelper.to_exechint(self.hintdict)
        self.currlatency = self.planhelper.tryGetLatency(exechint,self.db,self.query_id)
        feature_dict,_,_,cost_plan_json = self.planhelper.get_feature(exechint, self.sql, False, self.db, query_id = self.query_id)
        self.plan_time_cum_ms += float(cost_plan_json.get('Planning Time', 0.0)) * 1000.0
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
        currplan = deepcopy(feature_dict)
        feature_dict['action_mask'] = self.action_mask
        self.currplan = deepcopy(currplan)
        self.curr_obs = deepcopy(feature_dict)
        #=========calculate penalty=======
        minsteps = min_steps(self.candidatehint[0], self.hintdict)
        penalty = (minsteps - self.stepnum) * self.config.penalty_coeff

        bounty = 0
        if not isloop:
            betteridx, _, _, score_margin = self.pointtrainer.compare_features(self.esbestplan, currplan)
            curr_objective = self._estimate_plan_objective_ms(currplan, self.currlatency, self.plan_time_cum_ms)
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
        return feature_dict, reward, self.is_done,self.truncated,{}


class DynaHintEnvTrain(MultiAgentEnv):

    def __init__(self,out_config):
        super().__init__()
        self.config      = out_config['genConfig']
        self.pointtrainer = PointTrainer(self.config, device=self.config.device)
        self.planhelper  = PlanHelper(self.config)
        self.querymanger = QueryManager(self.config, planhelper = self.planhelper,isremote=False)
        self.bpm         = ray.get_actor('bpm')
        scorer_esbest_feature = ray.get(self.bpm.get_scorer_esbest.remote())
        self.querymanger.update_scorer_esbest(scorer_esbest_feature)
        median_hint_latency = ray.get(self.bpm.get_median_plan.remote())
        self.querymanger.update_Median(median_hint_latency)
        self.tmp_scorer_dir = os.path.dirname(self.config.scorer_path)
        self.tmp_scorer_path = os.path.join(self.tmp_scorer_dir, 'tmp_scorer.pt')
        # RL_esbest_feature = ray.get(self.bpm.get_RL_esbest.remote())
        # self.querymanger.update_RL_esbest(RL_esbest_feature)
        self.scorerversion = None
        current_signature = self._get_scorer_signature(self.tmp_scorer_path)
        if current_signature is not None:
            if self.pointtrainer.try_load_model(self.tmp_scorer_path, load_tag='env_init_tmp_scorer'):
                self.scorerversion = current_signature
        tableNum = self.planhelper.get_table_num()
        env_config = {'pointtrainer':self.pointtrainer,'querymanger':self.querymanger,
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
            return (os.path.getmtime(model_path), os.path.getsize(model_path))
        except OSError:
            return None

    def reset(self, *, seed=None, options=None):
        if not self.resetted:
            self.resetted = True
            return {},{}
        super().reset(seed=seed)
        now_signature = self._get_scorer_signature(self.tmp_scorer_path)
        if now_signature is not None:
            if now_signature != self.scorerversion:
                queryImportance = ray.get(self.bpm.update_weightsByRLesbest.remote())
                self.querymanger.updateBuffer(queryImportance) 
                latencyBuffer = ray.get(self.bpm.get_latencyBuffer.remote())
                self.planhelper.updatePGLatencyBuffer(latencyBuffer)
                scorer_esbest_feature = ray.get(self.bpm.get_scorer_esbest.remote())
                self.querymanger.update_scorer_esbest(scorer_esbest_feature)
                median_hint_latency = ray.get(self.bpm.get_median_plan.remote())
                self.querymanger.update_Median(median_hint_latency)
                # RL_esbest_feature = ray.get(self.bpm.get_RL_esbest.remote())
                # self.querymanger.update_RL_esbest(RL_esbest_feature)
                if self.pointtrainer.try_load_model(self.tmp_scorer_path, load_tag='env_reset_tmp_scorer'):
                    self.scorerversion = now_signature
                    self.scorer_update_times += 1
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
        action_mask[self.unwrapped_env.stop_action_idx] = 1 if self._allow_stop() else 0

    def _enable_operator_actions(self, action_mask):
        if self.count_table <= 1:
            return
        for join_idx in range(self.count_table - 1):
            start, end = self.unwrapped_env.operator_group_bounds(join_idx)
            action_mask[start:end] = 1

    def _mask_current_operator_actions(self, action_mask, hintdict):
        for i, jo in enumerate(hintdict['join operator']):
            action_idx = self.unwrapped_env.operator_action_index(i, self.config.OperatorDict[jo])
            action_mask[action_idx] = 0

    def _enable_swap_actions(self, action_mask):
        for i in range(self.count_table - 1):
            action_mask[self.unwrapped_env.action_inteval[i]: self.unwrapped_env.action_inteval[i] + self.count_table - i - 1] = 1

    def _rebuild_full_action_mask(self, action_mask, hintdict):
        action_mask.fill(0)
        self._enable_swap_actions(action_mask)
        self._enable_operator_actions(action_mask)
        self._mask_current_operator_actions(action_mask, hintdict)
        self._enable_stop_action(action_mask)

    def _enable_operator_group(self, action_mask, join_idx):
        if join_idx < 0 or join_idx >= self.count_table - 1:
            return
        start, end = self.unwrapped_env.operator_group_bounds(join_idx)
        action_mask[start:end] = 1

    def _rebuild_swap_followup_mask(self, action_mask, t1, t2):
        action_mask.fill(0)
        if t1 == 0 or t1 == 1 or t2 == 1:
            self._enable_operator_group(action_mask, 0)
        else:
            self._enable_operator_group(action_mask, t1 - 1)
            self._enable_operator_group(action_mask, t2 - 1)
        self._enable_stop_action(action_mask)
        
    def reset(self, *, seed=None, options=None):
        
        super().reset(seed=seed)
        self.stepnum = 0
        self.db = options['database']
        self.sql = options['sql']
        self.query_id = options['query_id']

        feature_dict,self.hintdict,left_deep,cost_plan_json = ray.get(
            self.planhelper.GetFeature.remote('', self.sql, True, self.db, query_id=self.query_id, need_card_label=False)
        )
        self.plantime = float(cost_plan_json.get('Planning Time', 0.0)) * 1000.0

        self.count_table = len(self.hintdict['join order'])
        
        if self.count_table > 1:
            assert self.count_table == len(self.hintdict['join operator']) + 1
            self.use_DynaHint = True
        else:
            self.use_DynaHint = False
            print(self.query_id + ' has no optimization space. DynaHint is not needed.')
        
        # init action
        self.action_mask = np.zeros(self.unwrapped_env.action_space_size)
        self._rebuild_full_action_mask(self.action_mask, self.hintdict)
        # process state
        feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
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
            'useDynaHint':self.use_DynaHint,
            'planning_time_cum_ms': self.plantime,
        }
        return (state_all,info)
    
    def step(self, action_dict):
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
            if action >= self.unwrapped_env.swap_action_size:
                idx = (self.unwrapped_env.stop_action_idx - 1) - action
                self.hintdict_total[agent_id]['join operator'][int(idx/3)] = self.config.Operatortype[idx % 3]
                self._rebuild_full_action_mask(self.action_mask_tatal[agent_id], self.hintdict_total[agent_id])
            
            else:
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
                    self._rebuild_swap_followup_mask(self.action_mask_tatal[agent_id], t1, t2)
            exechint = ray.get(self.planhelper.GetExechint.remote(self.hintdict_total[agent_id]))
            feature_dict,_,_,cost_plan_json = ray.get(
                self.planhelper.GetFeature.remote(exechint, self.sql, False, self.db, query_id=self.query_id, need_card_label=False)
            )
            self.plantime += float(cost_plan_json.get('Planning Time', 0.0)) * 1000.0
            feature_dict['steps'] = np.array([self.stepnum * 1.0 / self.config.maxsteps])
            state[agent_id] = feature_dict.copy()
            state[agent_id]['action_mask'] = self.action_mask_tatal[agent_id]
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
