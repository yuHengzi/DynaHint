import torch
import numpy as np
from collections import deque,defaultdict
from database_util import *
from encoding import Encoding, is_database_scoped_column_min_max_vals
import os, re
import ray
import copy
from util import swap_dict_items, load_db_feature_bank, get_db_features, build_query_feat_from_tables, make_writable_np, build_query_hist_vector, load_histogram_json_as_df
class PlanHelper:
    def __init__(self, globalConfig):
        self.config = globalConfig
        encoding_dirty = False
        if self.config.DBMS == 'postgres':
            from pghelper import PGHelper
            self.dbrunner = PGHelper(self.config)
        else:
            raise ValueError('DBMS not supported')
        self.encoding = Encoding(self.config)
        if os.path.exists(self.config.encoding_path):
            self.encoding.load_from_file(self.config.encoding_path)
            try:
                column_data_properties = self.dbrunner.get_column_data_properties()
                if self.encoding.column_min_max_vals is None:
                    self.encoding.column_min_max_vals = column_data_properties
                    encoding_dirty = True
                else:
                    if not is_database_scoped_column_min_max_vals(self.encoding.column_min_max_vals):
                        self.encoding.column_min_max_vals = column_data_properties
                        encoding_dirty = True
            except Exception as e:
                print(f"[Warn] Failed to refresh column data properties from catalog: {e}")
        else:
            print(' Init Encoding......')
            column_data_properties = self.dbrunner.get_column_data_properties()
            self.encoding.loadcdp(column_data_properties)
            encoding_dirty = True
        self.dbrunner.bind_encoding(self.encoding)
        if getattr(self.config, "enable_db_meta", False):
            try:
                if self.dbrunner.ensure_exact_table_row_counts(self.encoding):
                    encoding_dirty = True
            except Exception as e:
                print(f"[Warn] Failed to ensure exact table row counts: {e}")
        if encoding_dirty:
            self._save_encoding_cache()
        if getattr(self.config, "enable_db_meta", False):
            self.dbrunner.initialize_db_meta(self.encoding)
        self.alias2table = {}

        # ===== Optional: load per-database histogram features (for data drift) =====
        # If config.use_db_features=True and config.db_feature_npz exists, load it once here.
        self.db_feature_bank = None
        if getattr(self.config, "use_db_features", False):
            try:
                npz_path = getattr(self.config, "db_feature_npz", None)
                if npz_path and os.path.exists(npz_path):
                    self.db_feature_bank = load_db_feature_bank(npz_path, databases=getattr(self.config, "databases", None))
            except Exception as e:
                print(f"[Warn] Failed to load db_feature_npz: {e}")
                self.db_feature_bank = None

        self._curr_tables_used = set()
        self._num_joins = 0
        self._num_filters = 0
        self.table_names = list(getattr(self.dbrunner, 'table_names', []))
        self.table_name_to_pos = {t: i for i, t in enumerate(self.table_names)}
        self._curr_cols_used = set()
        self._curr_hist_file = None
        self._query_static_feature_cache = {}

    def _save_encoding_cache(self):
        if not os.path.exists(os.path.dirname(self.config.encoding_path)):
            os.makedirs(os.path.dirname(self.config.encoding_path), exist_ok=True)
        self.encoding.save_to_file(self.config.encoding_path)

    def _make_query_cache_key(self, database, query_id, sql):
        return (database, query_id, hash(sql))

    def _clone_static_feature_bundle(self, feature_bundle):
        cloned = {}
        for key, value in feature_bundle.items():
            cloned[key] = make_writable_np(np.array(value, copy=True)).astype(np.float32)
        return cloned

    def _build_db_feature_bundle(self, database, query_id):
        feature_bundle = {}
        hist_dim = int(getattr(self.config, 'db_hist_dim', 0))
        src = getattr(self.config, 'db_feature_source', 'bank')
        if src == 'DynaHint':
            db_hist = build_query_hist_vector(database, getattr(self, '_curr_filter_dict', None), self.encoding, self.config)
        else:
            db_hist = get_db_features(self.db_feature_bank, database, hist_dim=hist_dim)
        if db_hist is not None:
            feature_bundle['db_hist'] = np.asarray(db_hist, dtype=np.float32)
        try:
            query_feat = build_query_feat_from_tables(
                tables_used=list(self._curr_tables_used),
                table2idx=self.encoding.table2idx,
                tablenum=len(self.encoding.table2idx),
                num_joins=int(self._num_joins),
                num_filters=int(self._num_filters),
            )
            feature_bundle['query_feat'] = np.asarray(query_feat, dtype=np.float32)
        except Exception:
            pass
        return feature_bundle

    def _build_db_meta_feature_bundle(self, database: str):
        feature_bundle = {}
        table_feat, global_feat, col_stats = self.dbrunner.get_db_meta(database)
        if table_feat is None:
            return feature_bundle

        table_names = list(getattr(self.dbrunner, 'table_names', []))
        tablenum = len(table_names) if len(table_names) > 0 else int(getattr(self.config, 'tablenum', 0))
        d_table = int(getattr(self.config, 'db_table_feat_dim', 6))

        tf = np.asarray(table_feat, dtype=np.float32)
        if tf.ndim != 2:
            tf = tf.reshape((-1, d_table))
        if tf.shape[1] != d_table:
            if tf.shape[1] > d_table:
                tf = tf[:, :d_table]
            else:
                pad = np.zeros((tf.shape[0], d_table - tf.shape[1]), dtype=np.float32)
                tf = np.concatenate([tf, pad], axis=1)
        if tf.shape[0] < tablenum:
            pad = np.zeros((tablenum - tf.shape[0], d_table), dtype=np.float32)
            tf = np.concatenate([tf, pad], axis=0)
        elif tf.shape[0] > tablenum:
            tf = tf[:tablenum, :]
        feature_bundle['db_table_stats'] = tf

        d_global = int(getattr(self.config, 'db_global_feat_dim', 4))
        gf = np.asarray(global_feat, dtype=np.float32).reshape(-1)
        if gf.shape[0] != d_global:
            if gf.shape[0] > d_global:
                gf = gf[:d_global]
            else:
                gf = np.concatenate([gf, np.zeros((d_global - gf.shape[0],), dtype=np.float32)], axis=0)
        feature_bundle['db_global_stats'] = gf

        mask = np.zeros((tablenum,), dtype=np.float32)
        name_to_pos = {t: i for i, t in enumerate(table_names)}
        for t in self._curr_tables_used:
            idx = name_to_pos.get(t, None)
            if idx is None:
                continue
            idx = int(idx)
            if 0 <= idx < tablenum:
                mask[idx] = 1.0
        feature_bundle['query_table_mask'] = mask

        d_col = int(getattr(self.config, 'db_col_feat_dim', 4))
        if (col_stats is None) or (len(self._curr_cols_used) == 0):
            qcol = np.zeros((d_col,), dtype=np.float32)
        else:
            ndv_list, null_list, w_list, mcv_list = [], [], [], []
            for col in self._curr_cols_used:
                v = col_stats.get(col, None)
                if v is None:
                    continue
                nd, null_frac, avg_w, mcv1 = v
                try:
                    nd = float(nd)
                except Exception:
                    nd = 0.0
                ndv_list.append(np.log1p(abs(nd)))
                null_list.append(float(null_frac) if null_frac is not None else 0.0)
                w_list.append(np.log1p(float(avg_w) if avg_w is not None else 0.0))
                mcv = float(mcv1) if mcv1 is not None else 0.0
                mcv_list.append(mcv if np.isfinite(mcv) else 0.0)
            if len(ndv_list) == 0:
                qcol = np.zeros((d_col,), dtype=np.float32)
            else:
                qcol = np.asarray([
                    float(np.mean(ndv_list)),
                    float(np.mean(null_list)),
                    float(np.mean(w_list)),
                    float(np.mean(mcv_list)),
                ], dtype=np.float32)[:d_col]
        feature_bundle['query_col_stats'] = qcol
        return feature_bundle

    def _get_or_build_query_static_feature_bundle(self, database, query_id, sql):
        cache_key = self._make_query_cache_key(database, query_id, sql)
        if cache_key in self._query_static_feature_cache:
            return self._clone_static_feature_bundle(self._query_static_feature_cache[cache_key])

        feature_bundle = {}
        if self.config.use_db_features:
            feature_bundle.update(self._build_db_feature_bundle(database, query_id))
        if self.config.enable_db_meta:
            feature_bundle.update(self._build_db_meta_feature_bundle(database))
        self._query_static_feature_cache[cache_key] = {
            key: np.asarray(value, dtype=np.float32) for key, value in feature_bundle.items()
        }
        return self._clone_static_feature_bundle(self._query_static_feature_cache[cache_key])

    def GetParam(self):
        import json
        with open(self.config.auto_config, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        bins_per_col = data.get("db_hist_bins_per_col", getattr(self.config, "db_hist_bins_per_col", 51))
        data["node_hist_dim"] = data.get("node_hist_dim", 3 * (bins_per_col - 1))
        if "maxjoins" in data and "filtmaxnum" in data:
            data["num_node_feature"] = 7 + data["maxjoins"] + 5 * data["filtmaxnum"] + data["node_hist_dim"]
        return data
    
    def getPGLatencyBuffer(self):
        return self.dbrunner.latencyBuffer
    
    def updatePGLatencyBuffer(self,latencyBuffer):
        self.dbrunner.latencyBuffer = latencyBuffer

    def get_table_num(self):
        return self.dbrunner.get_table_num()
    
    def getLatency(self, hint, sql, db, query_id, timeout = None, hintstyle = 'DynaHint', use_buffer=True, step=0):
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.dbrunner.getLatency(hint, sql, db, query_id, timeout, hintstyle, use_buffer, step)
    
    def getMinLatency(self):
        return self.dbrunner.get_minLatency()
    
    def tryGetLatency(self, hint, db, query_id):
        return self.dbrunner.tryGetLatency(hint, db, query_id)
    
    def getLatencyNoCache(self,hint,sql,query_id,timeout = None):
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.dbrunner.getLatencyNoCache(hint, sql, query_id, timeout)
    
    def get_feature(self, exechint, sql, toextract, database, query_id = None, plan_json = None, source = 'DynaHint',is_encode = True, need_card_label = True):
        # if source!='DynaHint':
        #     print(source)
        self.encoding.set_current_database(database)
        if os.path.exists(self.config.auto_config) and self.config.AutoGetParam:
            data = self.GetParam()
            self.config.maxnode = data["maxnode"]
            self.config.node_hist_dim = data.get("node_hist_dim", getattr(self.config, "node_hist_dim", 0))
            if is_encode:
                self.config.db_hist_dim = data["db_hist_dim"]
                self.config.query_feat_dim = data["query_feat_dim"]
        # if self.config.DBMS != 'gaussdb':
        sql = sql.lower()
        if plan_json == None:
            plan_json = self.dbrunner.getCostPlanJson(exechint, sql, source, database, query_id)
        # self.extractAliasFromSql(sql)

        self.alias2table = {}
        # reset per-query trackers
        self._curr_tables_used = set()
        self._num_joins = 0
        self._num_filters = 0
        self._curr_hist_file = load_histogram_json_as_df(database, getattr(self.config, 'hist_json_dir', './experiment/histogram'))
        
        self.extract_alias_from_plan(plan_json['Plan'])
        hintdict = {'scan table':{},'join operator':{}}
        self._curr_cols_used = set()
        ori_dict = self.traversePlan(plan_json['Plan'], hintdict, query_id)
        plan_feature = self.pre_collate(ori_dict, max_node=self.config.maxnode)
        if need_card_label:
            plan_feature['card_label'] = np.array([self.dbrunner.getCardinality(database, query_id, sql=sql)], dtype=np.float32)
        else:
            plan_feature['card_label'] = np.zeros((1,), dtype=np.float32)
        if is_encode:
            plan_feature.update(self._get_or_build_query_static_feature_bundle(database, query_id, sql))
        left_deep = None
        if toextract:
            hintdict,left_deep = self.processhint(hintdict)
        # print(hintdict)
        return plan_feature, hintdict, left_deep, plan_json
    
    def get_param(self, exechint, sql, toextract, database, query_id = None, plan_json = None, hintstyle = 'DynaHint'):
        self.encoding.set_current_database(database)
        sql = sql.lower()
        if plan_json == None:
            plan_json = self.dbrunner.getCostPlanJson(exechint, sql, hintstyle, database, query_id)
        # self.extractAliasFromSql(sql)
        self.alias2table = {}
        self.extract_alias_from_plan(plan_json['Plan'])
        return self.traversePlan4GetParam(plan_json['Plan'])
    

    def extract_alias_from_plan(self, plan):
        if plan.get('Node Type') in SCANTYPE or 'Alias' in plan:
            table = plan.get('Relation Name')
            alias = plan.get('Alias')
            if alias and table:
                self.alias2table[alias] = table
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                self.extract_alias_from_plan(subplan)
        # print(self.alias2table)

    def extractAliasFromSql(self, sql):
        try:
            fromclause = re.split(r'from[\n \t]', sql, flags=re.IGNORECASE)[1]
            fromclause = re.split(r'where[\n \t]', fromclause, flags=re.IGNORECASE)[0]
            fromclause = [oneclause.strip('\n ') for oneclause in fromclause.split(',')]
        except:
            print(sql)
            raise ValueError('SQL!')
        for fc in fromclause:
            fc = fc.replace('\t','')
            fc = fc.replace('\n','')
            fc = fc.strip(' ')
            if ' as ' in fc:
                fcs = fc.split(' as ')
                self.alias2table[fcs[1]] = fcs[0]
            else:
                fcs = fc.split(' ')
                if len(fcs) == 2:
                    fcs[0] = fcs[0].strip(' ')
                    fcs[1] = fcs[1].strip(' ')
                    self.alias2table[fcs[1]] = fcs[0]
                else:
                    fcs[0] = fcs[0].strip(' ')
                    self.alias2table[fcs[0]] = fcs[0]
        # print(self.alias2table)

    def get_hintNum(self):
        hintNum = {}
        for queryid in self.dbrunner.latencyBuffer:
            hintNum[queryid] = len(self.dbrunner.latencyBuffer[queryid])
        return hintNum
    
    def processhint(self, hint):
        # print(hint['scan table'])
        left_deep = True
        ICP = {'join order':[],'join operator':[],'structure':[]}
        if len(hint['join operator']) <= 0:
            return ICP, None
        hint['join operator'] = dict(reversed(list(hint['join operator'].items())))
        encodOfJoin = list(hint['join operator'].keys())
        for k in range(0,len(encodOfJoin) - 1):
            if len(encodOfJoin[k]) == len(encodOfJoin[k + 1]) and encodOfJoin[k][-1] > encodOfJoin[k + 1][-1]:
                hint['join operator'] = swap_dict_items(hint['join operator'], encodOfJoin[k], encodOfJoin[k + 1])
        padLen = max([len(k) for k in hint['scan table'].keys()])
        sortbyencod = []
        for k in hint['scan table'].keys():
            sum_e = 2 ** (padLen - len(k)) - 1
            for i_e, e in enumerate(k[-1::-1]):
                sum_e += eval(e) * (2 ** (padLen - len(k) + i_e))
            sortbyencod.append((sum_e, k, hint['scan table'][k]))
        sortbyencod.sort(key = lambda x: x[0])
        encod = [x[1] for x in sortbyencod]
        jointable = [x[2] for x in sortbyencod]
        ICP['join order'] = [table[0] for _, _, table in sortbyencod]
        # print(ICP['join order'])
        for joinEncod in hint['join operator']:
            prefixLen = len(joinEncod)
            JoinE = []
            JoinI = []
            for i_, scanEncod in enumerate(encod):
                # print(scanEncod[0:prefixLen])
                # print(joinEncod)
                if scanEncod[0:prefixLen] == joinEncod:
                    JoinI.append(i_)
                    JoinE.append(scanEncod)
            # print(f"joinEncod: {joinEncod}, encod: {encod}, JoinI: {JoinI}")
            if len(JoinI) != 2 or (JoinI[1] - JoinI[0]) != 1:
                raise KeyError('Parse Error')
            if JoinI[0] != 0:
                left_deep = False
            encod[JoinI[0]] = joinEncod
            del  encod[JoinI[1]]
            jointable[JoinI[0]] = jointable[JoinI[0]] + jointable[JoinI[1]]
            del jointable[JoinI[1]]
            ICP['structure'].append(JoinI[0])
            ICP['join operator'].append(self.config.operator_pg2hint[hint['join operator'][joinEncod]])
        return ICP,left_deep

    def pre_collate(self, the_dict, max_node, rel_pos_max=20, alpha=0):
        x = pad_2d_unsqueeze(the_dict['features'], max_node)
        N = len(the_dict['features'])
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)
        pc_dict = the_dict['pc_dict']
        assert len(pc_dict) != 0
        distance_matrix = bfs(N ,pc_dict, rel_pos_max)
        attn_bias[1:, 1:] = torch.from_numpy(distance_matrix).float() * alpha + (1 - torch.from_numpy(distance_matrix).float())
        attn_bias[0, :] = 1
        attn_bias[:, 0] = alpha
        attn_bias[0, 0] = 1
        attn_bias = pad_attn_bias_unsqueeze(attn_bias, max_node + 1, alpha)
        heights = pad_1d_unsqueeze(the_dict['heights'], max_node)
        return {
            'x': np.array(x.numpy(), copy=True),
            'attn_bias': np.array(attn_bias.numpy(), copy=True),
            'heights': np.array(heights.numpy(), copy=True),
        }
    
    def _attach_db_meta(self, plan_feature: dict, database: str):
        plan_feature.update(self._clone_static_feature_bundle(self._build_db_meta_feature_bundle(database)))

    def _merge_query_filter_dict(self, records):
        merged = {'colId': [], 'opId': [], 'val': [], 'dtype': []}
        seen = set()
        for node, _, _, _, _ in records:
            filter_dict = getattr(node, 'filterDict', None)
            if not filter_dict:
                continue
            cols = list(filter_dict.get('colId', []))
            ops = list(filter_dict.get('opId', []))
            vals = list(filter_dict.get('val', []))
            dtypes = list(filter_dict.get('dtype', []))
            for col_id, op_id, val, dtype in zip(cols, ops, vals, dtypes):
                key = (int(col_id), int(op_id), float(val), int(dtype))
                if key in seen:
                    continue
                seen.add(key)
                merged['colId'].append(int(col_id))
                merged['opId'].append(int(op_id))
                merged['val'].append(float(val))
                merged['dtype'].append(int(dtype))
        return merged

    def _col_ids_to_names(self, col_ids):
        cols_used = set()
        for cid in col_ids:
            try:
                colname = self.encoding.idx2col.get(int(cid), None)
                if colname and colname != 'NA':
                    cols_used.add(colname)
            except Exception:
                pass
        return cols_used

    def _build_plan_node(self, plannode, pos=None, parentAlias=None, *, encode_join=True, build_feature=True):
        nodeType = plannode['Node Type']
        typeId = self.encoding.encode_type(nodeType)
        table = 'NA'
        table_id = 0
        alias = None

        if nodeType in SCANTYPE:
            try:
                table = plannode['Relation Name']
                table_id = self.encoding.encode_table(plannode['Relation Name'])
            except Exception:
                raise ValueError('Relation Name Parse Error')
            try:
                alias = plannode.get('Alias')
                if alias is not None and alias not in self.alias2table:
                    self.alias2table[alias] = table
            except Exception:
                raise ValueError('Alias Parse Error')
        elif nodeType == 'Bitmap Index Scan':
            alias = parentAlias

        join, filters, db_est = processCond(plannode, alias, self.alias2table)
        joinids = self.encoding.encode_join(join) if encode_join else 'joinids'
        filters_encoded = self.encoding.encode_filters(filters, alias, self.alias2table)

        node = TreeNode(nodeType, table, table_id, typeId, filters, joinids, filters_encoded, db_est, pos)
        node.alias = alias
        if build_feature:
            node.feature = node2feature(node, self.config, self.encoding, self._curr_hist_file)

        metadata = {
            'node_type': nodeType,
            'table': table,
            'is_scan': nodeType in SCANTYPE,
            'joinids': list(joinids) if encode_join else [],
            'filter_col_ids': list(filters_encoded.get('colId', [])),
            'filter_count': int(len(filters_encoded.get('colId', []))),
        }
        return node, metadata

    def _update_query_state(self, node_meta):
        if node_meta['node_type'] in JOINTYPE:
            self._num_joins += 1

        table = node_meta['table']
        if table and table != 'NA':
            self._curr_tables_used.add(table)

        self._num_filters += int(node_meta['filter_count'])
        self._curr_cols_used.update(self._col_ids_to_names(node_meta['joinids']))
        self._curr_cols_used.update(self._col_ids_to_names(node_meta['filter_col_ids']))

    def _update_param_state(self, node_meta):
        self._curr_cols_used.update(self._col_ids_to_names(node_meta['filter_col_ids']))

    def _traverse_plan_bfs(self, plan, *, root_pos, unary_child_pos, multi_child_pos_fn, node_builder):
        root, root_meta = node_builder(plan, pos=root_pos, parentAlias=None)
        records = []
        adj_list = []
        NodeList = deque()
        NodeList.append((root, root_meta, plan, '0', 0))
        next_id = 1

        while NodeList:
            parentNode, parentMeta, parentPlan, parentEncod, idx = NodeList.popleft()
            records.append((parentNode, parentMeta, parentPlan, parentEncod, idx))

            subplans = parentPlan.get('Plans', [])
            subPlanNum = len(subplans)
            if subPlanNum == 1:
                subplan = subplans[0]
                node, node_meta = node_builder(subplan, pos=unary_child_pos, parentAlias=parentNode.alias)
                subEncod = parentEncod + '0'
                node.parent = parentNode
                parentNode.addChild(node)
                NodeList.append((node, node_meta, subplan, subEncod, next_id))
                adj_list.append((idx, next_id))
                next_id += 1
            elif subPlanNum > 1:
                for child_idx in range(subPlanNum - 1, -1, -1):
                    subplan = subplans[child_idx]
                    node, node_meta = node_builder(subplan, pos=multi_child_pos_fn(child_idx), parentAlias=parentNode.alias)
                    subEncod = parentEncod + str(child_idx)
                    node.parent = parentNode
                    parentNode.addChild(node)
                    NodeList.append((node, node_meta, subplan, subEncod, next_id))
                    adj_list.append((idx, next_id))
                    next_id += 1

        return root, records, adj_list

    def traverseNode(self, plannode,  pos = None, parentAlias = None, query_id = None):
        node, _ = self._build_plan_node(plannode, pos=pos, parentAlias=parentAlias, encode_join=True, build_feature=True)
        return node
    
    def traversePlan(self, plan, hint, query_id): 
        # pos:{3:'root', 0:'left', 1:'right', 2:'internal-no-brother'}
        features = []
        heights = []
        root, records, adj_list = self._traverse_plan_bfs(
            plan,
            root_pos=3,
            unary_child_pos=2,
            multi_child_pos_fn=lambda child_idx: child_idx,
            node_builder=lambda subplan, pos, parentAlias: self._build_plan_node(
                subplan, pos=pos, parentAlias=parentAlias, encode_join=True, build_feature=True
            ),
        )
        self._curr_filter_dict = self._merge_query_filter_dict(records)

        for parentNode, parentMeta, parentPlan, parentEncod, idx in records:
            self._update_query_state(parentMeta)
            features.append(parentNode.feature)
            heights.append(len(parentEncod))
            if parentMeta['node_type'] in JOINTYPE:
                hint['join operator'][parentEncod] = parentPlan['Node Type']
            elif parentMeta['is_scan'] and parentNode.alias is not None:
                hint['scan table'][parentEncod] = [parentNode.alias]

        pc_dict = defaultdict(list)
        for parent, child in adj_list:
            pc_dict[parent].append(child)
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights':  torch.LongTensor(heights),
            'pc_dict':  pc_dict}
    
    def traverseNode4GetParam(self, plannode,  pos = None, parentAlias = None):
        node, _ = self._build_plan_node(plannode, pos=pos, parentAlias=parentAlias, encode_join=False, build_feature=False)
        return node
    
    def traversePlan4GetParam(self, plan): 
        heights = []
        _, records, _ = self._traverse_plan_bfs(
            plan,
            root_pos=0,
            unary_child_pos=1,
            multi_child_pos_fn=lambda child_idx: child_idx + 2,
            node_builder=lambda subplan, pos, parentAlias: self._build_plan_node(
                subplan, pos=pos, parentAlias=parentAlias, encode_join=False, build_feature=False
            ),
        )

        joinNum = 0
        filterNum = 0
        max_posNum = 0
        nodeNum = 0

        for _, parentMeta, parentPlan, parentEncod, _ in records:
            nodeNum += 1
            self._update_param_state(parentMeta)
            heights.append(len(parentEncod))
            if parentMeta['node_type'] in JOINTYPE:
                joinNum += 1
            filterNum = max(filterNum, int(parentMeta['filter_count']))
            subPlanNum = len(parentPlan.get('Plans', []))
            if subPlanNum > max_posNum:
                max_posNum = subPlanNum

        return [joinNum, filterNum, max_posNum, max(heights), nodeNum]

    def to_exechint(self, hintdict):
        if self.config.left_deep_restriction:
            hintdict['structure'] = [0] * len(hintdict['structure']) # bad pg_hint_plan!
        join_hint = []
        order_hint = copy.deepcopy(hintdict['join order'])
        join_hint_help = copy.deepcopy(hintdict['join order'])
        leading_hint = 'LEADING'
        for i in range(len(hintdict['join operator'])):
            i_struct = hintdict['structure'][i]
            join_hint_tmp = ' '.join(join_hint_help[i_struct:i_struct + 2])
            join_hint_help = join_hint_help[:i_struct] + [join_hint_tmp] + join_hint_help[i_struct + 2:]
            join_hint.append(hintdict['join operator'][i] + '('+ join_hint_tmp + ')')
            
            order_hint_tmp = '(' + ' '.join(order_hint[i_struct:i_struct + 2]) + ')'
            order_hint = order_hint[:i_struct] + [order_hint_tmp] + order_hint[i_struct + 2:]
        join_hint.reverse()
        leading_hint = leading_hint + '(' + order_hint[0] + ')'
        exechint = '/*+' + leading_hint + '\n' + '\n'.join(join_hint) + '*/\n'
        return exechint

@ray.remote
class RemotePlanHelper():
    def __init__(self,globalConfig):
        self.config = globalConfig
        self.planhelper = PlanHelper(globalConfig)
    def GetFeature(self,hint,sql,toextract, database,query_id = None,is_encode = True,source = 'DynaHint', need_card_label = True):
        return self.planhelper.get_feature(
            hint,
            sql,
            toextract,
            database,
            query_id = query_id,
            source = source,
            is_encode = is_encode,
            need_card_label = need_card_label,
        ) 
    def GetLatency(self,hint,sql, database, query_id, timeout = None, use_buffer = True, step = 0):
        if timeout == None:
            timeout = self.config.max_time_out
        return self.planhelper.getLatency(hint,sql, database, query_id, timeout=timeout, use_buffer=use_buffer, step=step)
    def GetCardinality(self, database, query_id, sql):
        return self.planhelper.dbrunner.getCardinality(database, query_id, sql=sql)
    def SaveEncoding(self,path):
        self.planhelper.encoding.save_to_file(path)
    def GetPGLatencyBuffer(self):
        return self.planhelper.getPGLatencyBuffer()
    def GetTableNum(self):
        return self.planhelper.get_table_num()
    def GetExechint(self,hintdict):
        return self.planhelper.to_exechint(hintdict)
    def GetSortedQueryID(self):
        hintNum = self.planhelper.get_hintNum()
        sorted_keys = sorted(hintNum, key=hintNum.get)
        return sorted_keys
    def GetParam(self,hint,sql,toextract, database,query_id = None):
        return self.planhelper.get_param(hint,sql, toextract, database,query_id = query_id)

if __name__ == "__main__":
    import os
    import random
    from config import Config
    config = Config()   
    config.ConfirmPath()
    train_path = 'experiment/JOB/train'
    sql_files = [f for f in os.listdir(train_path) if f.endswith('.sql')]
    chosen_sql_file = random.choice(sql_files)
    full_sql_path = os.path.join(train_path, chosen_sql_file)
    with open(full_sql_path, 'r') as f:
        sql = f.read()
    
    planhelper  = PlanHelper(config)
    hint = ''
    toextract = None
    database = "imdb"
    query_id = None
    feature = planhelper.get_feature(hint, sql, toextract, database, query_id=query_id)
