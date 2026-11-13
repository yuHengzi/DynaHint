"""PostgreSQL and openGauss plan helpers."""
import torch
import numpy as np
from collections import deque,defaultdict
from collection.database_util import *
from collection.encoding import Encoding, is_database_scoped_column_min_max_vals
import os, re
import ray
import copy
from shared.util import swap_dict_items, load_db_feature_bank, get_db_features, build_query_feat_from_tables, make_writable_np, build_query_hist_vector, build_query_bitmap_vector, load_histogram_json_as_df
class PlanHelper:
    # Plan helper for extracting query-plan features.
    def __init__(self, globalConfig):
        self.config = globalConfig
        encoding_dirty = False
        if self.config.DBMS == 'postgres':
            from collection.pghelper import PGHelper
            self.dbrunner = PGHelper(self.config)
        else:
            raise ValueError('DBMS not supported')
        self.encoding = Encoding(self.config)
        if os.path.exists(self.config.encoding_path):
            self.encoding.load_from_file(self.config.encoding_path)
            if getattr(self.config, 'encoding_frozen', False):
                self.encoding.freeze()
        else:
            print(' Init Encoding......')
        encoding_dirty = self._ensure_column_ranges() or encoding_dirty
        self.dbrunner.bind_encoding(self.encoding)  # Expose exact row counts from Encoding to PGHelper.
        if getattr(self.config, "enable_db_meta", False):
            try:
                if self.dbrunner.ensure_exact_table_row_counts(self.encoding):
                    encoding_dirty = True
            except Exception as e:
                print(f"[Warn] Failed to ensure exact table row counts: {e}")
        if encoding_dirty:
            self._save_encoding_cache()  # Persist newly collected column ranges and exact row counts for reuse.
        if getattr(self.config, "enable_db_meta", False):
            self.dbrunner.initialize_db_meta(self.encoding)
        self.alias2table = {}

        # ===== Optional: load per-database histogram / sample-bitmap features (for data drift) =====
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

        # Per-query running statistics for query features and table masks.
        self._curr_tables_used = set()
        self._num_joins = 0
        self._num_filters = 0
        # Use a fixed schema-table order for statistics and query masks.
        self.table_names = list(getattr(self.dbrunner, 'table_names', []))
        self.table_name_to_pos = {t: i for i, t in enumerate(self.table_names)}
        self._curr_cols_used = set()
        self._curr_hist_file = None
        self._query_static_feature_cache = {}  # Cache static query features for reuse across search steps.
        self._logical_graph_cache = {}  # Reuse the logical join graph independently of physical plans.

    def _save_encoding_cache(self):
        # Persist column ranges and exact row counts in the encoding cache.
        if not os.path.exists(os.path.dirname(self.config.encoding_path)):
            os.makedirs(os.path.dirname(self.config.encoding_path), exist_ok=True)
        self.encoding.save_to_file(self.config.encoding_path)

    def _ensure_column_ranges(self):
        cached = self.encoding.column_min_max_vals
        if not is_database_scoped_column_min_max_vals(cached):
            cached = {}
        missing = [
            database for database in self.config.databases
            if getattr(self.config, 'refresh_column_range_cache', False)
            or not isinstance(cached.get(database), dict)
            or not cached[database]
        ]
        if not missing:
            return False
        collected = self.dbrunner.get_column_data_properties(databases=missing)
        if any(not collected.get(database) for database in missing):
            raise RuntimeError('Column range collection incomplete for: {}'.format(', '.join(missing)))
        cached.update(collected)
        self.encoding.loadcdp(cached)
        print('Column ranges refreshed for {} database(s): {}'.format(len(missing), ', '.join(missing)), flush=True)
        return True

    def _make_query_cache_key(self, database, query_id, sql):
        return (database, query_id, hash(sql))  # Include the SQL fingerprint to distinguish queries with the same ID.

    def _clone_static_feature_bundle(self, feature_bundle):
        cloned = {}
        for key, value in feature_bundle.items():
            cloned[key] = make_writable_np(np.array(value, copy=True))  # Preserve integer dtypes for identifier features.
        return cloned

    def _build_logical_graph_bundle(self, plan, database, query_id, sql):
        cache_key = self._make_query_cache_key(database, query_id, sql)
        cached = self._logical_graph_cache.get(cache_key)
        if cached is not None:
            return cached

        logical_graph = extract_logical_join_graph(plan, self.alias2table)
        max_relations = int(self.dbrunner.get_table_num())
        max_edges = int(getattr(self.config, 'max_logical_edges', max(1, self.config.maxjoins)))
        relation_count = len(logical_graph['aliases'])
        edge_count = len(logical_graph['edges'])
        query_name = str(query_id) if query_id is not None else '<unknown>'
        if relation_count > max_relations:
            raise ValueError(
                f'logical relation overflow for query {query_name}: '
                f'{relation_count} > max_relations={max_relations}'
            )
        if edge_count > max_edges:
            raise ValueError(
                f'logical edge overflow for query {query_name}: '
                f'{edge_count} > max_logical_edges={max_edges}; rerun AutoGetParam'
            )

        rel_table_ids = np.zeros((max_relations,), dtype=np.int64)
        rel_db_positions = np.zeros((max_relations,), dtype=np.int64)
        rel_mask = np.zeros((max_relations,), dtype=np.float32)
        for idx, table_name in enumerate(logical_graph['base_tables']):
            rel_table_ids[idx] = int(self.encoding.encode_table(table_name))
            db_pos = self.table_name_to_pos.get(table_name)
            rel_db_positions[idx] = 0 if db_pos is None else int(db_pos) + 1
            rel_mask[idx] = 1.0

        edge_index = np.zeros((max_edges, 2), dtype=np.int64)
        edge_columns = np.zeros((max_edges, 2), dtype=np.int64)
        edge_ops = np.zeros((max_edges,), dtype=np.int64)
        edge_mask = np.zeros((max_edges,), dtype=np.float32)
        for idx, edge in enumerate(logical_graph['edges']):
            edge_index[idx] = [edge['left_rel_id'], edge['right_rel_id']]
            edge_columns[idx] = [
                self.encoding.encode_column(edge['left_column']),
                self.encoding.encode_column(edge['right_column']),
            ]
            edge_ops[idx] = self.encoding.encode_operator(edge['operator'])
            edge_mask[idx] = 1.0

        cached = {
            'features': {
                'logical_rel_table_ids': rel_table_ids,
                'logical_rel_db_positions': rel_db_positions,
                'logical_rel_mask': rel_mask,
                'logical_edge_index': edge_index,
                'logical_edge_columns': edge_columns,
                'logical_edge_ops': edge_ops,
                'logical_edge_mask': edge_mask,
            },
            'alias_to_rel_id': logical_graph['alias_to_rel_id'],
            'edge_count': edge_count,
            'relation_count': relation_count,
        }
        self._logical_graph_cache[cache_key] = cached
        return cached

    @staticmethod
    def _build_join_order_rel_ids(join_order, alias_to_rel_id, max_relations, query_id):
        result = np.zeros((max_relations,), dtype=np.int64)
        if len(join_order) > max_relations:
            raise ValueError(f'join order overflow for query {query_id}: {len(join_order)} > {max_relations}')
        for idx, alias in enumerate(join_order):
            if alias not in alias_to_rel_id:
                raise ValueError(f'join-order alias {alias!r} missing from logical graph for query {query_id}')
            result[idx] = alias_to_rel_id[alias]
        return result

    def _build_db_feature_bundle(self, database, query_id):
        feature_bundle = {}
        hist_dim = int(getattr(self.config, 'db_hist_dim', 0))
        bmp_dim = int(getattr(self.config, 'db_bitmap_dim', 0))
        src = getattr(self.config, 'db_feature_source', 'bank')
        if src == 'DynaHint':
            db_hist = build_query_hist_vector(database, getattr(self, '_curr_filter_dict', None), self.encoding, self.config)
            if getattr(self.config, "use_db_bitmap_features", True):
                db_bitmap = build_query_bitmap_vector(database, query_id, self.encoding, self.config)
            else:
                db_bitmap = np.zeros((0,), dtype=np.float32)
        else:
            db_hist, db_bitmap = get_db_features(self.db_feature_bank, database, hist_dim=hist_dim, bitmap_dim=bmp_dim)
            if not getattr(self.config, "use_db_bitmap_features", True):
                db_bitmap = np.zeros((0,), dtype=np.float32)
        if db_hist is not None:
            feature_bundle['db_hist'] = np.asarray(db_hist, dtype=np.float32)
        if db_bitmap is not None:
            feature_bundle['db_bitmap'] = np.asarray(db_bitmap, dtype=np.float32)
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
            return self._clone_static_feature_bundle(self._query_static_feature_cache[cache_key])  # Reuse the static branch on a cache hit.

        feature_bundle = {}
        if self.config.use_db_features:
            feature_bundle.update(self._build_db_feature_bundle(database, query_id))
        if self.config.enable_db_meta:
            feature_bundle.update(self._build_db_meta_feature_bundle(database))
        self._query_static_feature_cache[cache_key] = {
            key: np.asarray(value) for key, value in feature_bundle.items()
        }  # Cache static query features after their initial construction.
        return self._clone_static_feature_bundle(self._query_static_feature_cache[cache_key])

    def GetParam(self):
        # Get encoding dimensions.
        import json
        with open(self.config.auto_config, 'r', encoding='utf-8') as json_file:  # Open the file with UTF-8 encoding.
            data = json.load(json_file)  # Load JSON data.
        bins_per_col = data.get("db_hist_bins_per_col", getattr(self.config, "db_hist_bins_per_col", 51))
        data["node_hist_dim"] = data.get("node_hist_dim", 3 * (bins_per_col - 1))
        if "maxjoins" in data and "filtmaxnum" in data:
            data["num_node_feature"] = 7 + data["maxjoins"] + 5 * data["filtmaxnum"] + data["node_hist_dim"]
        return data
    
    def getPGLatencyBuffer(self):
        # Get the latency buffer.
        return self.dbrunner.latencyBuffer
    
    def updatePGLatencyBuffer(self,latencyBuffer):
        # Update the latency buffer.
        self.dbrunner.latencyBuffer = latencyBuffer

    def get_table_num(self):
        # Get the table count.
        return self.dbrunner.get_table_num()
    
    def getLatency(self, hint, sql, db, query_id, timeout = None, hintstyle = 'DynaHint', use_buffer=True,
                   step=0, collect_cardinality=True, measurement_phase=None):
        # Get latency.
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.dbrunner.getLatency(
            hint,
            sql,
            db,
            query_id,
            timeout,
            hintstyle,
            use_buffer,
            step,
            collect_cardinality=collect_cardinality,
            measurement_phase=measurement_phase,
        )
    
    def getMinLatency(self):
        # Get minimum latency.
        return self.dbrunner.get_minLatency()
    
    def tryGetLatency(self, hint, db, query_id):
        # Try to retrieve cached latency.
        return self.dbrunner.tryGetLatency(hint, db, query_id)
    
    def getLatencyNoCache(self,hint,sql,query_id,timeout = None):
        # Get uncached latency.
        if timeout == None or timeout >= self.config.max_time_out:
            timeout = self.config.max_time_out
        return self.dbrunner.getLatencyNoCache(hint, sql, query_id, timeout)
    
    def get_feature(self, exechint, sql, toextract, database, query_id = None, plan_json = None, source = 'DynaHint',is_encode = True, need_card_label = True, cost_plan_timeout=None):
        # Extract features from hints, SQL, query IDs and an optional cost plan.
        # if source!='DynaHint':
        #     print(source)
        self.encoding.set_current_database(database)
        if os.path.exists(self.config.auto_config) and self.config.AutoGetParam:
            data = self.GetParam()
            self.config.maxnode = data["maxnode"]
            self.config.max_logical_edges = data["max_logical_edges"]
            self.config.node_hist_dim = data.get("node_hist_dim", getattr(self.config, "node_hist_dim", 0))
            if is_encode:
                self.config.db_hist_dim = data["db_hist_dim"]
                self.config.db_bitmap_dim = data["db_bitmap_dim"]
                self.config.query_feat_dim = data["query_feat_dim"]
        # if self.config.DBMS != 'gaussdb':
        sql = sql.lower()   # Convert SQL to lowercase.
        if plan_json == None:
            plan_json = self.dbrunner.getCostPlanJson(
                exechint, sql, source, database, query_id, timeout=cost_plan_timeout
            )
        # self.extractAliasFromSql(sql)

        # Clear aliases to avoid cross-query contamination.
        self.alias2table = {}
        # reset per-query trackers
        self._curr_tables_used = set()
        self._num_joins = 0
        self._num_filters = 0
        self._curr_hist_file = load_histogram_json_as_df(database, getattr(self.config, 'hist_json_dir', './experiment/histogram'))
        
        self.extract_alias_from_plan(plan_json['Plan'])
        logical_graph = self._build_logical_graph_bundle(plan_json['Plan'], database, query_id, sql)
        hintdict = {'scan table':{},'join operator':{},'join node index':{}}
        # Columns referenced by filters and join predicates in this query.
        self._curr_cols_used = set()
        ori_dict = self.traversePlan(plan_json['Plan'], hintdict, query_id)
        plan_feature = self.pre_collate(ori_dict, max_node=self.config.maxnode)
        current_icp, parsed_left_deep = self.processhint(copy.deepcopy(hintdict))
        join_node_indices = current_icp.pop('_join node indices', [])
        max_relations = int(self.dbrunner.get_table_num())
        if self.config.action_mode == 'join_order_and_operator':
            physical_join_node_indices = np.zeros((max(0, max_relations - 1),), dtype=np.int64)
            if len(join_node_indices) > len(physical_join_node_indices):
                raise ValueError(f'physical join-node overflow for query {query_id}')
            if join_node_indices:
                physical_join_node_indices[:len(join_node_indices)] = np.asarray(join_node_indices, dtype=np.int64) + 1
            plan_feature['physical_join_node_indices'] = physical_join_node_indices
        plan_feature.update(self._clone_static_feature_bundle(logical_graph['features']))
        plan_feature['join_order_rel_ids'] = self._build_join_order_rel_ids(
            current_icp['join order'], logical_graph['alias_to_rel_id'], max_relations, query_id
        )
        if need_card_label:
            plan_feature['card_label'] = np.array([self.dbrunner.getCardinality(database, query_id, sql=sql)], dtype=np.float32)
        else:
            plan_feature['card_label'] = np.zeros((1,), dtype=np.float32)
        if is_encode:
            plan_feature.update(self._get_or_build_query_static_feature_bundle(database, query_id, sql))  # Update plan tokens while reusing cached static features across steps.
        left_deep = None
        if toextract:
            hintdict = current_icp
            left_deep = parsed_left_deep
        # print(hintdict)
        return plan_feature, hintdict, left_deep, plan_json
    
    def get_param(self, exechint, sql, toextract, database, query_id = None, plan_json = None, hintstyle = 'DynaHint'):
        # Get model dimensions.
        self.encoding.set_current_database(database)
        sql = sql.lower()   # Convert SQL to lowercase.
        if plan_json == None:
            plan_json = self.dbrunner.getCostPlanJson(exechint, sql, hintstyle, database, query_id)
        # self.extractAliasFromSql(sql)
        # Clear aliases to avoid cross-query contamination.
        self.alias2table = {}
        self.extract_alias_from_plan(plan_json['Plan'])
        params = self.traversePlan4GetParam(plan_json['Plan'])
        params.append(len(extract_logical_join_graph(plan_json['Plan'], self.alias2table)['edges']))
        return params
    

    def extract_alias_from_plan(self, plan):
        """Recursively extract plan aliases and table names into self.alias2table."""
        # Extract table names and aliases from scan nodes.
        # Read table and alias fields when present.
        if plan.get('Node Type') in SCANTYPE or 'Alias' in plan:
            table = plan.get('Relation Name')
            alias = plan.get('Alias')
            if alias and table:
                self.alias2table[alias] = table
        # Recursively process child plans.
        if 'Plans' in plan:
            for subplan in plan['Plans']:
                self.extract_alias_from_plan(subplan)
        # print(self.alias2table)

    def extractAliasFromSql(self, sql):
        # Extract aliases from SQL.
        try:
            # Locate the FROM clause with a regular expression.
            fromclause = re.split(r'from[\n \t]', sql, flags=re.IGNORECASE)[1]
            # Extract the FROM clause before WHERE.
            fromclause = re.split(r'where[\n \t]', fromclause, flags=re.IGNORECASE)[0]
            # Strip whitespace from each clause.
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
        # Get the hint count.
        hintNum = {}
        for queryid in self.dbrunner.latencyBuffer:
            hintNum[queryid] = len(self.dbrunner.latencyBuffer[queryid])
        return hintNum
    
    def processhint(self, hint):
        # Convert a hint dictionary into ICP and a left-deep flag.
        # print(hint['scan table'])
        left_deep = True
        ICP = {'join order':[],'join operator':[],'structure':[],'_join node indices':[]}
        if len(hint['join operator']) <= 0:
            ICP['join order'] = [value[0] for _, value in sorted(hint.get('scan table', {}).items())]
            return ICP, None
        hint['join operator'] = dict(reversed(list(hint['join operator'].items())))
        encodOfJoin = list(hint['join operator'].keys())
        for k in range(0,len(encodOfJoin) - 1):
            if len(encodOfJoin[k]) == len(encodOfJoin[k + 1]) and encodOfJoin[k][-1] > encodOfJoin[k + 1][-1]:
                hint['join operator'] = swap_dict_items(hint['join operator'], encodOfJoin[k], encodOfJoin[k + 1])
        padLen = max([len(k) for k in hint['scan table'].keys()])
        sortbyencod = []
        for k in hint['scan table'].keys():
            sum_e = 2 ** (padLen - len(k)) - 1  # Count bit positions from right to left with 2 ** (padLen - len(k)) - 1.
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
            if joinEncod in hint.get('join node index', {}):
                ICP['_join node indices'].append(int(hint['join node index'][joinEncod]))
        return ICP,left_deep

    def pre_collate(self, the_dict, max_node, rel_pos_max=20, alpha=0):
        # Convert the tree into features, attention bias and heights.
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
        """Attach table, database and query-column statistics as float32 features. Pad or truncate features to match the configured observation space."""
        plan_feature.update(self._clone_static_feature_bundle(self._build_db_meta_feature_bundle(database)))  # Keep the existing interface while reusing the feature-bundle builder.

    def _merge_query_filter_dict(self, records):
        merged = {'colId': [], 'opId': [], 'val': [], 'dtype': []}  # Aggregate filters across the tree for query-level histograms.
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
        cols_used = set()  # Resolve encoded column IDs to column names for query-level statistics.
        for cid in col_ids:
            try:
                colname = self.encoding.idx2col.get(int(cid), None)  # Look up each column name in the encoder.
                if colname and colname != 'NA':
                    cols_used.add(colname)
            except Exception:
                pass
        return cols_used

    def _build_plan_node(self, plannode, pos=None, parentAlias=None, *, encode_join=True, build_feature=True):
        nodeType = plannode['Node Type']  # Parse one plan node into a TreeNode and lightweight metadata without global side effects.
        typeId = self.encoding.encode_type(nodeType)
        table = 'NA'
        table_id = 0
        alias = None

        if nodeType in SCANTYPE:  # Scan nodes populate Relation Name, Alias and the alias-to-table mapping.
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
            alias = parentAlias  # Bitmap Index Scan often inherits its alias from its parent.

        join, filters, db_est = processCond(plannode, alias, self.alias2table)  # Parse joins, filters and estimates, encoding predicates when requested.
        joinids = self.encoding.encode_join(join) if encode_join else 'joinids'
        filters_encoded = self.encoding.encode_filters(filters, alias, self.alias2table)

        node = TreeNode(nodeType, table, table_id, typeId, filters, joinids, filters_encoded, db_est, pos)  # Build the node; the caller controls feature generation.
        node.alias = alias
        if build_feature:
            node.feature = node2feature(node, self.config, self.encoding, self._curr_hist_file)

        metadata = {  # Return a node summary so traversal need not parse it again.
            'node_type': nodeType,
            'table': table,
            'is_scan': nodeType in SCANTYPE,
            'joinids': list(joinids) if encode_join else [],
            'filter_col_ids': list(filters_encoded.get('colId', [])),
            'filter_count': int(len(filters_encoded.get('colId', []))),
        }
        return node, metadata

    def _update_query_state(self, node_meta):
        if node_meta['node_type'] in JOINTYPE:  # Merge node metadata into query-wide statistics.
            self._num_joins += 1

        table = node_meta['table']  # Record tables referenced by this query.
        if table and table != 'NA':
            self._curr_tables_used.add(table)

        self._num_filters += int(node_meta['filter_count'])  # Accumulate filter counts and columns referenced by joins and filters.
        self._curr_cols_used.update(self._col_ids_to_names(node_meta['joinids']))
        self._curr_cols_used.update(self._col_ids_to_names(node_meta['filter_col_ids']))

    def _update_param_state(self, node_meta):
        self._curr_cols_used.update(self._col_ids_to_names(node_meta['filter_col_ids']))  # Dimension discovery needs filter columns but not encoded joins.

    def _traverse_plan_bfs(self, plan, *, root_pos, unary_child_pos, multi_child_pos_fn, node_builder):
        root, root_meta = node_builder(plan, pos=root_pos, parentAlias=None)  # Traverse the plan with BFS and return the root, visit records and parent-child edges.
        records = []
        adj_list = []
        NodeList = deque()
        NodeList.append((root, root_meta, plan, '0', 0))
        next_id = 1

        while NodeList:
            parentNode, parentMeta, parentPlan, parentEncod, idx = NodeList.popleft()  # Record nodes in BFS order for later feature construction and statistics.
            records.append((parentNode, parentMeta, parentPlan, parentEncod, idx))

            subplans = parentPlan.get('Plans', [])
            subPlanNum = len(subplans)
            if subPlanNum == 1:
                subplan = subplans[0]  # Use dedicated positional encoding for a single child.
                node, node_meta = node_builder(subplan, pos=unary_child_pos, parentAlias=parentNode.alias)
                subEncod = parentEncod + '0'
                node.parent = parentNode
                parentNode.addChild(node)
                NodeList.append((node, node_meta, subplan, subEncod, next_id))
                adj_list.append((idx, next_id))
                next_id += 1
            elif subPlanNum > 1:
                for child_idx in range(subPlanNum - 1, -1, -1):  # Enqueue multiple children in reverse order to preserve BFS encoding order.
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
        node, _ = self._build_plan_node(plannode, pos=pos, parentAlias=parentAlias, encode_join=True, build_feature=True)  # Keep the single-node interface while sharing node-construction logic.
        return node
    
    def traversePlan(self, plan, hint, query_id): 
        # Convert a JSON plan into a tree.
        # pos:{3:'root', 0:'left', 1:'right', 2:'internal-no-brother'}
        features = []
        heights = []
        root, records, adj_list = self._traverse_plan_bfs(  # Build the BFS traversal first, then consume its results.
            plan,
            root_pos=3,
            unary_child_pos=2,
            multi_child_pos_fn=lambda child_idx: child_idx,
            node_builder=lambda subplan, pos, parentAlias: self._build_plan_node(
                subplan, pos=pos, parentAlias=parentAlias, encode_join=True, build_feature=True
            ),
        )
        self._curr_filter_dict = self._merge_query_filter_dict(records)  # Collect query-level histograms from filters across the tree, not just the root.

        for parentNode, parentMeta, parentPlan, parentEncod, idx in records:
            self._update_query_state(parentMeta)  # Aggregate query statistics during plan traversal, not node construction.
            features.append(parentNode.feature)
            heights.append(len(parentEncod))
            if parentMeta['node_type'] in JOINTYPE:
                hint['join operator'][parentEncod] = parentPlan['Node Type']  # Record hierarchical join-operator encodings for join nodes.
                hint['join node index'][parentEncod] = idx
            elif parentMeta['is_scan'] and parentNode.alias is not None:
                hint['scan table'][parentEncod] = [parentNode.alias]  # Record hierarchical table encodings for scan nodes.

        pc_dict = defaultdict(list)  # Build a parent-to-children map from BFS edges.
        for parent, child in adj_list:
            pc_dict[parent].append(child)
        return {
            'features': torch.FloatTensor(np.array(features)),
            'heights':  torch.LongTensor(heights),
            'pc_dict':  pc_dict}
    
    def traverseNode4GetParam(self, plannode,  pos = None, parentAlias = None):
        node, _ = self._build_plan_node(plannode, pos=pos, parentAlias=parentAlias, encode_join=False, build_feature=False)  # Dimension discovery needs node structure and filters, not features or joins.
        return node
    
    def traversePlan4GetParam(self, plan): 
        # Traverse a JSON plan to discover model dimensions.
        heights = []
        _, records, _ = self._traverse_plan_bfs(  # Reuse BFS with positional encoding for dimension discovery.
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
            nodeNum += 1  # Derive join, filter, height and node-count limits from traversal records.
            self._update_param_state(parentMeta)
            heights.append(len(parentEncod))
            if parentMeta['node_type'] in JOINTYPE:
                joinNum += 1
            filterNum = max(filterNum, int(parentMeta['filter_count']))
            subPlanNum = len(parentPlan.get('Plans', []))
            if subPlanNum > max_posNum:
                max_posNum = subPlanNum

        return [joinNum, filterNum, max_posNum, max(heights), nodeNum]

    def _set_hint_directives(self, hint_set):
        gucs = list(getattr(self.config, 'set_hint_gucs', ()))
        set_size = int(getattr(self.config, 'set_size', 2 ** len(gucs)))
        hint_set = int(hint_set)
        if len(gucs) == 0 or hint_set < 0 or hint_set >= set_size:
            raise ValueError(f'Invalid SET hint: {hint_set}')
        bits = [int(bit) for bit in bin(hint_set)[2:].zfill(len(gucs))]
        return [f'Set({name} {"on" if enabled else "off"})' for name, enabled in zip(gucs, bits)]

    def to_exechint(self, hintdict):
        # Set(...) keeps GUC combinations local to this statement. The first
        # action intentionally lets PostgreSQL choose
        # a fresh order; subsequent swap actions add Leading(...).
        hint_lines = []
        hint_set = hintdict.get('hint set')
        if hint_set is not None:
            hint_lines.extend(self._set_hint_directives(hint_set))
            if not hintdict.get('force join order', True):
                return '/*+\n' + '\n'.join(hint_lines) + '\n*/\n'

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
            if hint_set is None:
                join_hint.append(hintdict['join operator'][i] + '('+ join_hint_tmp + ')')

            order_hint_tmp = '(' + ' '.join(order_hint[i_struct:i_struct + 2]) + ')'
            order_hint = order_hint[:i_struct] + [order_hint_tmp] + order_hint[i_struct + 2:]
        join_hint.reverse()
        hint_lines.append(leading_hint + '(' + order_hint[0] + ')')
        hint_lines.extend(join_hint)
        return '/*+\n' + '\n'.join(hint_lines) + '\n*/\n'

@ray.remote
class RemotePlanHelper():
    # Remote plan helper for concurrent plan-feature extraction.
    def __init__(self,globalConfig):
        self.config = globalConfig
        self.planhelper = PlanHelper(globalConfig)
    def GetFeature(self,hint,sql,toextract, database,query_id = None,is_encode = True,source = 'DynaHint', need_card_label = True, cost_plan_timeout=None):
        # Get plan features.
        return self.planhelper.get_feature(
            hint,
            sql,
            toextract,
            database,
            query_id = query_id,
            source = source,
            is_encode = is_encode,
            need_card_label = need_card_label,
            cost_plan_timeout = cost_plan_timeout,
        ) 
    def GetLatency(self, hint, sql, database, query_id, timeout=None, use_buffer=True, step=0,
                   collect_cardinality=True, measurement_phase='experience'):
        # Get query execution latency.
        if timeout == None:
            timeout = self.config.max_time_out
        return self.planhelper.getLatency(
            hint,
            sql,
            database,
            query_id,
            timeout=timeout,
            use_buffer=use_buffer,
            step=step,
            collect_cardinality=collect_cardinality,
            measurement_phase=measurement_phase,
        )
    def GetCardinality(self, database, query_id, sql):
        # Get cardinality to warm its cache before baseline collection.
        return self.planhelper.dbrunner.getCardinality(database, query_id, sql=sql)

    def GetNearestTAGInput(self, sql, database, query_id, timeout=None):
        from collection.nearest_tag_context import collect_context
        from shared.nearest_latency_gate import build_signature
        prepared = self.planhelper.get_feature('', sql, True, database, query_id=query_id,
                                                need_card_label=False, cost_plan_timeout=timeout)
        cache = getattr(self, '_nearest_tag_context', {})
        if database not in cache:
            cache[database] = collect_context(self.planhelper.dbrunner.cur[database], self.config.mode)
        self._nearest_tag_context = cache
        tables, columns, dates = cache[database]
        state = build_signature(self.config.mode, sql, prepared[3], tables, columns, dates)
        return state, prepared
    def SaveEncoding(self,path):
        # Save encoding data.
        self.planhelper.encoding.save_to_file(path)
    def FreezeEncoding(self):
        self.planhelper.encoding.freeze()
    def GetPGLatencyBuffer(self):
        # Get the latency buffer.
        return self.planhelper.getPGLatencyBuffer()
    def GetTableNum(self):
        # Get the table count.
        return self.planhelper.get_table_num()
    def GetExechint(self,hintdict):
        # Get execution hints.
        return self.planhelper.to_exechint(hintdict)
    def GetSortedQueryID(self):
        # Get sorted query IDs.
        hintNum = self.planhelper.get_hintNum()
        sorted_keys = sorted(hintNum, key=hintNum.get)
        return sorted_keys
    def GetParam(self,hint,sql,toextract, database,query_id = None):
        # Get parameters.
        return self.planhelper.get_param(hint,sql, toextract, database,query_id = query_id)

if __name__ == "__main__":
    import os
    import random
    from shared.config import Config
    config = Config()   
    config.ConfirmPath()
    # Set the training-workload path.
    train_path = 'experiment/JOB/train'
    # List files in the directory.
    sql_files = [f for f in os.listdir(train_path) if f.endswith('.sql')]
    # Select a random SQL file.
    chosen_sql_file = random.choice(sql_files)
    full_sql_path = os.path.join(train_path, chosen_sql_file)
    with open(full_sql_path, 'r') as f:
        sql = f.read()
    
    planhelper  = PlanHelper(config)
    # Example feature-extraction call; adjust arguments to the environment.
    hint = ''  # Replace with an environment-specific value.
    toextract = None  # Replace with an environment-specific value.
    database = "imdb"  # Example database name.
    query_id = None
    feature = planhelper.get_feature(hint, sql, toextract, database, query_id=query_id)
