from operator import index
import psycopg2
from psycopg2 import sql as pg_sql
# from config import Config
import os, shutil
import json
import pandas as pd
import time
import numpy as np
from datetime import datetime
import hashlib
import re
PGDATATYPE = ['smallint','integer','bigint','decimal','numeric','real',
                'double precision','smallserial','serial','bigserial']
CARDINALITY_EXTRACT_MODES = {'count_rewrite', 'aggregate_input'}
AGGREGATE_NODE_TYPES = {'Aggregate', 'HashAggregate', 'GroupAggregate'}
LATENCY_TIMEOUT_EXCEPTIONS = (psycopg2.errors.QueryCanceled,)
LATENCY_CONNECTION_EXCEPTIONS = (psycopg2.OperationalError, psycopg2.InterfaceError)
LATENCY_RESOURCE_ERROR_PATTERNS = (
    "could not resize shared memory segment",
    "could not map anonymous shared memory",
    "invalid dsa memory alloc request size",
    "out of memory",
    "no space left on device",
)
class PGHelper:
    def __init__(self, globalConfig):
        self.con = {}
        self.cur = {}
        self.config = globalConfig
        self.databases = globalConfig.databases
        for database in self.databases:
            con = psycopg2.connect(database=database, user=globalConfig.user,
                                    password=globalConfig.password, host=globalConfig.ip,
                                    port=globalConfig.port)
            cur = con.cursor()
            cur.execute("SET geqo=off;")
            cur.execute("load 'pg_hint_plan';")
            self.con[database]=con
            self.cur[database]=cur
        self.latencyBuffer = {}
        self.cardinalityCache = {}
        self._load_cardinality_cache()
        self._buffer_load_cardinality_repairs = 0
        if os.path.exists(self.config.latency_buffer_path):
            print('Loading buffer...')
            tmp_buffer_file = open(self.config.latency_buffer_path,"r")
            lines = tmp_buffer_file.readlines()
            tmp_buffer_file.close()
            for line in lines:
                data = json.loads(line)
                self._load_buffer_record(data)
            self.buffer_file = open(self.config.latency_buffer_path,"a")
            print(f'Loaded {len(lines)} buffer records')
            if self._buffer_load_cardinality_repairs > 0:
                print(f'Repaired {self._buffer_load_cardinality_repairs} cardinality records')
        else:
            if not os.path.exists(os.path.dirname(self.config.latency_buffer_path)):
                os.makedirs(os.path.dirname(self.config.latency_buffer_path))
            if os.path.exists(self.config.pg_latency):
                print('Copying buffer file...')
                shutil.copy(self.config.pg_latency, self.config.latency_buffer_path)
                tmp_buffer_file = open(self.config.latency_buffer_path,"r")
                lines = tmp_buffer_file.readlines()
                tmp_buffer_file.close()
                for line in lines:
                    data = json.loads(line)
                    self._load_buffer_record(data)
                self.buffer_file = open(self.config.latency_buffer_path,"a")
                print(f'Loaded {len(lines)} buffer records')
            else:
                self.buffer_file = open(self.config.latency_buffer_path,"w")
                print('Buffer file created')
        self.getTables()
        # ===== DB meta cache (table/column statistics) =====
        self.db_meta_cache_version = 2
        self.db_meta = {}
        self.encoding = None
        self.db_meta_initialized = False

    def _is_valid_cardinality(self, cardinality):
        if cardinality is None:
            return False
        try:
            if pd.isna(cardinality):
                return False
        except Exception:
            pass
        return float(cardinality) >= 0

    def _cardinality_cache_path(self):
        return getattr(self.config, 'cardinality_cache_path', None)

    def _get_cardinality_extract_mode(self):
        mode = getattr(self.config, 'cardinality_extract_mode', 'count_rewrite')
        if mode not in CARDINALITY_EXTRACT_MODES:
            print(f"[Warning] Unknown cardinality_extract_mode={mode}, fallback to count_rewrite")
            return 'count_rewrite'
        return mode

    def _sql_hash(self, sql):
        normalized_sql = ' '.join(str(sql).strip().rstrip(';').lower().split())
        return hashlib.md5(normalized_sql.encode('utf-8')).hexdigest()

    def _cardinality_cache_key(self, database, queryid, sql_hash, mode=None):
        mode = self._get_cardinality_extract_mode() if mode is None else mode
        return '|'.join([str(database), str(queryid), str(sql_hash), str(mode)])

    def _normalize_cardinality_value(self, cardinality):
        if not self._is_valid_cardinality(cardinality):
            return -1
        cardinality = float(cardinality)
        return int(cardinality) if cardinality.is_integer() else cardinality

    def _load_cardinality_cache(self):
        self.cardinalityCache = {}
        cache_path = self._cardinality_cache_path()
        if not cache_path or not os.path.exists(cache_path):
            return
        loaded = 0
        with open(cache_path, 'r') as cache_file:
            for line in cache_file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    cardinality = self._normalize_cardinality_value(record.get('cardinality', -1))
                    if not self._is_valid_cardinality(cardinality):
                        continue
                    mode = record.get('cardinality_mode', 'count_rewrite')
                    key = self._cardinality_cache_key(record['database'], record['query_id'], record['sql_hash'], mode=mode)
                    record['cardinality'] = cardinality
                    record['cardinality_mode'] = mode
                    self.cardinalityCache[key] = record
                    loaded += 1
                except Exception as exc:
                    print(f"[Warning] Skip malformed cardinality cache line: {exc}")
        if loaded > 0:
            print(f'Loaded {loaded} cardinality cache records')

    def _append_cardinality_cache(self, database, queryid, sql_hash, cardinality, source):
        cardinality = self._normalize_cardinality_value(cardinality)
        if not self._is_valid_cardinality(cardinality):
            return
        mode = self._get_cardinality_extract_mode()
        cache_path = self._cardinality_cache_path()
        if not cache_path:
            return
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        record = {
            'database': database,
            'query_id': queryid,
            'sql_hash': sql_hash,
            'cardinality': cardinality,
            'cardinality_mode': mode,
            'source': source,
            'updated_at': datetime.now().isoformat(timespec='seconds'),
        }
        key = self._cardinality_cache_key(database, queryid, sql_hash, mode=mode)
        self.cardinalityCache[key] = record
        with open(cache_path, 'a') as cache_file:
            cache_file.write(json.dumps(record, ensure_ascii=False) + '\n')
            cache_file.flush()

    def _get_cardinality_cache_record(self, database, queryid, sql):
        if sql is None:
            return None, None
        sql_hash = self._sql_hash(sql)
        mode = self._get_cardinality_extract_mode()
        return self.cardinalityCache.get(self._cardinality_cache_key(database, queryid, sql_hash, mode=mode)), sql_hash

    def _rewrite_sql_for_cardinality_explain(self, sql):
        stripped_sql = str(sql).strip().rstrip(';')
        match = re.match(r'^\s*select\s+count\s*\(\s*\*\s*\)\s+from\s+(.+?)\s*$', stripped_sql, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return 'SELECT 1 FROM ' + match.group(1)
        return stripped_sql

    def _build_cardinality_explain_sql(self, sql, mode=None):
        mode = self._get_cardinality_extract_mode() if mode is None else mode
        stripped_sql = str(sql).strip().rstrip(';')
        if mode == 'aggregate_input':
            return stripped_sql
        return self._rewrite_sql_for_cardinality_explain(stripped_sql)

    def _find_first_aggregate_node(self, plan):
        if not isinstance(plan, dict):
            return None
        if plan.get('Node Type') in AGGREGATE_NODE_TYPES:
            return plan
        for child in plan.get('Plans', []) or []:
            found = self._find_first_aggregate_node(child)
            if found is not None:
                return found
        return None

    def _extract_plan_actual_rows(self, plan):
        if not isinstance(plan, dict):
            return -1
        rows = self._normalize_cardinality_value(plan.get('Actual Rows', -1))
        loops = plan.get('Actual Loops', 1)
        try:
            loops = float(loops)
        except Exception:
            loops = 1
        if self._is_valid_cardinality(rows) and loops > 1:
            rows = float(rows) * loops
        return self._normalize_cardinality_value(rows)

    def _is_parallel_wrapper_node(self, plan):
        if not isinstance(plan, dict):
            return False
        node_type = plan.get('Node Type')
        if node_type in ('Gather', 'Gather Merge'):
            return True
        if node_type in AGGREGATE_NODE_TYPES:
            return True
        return False

    def _extract_real_aggregate_input_rows(self, plan):
        if not isinstance(plan, dict):
            return -1
        children = plan.get('Plans', []) or []
        if len(children) == 0:
            return self._extract_plan_actual_rows(plan)
        if self._is_parallel_wrapper_node(plan):
            for child in children:
                rows = self._extract_real_aggregate_input_rows(child)
                if self._is_valid_cardinality(rows):
                    return rows
        return self._extract_plan_actual_rows(plan)

    def _extract_aggregate_input_rows_from_plan(self, plan):
        aggregate_plan = self._find_first_aggregate_node(plan)
        if aggregate_plan is None:
            return -1
        children = aggregate_plan.get('Plans', []) or []
        if len(children) == 0:
            return -1
        return self._extract_real_aggregate_input_rows(children[0])

    def _extract_actual_rows_from_explain_json(self, explain_json, mode=None):
        mode = self._get_cardinality_extract_mode() if mode is None else mode
        plan_json = explain_json[0] if isinstance(explain_json, list) else explain_json
        plan = plan_json.get('Plan', plan_json)
        if mode == 'aggregate_input':
            aggregate_input_rows = self._extract_aggregate_input_rows_from_plan(plan)
            if self._is_valid_cardinality(aggregate_input_rows):
                return aggregate_input_rows
        return self._extract_plan_actual_rows(plan)

    def _get_cardinality_by_explain_analyze(self, database, queryid, sql, hint='', timeout=None):
        if sql is None:
            return -1
        timeout = self.config.max_time_out if timeout is None else timeout
        mode = self._get_cardinality_extract_mode()
        explain_sql = self._build_cardinality_explain_sql(sql, mode=mode)
        try:
            self.cur[database].execute("BEGIN;")
            self.cur[database].execute("SET LOCAL statement_timeout = " + str(int(timeout)) + ";")
            self.cur[database].execute("SET LOCAL max_parallel_workers_per_gather = 0;")
            self.cur[database].execute(hint + "EXPLAIN (ANALYZE, FORMAT JSON) " + explain_sql)
            rows = self.cur[database].fetchall()
            self.cur[database].execute("COMMIT;")
            if len(rows) == 0:
                return -1
            return self._extract_actual_rows_from_explain_json(rows[0][0], mode=mode)
        except KeyboardInterrupt:
            try:
                self.cur[database].execute("ROLLBACK;")
            except Exception:
                pass
            raise
        except Exception as exc:
            print(f"[Warning] EXPLAIN ANALYZE cardinality failed: database={database}, queryid={queryid}, error={exc}")
            try:
                self.cur[database].execute("ROLLBACK;")
            except Exception:
                pass
            return -1

    def _sync_cardinality_from_cache_record(self, database, queryid, entry, record):
        cardinality = self._normalize_cardinality_value(record.get('cardinality', -1))
        if not self._is_valid_cardinality(cardinality):
            return entry.get('cardinality', -1)
        return self._reconcile_cardinality(database, queryid, entry, cardinality, '', 'cardinality_cache', persist=False)

    def _fetch_and_cache_explain_cardinality(self, database, queryid, sql, entry, hint='', timeout=None):
        record, sql_hash = self._get_cardinality_cache_record(database, queryid, sql)
        if record is not None:
            return self._sync_cardinality_from_cache_record(database, queryid, entry, record)
        if sql_hash is None:
            return entry.get('cardinality', -1)
        observed_cardinality = self._get_cardinality_by_explain_analyze(database, queryid, sql, hint=hint, timeout=timeout)
        if not self._is_valid_cardinality(observed_cardinality):
            return entry.get('cardinality', -1)
        cached_cardinality = entry.get('cardinality', -1)
        source = 'explain_analyze'
        if self._is_valid_cardinality(cached_cardinality) and cached_cardinality != observed_cardinality:
            source = 'explain_analyze_repair'
            print(
                f"[Warning] Cardinality cache mismatch: db={database}, query={queryid}, "
                f"cached={cached_cardinality}, explain_analyze={observed_cardinality}, source=count_cache"
            )
        self._append_cardinality_cache(database, queryid, sql_hash, observed_cardinality, source)
        return self._reconcile_cardinality(database, queryid, entry, observed_cardinality, hint, source, persist=False)

    def _record_cardinality_evidence(self, entry, cardinality):
        if not self._is_valid_cardinality(cardinality):
            return None
        evidence = entry.setdefault('_cardinality_evidence', {})
        card_key = int(cardinality) if float(cardinality).is_integer() else float(cardinality)
        evidence[card_key] = evidence.get(card_key, 0) + 1
        return card_key

    def _select_canonical_cardinality(self, entry):
        evidence = entry.get('_cardinality_evidence', {})
        if len(evidence) == 0:
            return entry.get('cardinality', -1)
        return max(evidence.items(), key=lambda item: (item[1], item[0] != 0, item[0]))[0]

    def _persist_cardinality_only(self, database, queryid, cardinality):
        if not self._is_valid_cardinality(cardinality):
            return
        return

    def _reconcile_cardinality(self, database, queryid, entry, observed_cardinality, hint, source, persist=True):
        cached_cardinality = entry.get('cardinality', -1)
        if self._is_valid_cardinality(observed_cardinality):
            entry['cardinality_mode'] = self._get_cardinality_extract_mode()
            observed_cardinality = self._record_cardinality_evidence(entry, observed_cardinality)
            if not self._is_valid_cardinality(cached_cardinality):
                if cached_cardinality != observed_cardinality:
                    if source == 'buffer_load':
                        self._buffer_load_cardinality_repairs += 1
                    elif source not in ('lazy_cardinality', 'DynaHint', 'update_Median', 'explain_analyze', 'explain_analyze_repair', 'cardinality_cache'):
                        print(f"[Warning] Cardinality repaired: database={database}, queryid={queryid}, previous={cached_cardinality}, repaired={observed_cardinality}, source={source}")
                entry['cardinality'] = observed_cardinality
                if persist:
                    self._persist_cardinality_only(database, queryid, observed_cardinality)
            elif cached_cardinality != observed_cardinality:
                if source == 'buffer_load':
                    self._buffer_load_cardinality_repairs += 1
                elif source not in ('explain_analyze', 'explain_analyze_repair', 'cardinality_cache'):
                    print(f"[Warning] Cardinality updated: database={database}, queryid={queryid}, previous={cached_cardinality}, updated={observed_cardinality}, source={source}, hint={hint[:120] if isinstance(hint, str) else hint}")
                entry['cardinality'] = observed_cardinality
                if persist:
                    self._persist_cardinality_only(database, queryid, observed_cardinality)
        return entry.get('cardinality', cached_cardinality)

    def _load_buffer_record(self, data):
        self.latencyBuffer.setdefault(data[0], {})
        self.latencyBuffer[data[0]].setdefault(data[1], {})
        entry = self.latencyBuffer[data[0]][data[1]]
        entry['cardinality_mode'] = data[6] if len(data) > 6 else entry.get('cardinality_mode', 'count_rewrite')
        if 'cardinality' not in entry:
            entry['cardinality'] = data[2]
        else:
            self._reconcile_cardinality(data[0], data[1], entry, data[2], '', 'buffer_load', persist=False)

        hint = data[3] if len(data) > 3 else None
        latency_record = data[4] if len(data) > 4 else None
        if hint is None:
            return
        if hint == '' and latency_record is None:
            return
        entry.setdefault(hint, latency_record)

    def _rollback_latency_transaction(self, database):
        try:
            self.con[database].rollback()
        except Exception:
            try:
                self.con[database].commit()
            except Exception:
                pass

    def _classify_latency_exception(self, exc):
        if isinstance(exc, LATENCY_TIMEOUT_EXCEPTIONS):
            return 'timeout'
        message = str(exc).lower()
        if any(pattern in message for pattern in LATENCY_RESOURCE_ERROR_PATTERNS):
            return 'resource_error'
        if isinstance(exc, LATENCY_CONNECTION_EXCEPTIONS):
            return 'connection_error'
        if isinstance(exc, psycopg2.Error):
            return 'sql_error'
        return 'unknown_error'

    def _enable_latency_exception_classification(self):
        return bool(getattr(self.config, "enable_pg_latency_exception_classification", False))

    def getTables(self):
        database = self.databases[0]
        self.cur[database].execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
        self.table_names = [name[0] for name in self.cur[database].fetchall()]
        self.tablenum    = len(self.table_names)

    def getLatency(self, hint, sql, database, queryid, timeout, source, use_buffer,step, require_valid_cardinality=False):
        # print(queryid, hint)
        query_timeout = timeout
        if database in self.latencyBuffer and use_buffer is True:
            if queryid in self.latencyBuffer[database]:
                if hint in self.latencyBuffer[database][queryid]:
                    cached_cardinality = self.latencyBuffer[database][queryid].get('cardinality', -1)
                    if sql is not None and not self._is_valid_cardinality(cached_cardinality):
                        cached_cardinality = self._fetch_and_cache_explain_cardinality(
                            database,
                            queryid,
                            sql,
                            self.latencyBuffer[database][queryid],
                            hint=hint,
                            timeout=timeout,
                        )
                    if (not require_valid_cardinality) or self._is_valid_cardinality(cached_cardinality):
                        return self.latencyBuffer[database][queryid][hint], False, cached_cardinality
        self.latencyBuffer.setdefault(database, {})
        self.latencyBuffer[database].setdefault(queryid, {})
        self.latencyBuffer[database][queryid].setdefault('cardinality', -1)
        self.latencyBuffer[database][queryid].setdefault('cardinality_mode', self._get_cardinality_extract_mode())
        exectime = time.time()
        try:
            self.cur[database].execute("SET statement_timeout = " + str(int(timeout)) + ";")
            self.cur[database].execute(hint + sql)
            self.cur[database].fetchone()
            timeout = False
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self._rollback_latency_transaction(database)
            if not self._enable_latency_exception_classification():
                timeout = True
            else:
                error_type = self._classify_latency_exception(exc)
                if error_type in ('timeout', 'resource_error'):
                    if error_type == 'resource_error':
                        print(
                            f"[Warning] getLatency resource_error treated as timeout: database={database}, "
                            f"queryid={queryid}, source={source}, hint={hint[:120] if isinstance(hint, str) else hint}, error={exc}"
                        )
                    timeout = True
                else:
                    print(
                        f"[Error] getLatency failed: type={error_type}, database={database}, "
                        f"queryid={queryid}, source={source}, hint={hint[:120] if isinstance(hint, str) else hint}, error={exc}"
                    )
                    raise
        exectime = round((time.time() - exectime) * 1000, 3)
        entry = self.latencyBuffer[database][queryid]
        if timeout is True:
            cardinality = entry.get('cardinality', -1)
            if entry.get('cardinality_mode', 'count_rewrite') != self._get_cardinality_extract_mode():
                cardinality = -1
        else:
            cardinality = self._fetch_and_cache_explain_cardinality(
                database,
                queryid,
                sql,
                entry,
                hint=hint,
                timeout=query_timeout,
            )
        latency_timeout = [exectime, timeout]
        entry[hint] = latency_timeout
        canonical_cardinality = self._reconcile_cardinality(
            database,
            queryid,
            entry,
            cardinality,
            hint,
            source,
            persist=(timeout is not True),
        )
        if timeout is True:
            cardinality = canonical_cardinality
        elif self._is_valid_cardinality(canonical_cardinality):
            cardinality = canonical_cardinality
        if use_buffer is True:
            self.buffer_file.write(json.dumps([database, queryid, cardinality, hint, latency_timeout,step,self._get_cardinality_extract_mode()])+"\n")
            self.buffer_file.flush()
        return latency_timeout, True, cardinality
    
    def tryGetLatency(self,hint,database,query_id):
        try:
            lat_timeout = self.latencyBuffer[database][query_id][hint]
            if lat_timeout[1]:
                return None
            else:
                return lat_timeout[0]
        except:
            return None
    
    def getCardinality(self, database, query_id, sql=None, fallback_to_db=True):
        if sql is None:
            raise ValueError("getCardinality requires sql; cardinality cache key depends on sql_hash")
        self.latencyBuffer.setdefault(database, {})
        self.latencyBuffer[database].setdefault(query_id, {})
        entry = self.latencyBuffer[database][query_id]

        record, _ = self._get_cardinality_cache_record(database, query_id, sql)
        if record is not None:
            return self._sync_cardinality_from_cache_record(database, query_id, entry, record)
        if not fallback_to_db:
            return -1
        return self._fetch_and_cache_explain_cardinality(
            database,
            query_id,
            sql,
            entry,
            hint='',
            timeout=self.config.max_time_out,
        )
	        
    def getCostPlanJson(self, hint, sql, source, database, query_id = None):
        import time
        startTime = time.time()
        try:
            self.cur[database].execute("SET statement_timeout = " + str(int(self.config.max_time_out)) + ";")
            self.cur[database].execute(hint + "explain (COSTS, FORMAT JSON) " + sql)
            rows = self.cur[database].fetchall()
        except:
            print(database+'|'+query_id)
            print(hint)
            print(sql)
            raise
        plan_json = rows[0][0][0]
        plan_json['Planning Time'] = time.time() - startTime
        return plan_json

    def get_minLatency(self):
        minLatency = {}
        for queryid in self.latencyBuffer:
            minlat = self.config.max_time_out
            hint2send = ''
            for hint in self.latencyBuffer[queryid]:
                if self.latencyBuffer[queryid][hint][0] < minlat:
                    minlat = self.latencyBuffer[queryid][hint][0]
                    hint2send  = hint
            minLatency[queryid] = [minlat,hint2send]
        return minLatency

    
    def get_table_num(self):
        return self.tablenum
    
    def get_min_max_values(self,table_name, column_name, database):
        self.cur[database].execute(f"SELECT MIN({column_name}), MAX({column_name}) FROM {table_name};")
        min_val, max_val = self.cur[database].fetchone()
        if min_val != None and max_val != None:
            max_val = float(max_val)
            min_val = float(min_val)
        return min_val,max_val
    
    def get_column_data_properties(self):
        column_data_properties = {}
        for database in self.databases:
            column_data_properties[database] = {}
            for table_name in self.table_names:
                self.cur[database].execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{table_name}';")
                for column_name, data_type in self.cur[database].fetchall():
                    if data_type in PGDATATYPE and column_name != 'cc_closed_date_sk':
                        min_val, max_val = self.get_min_max_values(table_name, column_name,database)
                        column_data_properties[database][table_name + '.' + column_name] = (min_val, max_val)
        return column_data_properties

    def bind_encoding(self, encoding):
        self.encoding = encoding

    def get_exact_table_row_count(self, database, table_name):
        query = pg_sql.SQL("SELECT COUNT(*) FROM {}.{};").format(
            pg_sql.Identifier('public'),
            pg_sql.Identifier(table_name),
        )
        self.cur[database].execute(query)
        row_count = self.cur[database].fetchone()[0]
        return int(row_count)

    def ensure_exact_table_row_counts(self, encoding):
        changed = False
        encoding.table_row_counts = getattr(encoding, "table_row_counts", {}) or {}
        for database in self.databases:
            encoding.table_row_counts.setdefault(database, {})
            for table_name in self.table_names:
                if table_name in encoding.table_row_counts[database]:
                    continue
                try:
                    encoding.table_row_counts[database][table_name] = self.get_exact_table_row_count(database, table_name)
                    changed = True
                except Exception as e:
                    print(f"[Warning] Exact table row count fetch failed: database={database}, table={table_name}, err={e}")
        return changed

    def initialize_db_meta(self, encoding=None):
        if encoding is not None:
            self.bind_encoding(encoding)
        if not getattr(self.config, "enable_db_meta", False):
            return
        if self.encoding is None:
            print("[Warning] db_meta init skipped: encoding is not bound yet.")
            return
        self._init_db_meta_cache()
        self.db_meta_initialized = True
    
    # ====================== DB meta features ======================

    def _init_db_meta_cache(self):
        cache_dir = getattr(self.config, "db_meta_cache_dir", None)
        if cache_dir is None:
            return
        os.makedirs(cache_dir, exist_ok=True)
        for db in self.databases:
            try:
                if self._load_db_meta_from_cache(db, cache_dir):
                    continue
                self._refresh_db_meta(db, cache_dir)
            except Exception as e:
                print(f"[Warning] db_meta init failed: db={db}, err={e}")

    def _cache_path(self, database: str, cache_dir: str) -> str:
        safe = database.replace('/', '_')
        return os.path.join(cache_dir, f"{safe}_dbmeta.npz")

    def _load_db_meta_from_cache(self, database: str, cache_dir: str) -> bool:
        path = self._cache_path(database, cache_dir)
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path, allow_pickle=True)
            cache_version = int(data["db_meta_cache_version"]) if "db_meta_cache_version" in data else 1
            if cache_version != self.db_meta_cache_version:
                return False
            self.db_meta[database] = {
                "table_names": data["table_names"].tolist(),
                "table_feat": data["table_feat"].astype(np.float32, copy=False),
                "global_feat": data["global_feat"].astype(np.float32, copy=False),
                "col_stats": data["col_stats"].item(),  # dict
                "created_at": str(data["created_at"]),
                "db_meta_cache_version": cache_version,
            }
            return True
        except Exception:
            return False

    def _refresh_db_meta(self, database: str, cache_dir: str):
        table_df = self._fetch_table_stats(database)
        col_df = self._fetch_column_stats(database)
        exact_row_counts = {}
        if self.encoding is not None:
            exact_row_counts = getattr(self.encoding, "table_row_counts", {}).get(database, {})

        table_names = list(self.table_names)
        name2row = {r["table_name"]: r for _, r in table_df.iterrows()}

        # [log1p(n_live), log1p(total_bytes_mb), log1p(relpages), dead_ratio, tanh(age_days/30), has_stats]
        d_table = int(getattr(self.config, "db_table_feat_dim", 6))
        table_feat = np.zeros((len(table_names), d_table), dtype=np.float32)

        for i, t in enumerate(table_names):
            if t not in name2row:
                continue
            r = name2row[t]
            exact_row_count = exact_row_counts.get(t, None)
            if exact_row_count is None:
                try:
                    exact_row_count = self.get_exact_table_row_count(database, t)
                    if self.encoding is not None:
                        self.encoding.table_row_counts.setdefault(database, {})[t] = int(exact_row_count)
                except Exception as e:
                    print(f"[Warning] Exact table row count fallback failed: database={database}, table={t}, err={e}")
                    exact_row_count = float(r.get("n_live_tup", 0.0) or 0.0)
            n_live = float(exact_row_count or 0.0)
            n_dead = float(r.get("n_dead_tup", 0.0) or 0.0)
            total_bytes = float(r.get("total_bytes", 0.0) or 0.0)
            relpages = float(r.get("relpages", 0.0) or 0.0)
            age_days = float(r.get("stats_age_days", 0.0) or 0.0)
            has_stats = float(r.get("has_stats", 0.0) or 0.0)

            dead_ratio = n_dead / (n_live + n_dead + 1.0)
            age_norm = np.tanh(age_days / 30.0)

            feats = [
                np.log1p(n_live),
                np.log1p(total_bytes / (1024.0 * 1024.0)),
                np.log1p(relpages),
                dead_ratio,
                age_norm,
                has_stats,
            ]
            table_feat[i, :min(d_table, len(feats))] = np.asarray(feats[:d_table], dtype=np.float32)

        d_global = int(getattr(self.config, "db_global_feat_dim", 4))
        total_live = float(np.sum([float(exact_row_counts.get(t, 0.0) or 0.0) for t in table_names]))
        total_bytes = float(np.sum(table_df["total_bytes"].fillna(0.0).values))
        mean_dead = float(np.mean(table_df["dead_ratio"].fillna(0.0).values)) if len(table_df) else 0.0
        mean_age = float(np.mean(np.tanh(table_df["stats_age_days"].fillna(0.0).values / 30.0))) if len(table_df) else 0.0
        global_feat = np.asarray([
            np.log1p(total_live),
            np.log1p(total_bytes / (1024.0 * 1024.0)),
            mean_dead,
            mean_age,
        ], dtype=np.float32)[:d_global]

        # [ndv, null_frac, avg_width, mcv_freq1]
        col_stats = {}
        for _, r in col_df.iterrows():
            tc = f"{r['tablename']}.{r['attname']}"
            null_frac = float(r.get("null_frac", 0.0) or 0.0)
            nd = r.get("n_distinct", 0.0)
            try:
                nd = float(nd)
            except Exception:
                nd = 0.0
            avg_width = float(r.get("avg_width", 0.0) or 0.0)
            mcv = r.get("mcv_freq1", 0.0)
            try:
                mcv = float(mcv) if mcv is not None else 0.0
            except Exception:
                mcv = 0.0
            col_stats[tc] = (nd, null_frac, avg_width, mcv)

        self.db_meta[database] = {
            "table_names": table_names,
            "table_feat": table_feat,
            "global_feat": global_feat,
            "col_stats": col_stats,
            "created_at": datetime.utcnow().isoformat(),
            "db_meta_cache_version": self.db_meta_cache_version,
        }
        print(f"db_meta refreshed: database={database}, table_feat_shape={table_feat.shape}, global_feat_shape={global_feat.shape}, col_stats_num={len(col_stats)}")
        path = self._cache_path(database, cache_dir)
        np.savez_compressed(
            path,
            table_names=np.asarray(table_names, dtype=object),
            table_feat=table_feat,
            global_feat=global_feat,
            col_stats=np.asarray(col_stats, dtype=object),
            created_at=np.asarray(self.db_meta[database]["created_at"], dtype=object),
            db_meta_cache_version=np.asarray(self.db_meta_cache_version, dtype=np.int32),
        )

    def _fetch_table_stats(self, database: str) -> pd.DataFrame:
        sql = """
        SELECT
          c.relname AS table_name,
          c.reltuples::float8 AS reltuples,
          c.relpages::float8 AS relpages,
          pg_total_relation_size(c.oid)::float8 AS total_bytes,
          st.n_live_tup::float8 AS n_live_tup,
          st.n_dead_tup::float8 AS n_dead_tup,
          (st.last_analyze IS NOT NULL OR st.last_autoanalyze IS NOT NULL) AS has_stats,
          EXTRACT(EPOCH FROM (now() - COALESCE(st.last_analyze, st.last_autoanalyze, now()))) / 86400.0 AS stats_age_days
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        LEFT JOIN pg_stat_all_tables st ON st.relid = c.oid
        WHERE n.nspname = 'public' AND c.relkind = 'r';
        """
        self.cur[database].execute(sql)
        rows = self.cur[database].fetchall()
        cols = [d[0] for d in self.cur[database].description]
        df = pd.DataFrame(rows, columns=cols)
        df["n_live_tup"] = df["n_live_tup"].fillna(0.0)
        df["n_dead_tup"] = df["n_dead_tup"].fillna(0.0)
        df["dead_ratio"] = df["n_dead_tup"] / (df["n_live_tup"] + df["n_dead_tup"] + 1.0)
        df["has_stats"] = df["has_stats"].astype(float)
        return df

    def _fetch_column_stats(self, database: str) -> pd.DataFrame:
        sql = """
        SELECT
          tablename,
          attname,
          null_frac::float8 AS null_frac,
          n_distinct::float8 AS n_distinct,
          avg_width::float8 AS avg_width,
          CASE WHEN most_common_freqs IS NULL THEN NULL ELSE most_common_freqs[1] END AS mcv_freq1
        FROM pg_stats
        WHERE schemaname='public';
        """
        self.cur[database].execute(sql)
        rows = self.cur[database].fetchall()
        cols = [d[0] for d in self.cur[database].description]
        return pd.DataFrame(rows, columns=cols)

    def get_db_meta(self, database: str):
        if not getattr(self.config, "enable_db_meta", False):
            return None, None, None
        if not self.db_meta_initialized:
            self.initialize_db_meta(self.encoding)
        if database not in self.db_meta:
            cache_dir = getattr(self.config, "db_meta_cache_dir", "./dbmeta/")
            os.makedirs(cache_dir, exist_ok=True)
            self._refresh_db_meta(database, cache_dir)
        m = self.db_meta[database]
        return m["table_feat"], m["global_feat"], m["col_stats"]
