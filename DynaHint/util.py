from config import Config
import re
import os
import numpy as np
import json
from collections import defaultdict

config = Config()
def min_swap_steps(x, y):
    if len(x) != len(y):
        return -1  

    n = len(x)
    visited = [False] * n  
    swaps = 0

    for i in range(n):
        if x[i] != y[i] and not visited[i]:
            j = i
            cycle_size = 0

            while not visited[j]:
                visited[j] = True
                j = x.index(y[j])  
                cycle_size += 1

            if cycle_size > 0:
                swaps += cycle_size - 1

    return swaps
def diff_steps(x,y):
    if len(x) != len(y):
        return -1 
    modify = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            modify += 1
    return modify
def min_steps(base, modify):
    swap = min_swap_steps(base['join order'], modify['join order'])
    diff = diff_steps(base['join operator'], modify['join operator'])
    return swap + diff
def get_label(ref,cur):
    ratio = (ref - cur) / ref
    label = 0
    for l, p in enumerate(config.splitpoint):
        if ratio >= p:
            label = config.classNum - l
            break
    return label

def get_median(L1, L2):
    sorted_l1_indices = sorted(range(len(L1)), key=lambda i: L1[i])
    sorted_l1 = [L1[i] for i in sorted_l1_indices]
    sorted_l2 = [L2[i] for i in sorted_l1_indices]
    
    length = len(sorted_l1)
    median_index = length // 2
    
    median_value_l1 = sorted_l1[median_index]
    median_value_l2 = sorted_l2[median_index]
    
    return median_value_l1, median_value_l2
def swap_dict_items(data, key1, key2):
    if key1 not in data or key2 not in data:
        raise KeyError("Index Error")
    items = list(data.items())
    index1 = next(i for i, (k, v) in enumerate(items) if k == key1)
    index2 = next(i for i, (k, v) in enumerate(items) if k == key2)
    items[index1], items[index2] = items[index2], items[index1]
    new_data = dict(items)
    return new_data


def filterDict2Hist(hist_file, filterDict, encoding):
    buckets = len(hist_file['bins'].iloc[0]) 
    empty = np.zeros(buckets - 1)
    ress = np.zeros((3, buckets-1))
    for i in range(len(filterDict['colId'])):
        if i >= 3:
            break
        if int(filterDict['dtype'][i]) != 0:
            ress[i] = empty
            continue
        colId = filterDict['colId'][i]
        col = encoding.idx2col.get(int(colId), 'NA')
        if col == 'NA':
            ress[i] = empty
            continue
        matched = hist_file.loc[hist_file['table_column'] == col, 'bins']
        if len(matched) == 0 and '.' in col:
            table_name, col_name = col.split('.', 1)
            matched = hist_file.loc[
                hist_file['table_column'].apply(
                    lambda tc: isinstance(tc, str) and tc.startswith(table_name + '.') and tc.endswith('.' + col_name)
                ),
                'bins'
            ]
        if len(matched) == 0:
            ress[i] = empty
            continue
        bins = matched.iloc[0]
        
        opId = filterDict['opId'][i]
        op = encoding.idx2op.get(int(opId), 'NA')
        
        val = filterDict['val'][i]
        mini, maxi = encoding.column_min_max_vals[col]
        val_unnorm = val * (maxi-mini) + mini
        
        left = 0
        right = len(bins)-1
        for j in range(len(bins)):
            if bins[j]<val_unnorm:
                left = j
            if bins[j]>val_unnorm:
                right = j
                break

        res = np.zeros(len(bins)-1)

        if op == '=':
            res[left:right] = 1
        elif op == '<':
            res[:left] = 1
        elif op == '<=':
            res[:right] = 1
        elif op == '>':
            res[right:] = 1
        elif op == '>=':
            res[left:] = 1
        ress[i] = res
    ress = ress.flatten()
    return ress

class SQLSubqueryExtractor:
    def __init__(self, sql_file_path, output_dir="extracted_subqueries"):
        self.sql_file_path = sql_file_path
        self.output_dir = output_dir
        self.subquery_counter = defaultdict(int)
        self.subqueries = {}
        self.original_sql = ""
        self.processed_sql = ""
        
    def extract_subqueries(self):
        with open(self.sql_file_path, 'r') as file:
            self.original_sql = file.read()
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.processed_sql, _ = self._process_subqueries(self.original_sql, "main")
        
        for subquery_id, (sql, parent, is_correlated) in self.subqueries.items():
            file_name = f"{subquery_id}.sql"
            file_path = os.path.join(self.output_dir, file_name)
            
            with open(file_path, 'w') as file:
                file.write(f"-- Source: {parent}\n")
                file.write(f"-- CorrelatedSubquery: {is_correlated}\n")
                file.write(sql)
        
        main_query_path = os.path.join(self.output_dir, "main_query.sql")
        with open(main_query_path, 'w') as file:
            file.write(self.processed_sql)
            
        print(f"Successfully extracted {len(self.subqueries)} subqueries to directory: {self.output_dir}")
        
    def _process_subqueries(self, sql, parent_name):
        subquery_pattern = re.compile(
            r'\((\s*SELECT\s+(?:(?!\);?\s*\)).)*\s*)\)', 
            re.IGNORECASE | re.DOTALL
        )
        
        processed_sql = sql
        matches = list(subquery_pattern.finditer(sql))
        
        for match in reversed(matches):
            subquery_text = match.group(1).strip()
            is_correlated = self._is_correlated_subquery(subquery_text, sql)
            
            subquery_type = "correlated" if is_correlated else "non_correlated"
            self.subquery_counter[subquery_type] += 1
            subquery_id = f"{subquery_type}_{self.subquery_counter[subquery_type]}"
            
            processed_subquery, _ = self._process_subqueries(subquery_text, subquery_id)
            
            self.subqueries[subquery_id] = (processed_subquery, parent_name, is_correlated)
            
            placeholder = f"/* SUBQUERY_{subquery_id} */"
            processed_sql = processed_sql[:match.start()] + placeholder + processed_sql[match.end():]
        
        return processed_sql, matches
    
    def _is_correlated_subquery(self, subquery, outer_sql):
        table_aliases = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', outer_sql)
        table_aliases = [alias for _, alias in table_aliases]
        
        for alias in table_aliases:
            if re.search(rf'\b{alias}\.[a-zA-Z_][a-zA-Z0-9_]*\b', subquery):
                return True
                
        return False

# ===== Added helpers: DB feature bank / writable numpy (for drift features) =====

def make_writable_np(a: np.ndarray) -> np.ndarray:
    """Ensure numpy array is writable (avoid PyTorch warning about non-writable tensors).

    PyTorch warns when converting a non-writable numpy array to a tensor:
    'The given NumPy array is not writable...'
    This typically happens if the array comes from np.frombuffer()/memmap or shares memory.
    """
    if isinstance(a, np.ndarray) and (not a.flags.writeable):
        return np.array(a, copy=True)
    return a

def load_db_feature_bank(npz_path: str, databases=None):
    """Load per-database histogram features from a .npz file."""
    if (npz_path is None) or (not os.path.exists(npz_path)):
        return None

    data = np.load(npz_path, allow_pickle=True)
    keys = list(data.keys())
    dbs = sorted({k[:-5] for k in keys if k.endswith("_hist")}) if databases is None else list(databases)
    bank = {}
    for db in dbs:
        hist_key = f"{db}_hist"
        if hist_key in data:
            bank.setdefault(db, {})['hist'] = make_writable_np(np.array(data[hist_key], copy=True)).astype(np.float32)
    return bank


def get_db_features(bank, db: str, hist_dim: int = 0):
    """Fetch a fixed-length histogram vector for a database."""
    if bank is None or db not in bank:
        return np.zeros((hist_dim,), dtype=np.float32) if hist_dim > 0 else None

    hist = bank[db].get('hist', None)
    if hist_dim <= 0:
        return None
    if hist is None:
        hist = np.zeros((hist_dim,), dtype=np.float32)
    else:
        hist = np.asarray(hist, dtype=np.float32).reshape(-1)
        if hist.shape[0] != hist_dim:
            if hist.shape[0] > hist_dim:
                hist = hist[:hist_dim]
            else:
                pad = np.zeros((hist_dim - hist.shape[0],), dtype=np.float32)
                hist = np.concatenate([hist, pad], axis=0)
    return make_writable_np(np.array(hist, copy=True))

def build_query_feat_from_tables(
    tables_used,
    table2idx: dict,
    tablenum: int,
    num_joins: int,
    num_filters: int,
    extra: np.ndarray = None,
):
    """Build a simple query-level feature vector.

    Default design:
      - table mask (one-hot over tables)  [tablenum]
      - normalized (num_joins, num_filters) [2]
      - padding (optional) to match query_feat_dim in config

    You can extend it with:
      - predicate histogram summary
      - join-graph stats
      - query template id
    """
    mask = np.zeros((tablenum,), dtype=np.float32)

    for t in (tables_used or []):
        if t in table2idx:
            idx = int(table2idx[t])
            if 0 <= idx < tablenum:
                mask[idx] = 1.0

    stats = np.array([float(num_joins), float(num_filters)], dtype=np.float32)
    # mild normalization (avoid huge values)
    stats = np.tanh(stats / 10.0)

    if extra is None:
        extra = np.zeros((0,), dtype=np.float32)
    else:
        extra = np.asarray(extra, dtype=np.float32).reshape(-1)

    feat = np.concatenate([mask, stats, extra], axis=0)
    return make_writable_np(np.array(feat, copy=True))

_hist_df_cache = {}
def load_histogram_json_as_df(database: str, hist_json_dir: str):
    """Load collector-produced <db>_histogram_string.json into a DataFrame.
    The json is a list of records with keys: table, column, table_column, bins.
    """
    key = (database, hist_json_dir)
    if key in _hist_df_cache:
        return _hist_df_cache[key]
    path = os.path.join(hist_json_dir, f"{database}_histogram_string.json")
    if not os.path.exists(path):
        _hist_df_cache[key] = None
        return None
    import pandas as pd
    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    # bins might be stored as strings in some pipelines; normalize to python lists of floats
    def _norm_bins(b):
        if isinstance(b, list):
            return [float(x) for x in b]
        if isinstance(b, str):
            s = b.strip()
            if s.startswith('[') and s.endswith(']'):
                s = s[1:-1]
            if len(s) == 0:
                return []
            return [float(x) for x in s.split(',')]
        return []
    if 'bins' in df.columns:
        df['bins'] = df['bins'].apply(_norm_bins)
    _hist_df_cache[key] = df
    return df

def build_query_hist_vector(database: str, filter_dict: dict, encoding, config):
    """Build fixed-length histogram feature for current query using util.filterDict2Hist().
    Output length is config.db_hist_dim (pad/trunc).
    """
    hist_dim = int(getattr(config, 'db_hist_dim', 0))
    if hist_dim <= 0:
        return np.zeros((0,), dtype=np.float32)
    df = load_histogram_json_as_df(database, getattr(config, 'hist_json_dir', './experiment/histogram'))
    if df is None or filter_dict is None:
        return np.zeros((hist_dim,), dtype=np.float32)
    hist_filter_dict = {'colId': [], 'opId': [], 'val': [], 'dtype': []}
    for col_id, op_id, val, dtype in zip(
        filter_dict.get('colId', []),
        filter_dict.get('opId', []),
        filter_dict.get('val', []),
        filter_dict.get('dtype', []),
    ):
        if int(dtype) != 0:
            continue
        hist_filter_dict['colId'].append(int(col_id))
        hist_filter_dict['opId'].append(int(op_id))
        hist_filter_dict['val'].append(float(val))
        hist_filter_dict['dtype'].append(int(dtype))
        if len(hist_filter_dict['colId']) >= 3:
            break
    if len(hist_filter_dict['colId']) == 0:
        return np.zeros((hist_dim,), dtype=np.float32)

    try:
        # filterDict2Hist returns (3, buckets-1)
        h3 = filterDict2Hist(df, hist_filter_dict, encoding).astype(np.float32).reshape(-1)
    except Exception:
        return np.zeros((hist_dim,), dtype=np.float32)

    if h3.shape[0] == hist_dim:
        return h3
    if h3.shape[0] > hist_dim:
        return h3[:hist_dim]
    out = np.zeros((hist_dim,), dtype=np.float32)
    out[:h3.shape[0]] = h3
    return out

if __name__ == "__main__":
    
    sql_file = "input.sql"  
    extractor = SQLSubqueryExtractor(sql_file)
    extractor.extract_subqueries()
