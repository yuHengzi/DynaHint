"""Shared utilities."""
from shared.config import Config
import re
import os
import csv
import numpy as np
import json
from collections import defaultdict

config = Config()
def min_swap_steps(x, y):
    # Get the minimum number of swaps.
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
    # Get the difference in steps.
    if len(x) != len(y):
        return -1 
    modify = 0
    for i in range(len(x)):
        if x[i] != y[i]:
            modify += 1
    return modify
def min_steps(base, modify):
    # Get the minimum step count.
    swap = min_swap_steps(base['join order'], modify['join order'])
    diff = diff_steps(base['join operator'], modify['join operator'])
    return swap + diff
def get_label(ref,cur):
    # Get the label.
    ratio = (ref - cur) / ref
    label = 0
    for l, p in enumerate(config.splitpoint):
        if ratio >= p:
            label = config.classNum - l
            break
    return label

def get_median(L1, L2):
    # Get the median.
    sorted_l1_indices = sorted(range(len(L1)), key=lambda i: L1[i])
    sorted_l1 = [L1[i] for i in sorted_l1_indices]
    sorted_l2 = [L2[i] for i in sorted_l1_indices]
    
    length = len(sorted_l1)
    median_index = length // 2
    
    median_value_l1 = sorted_l1[median_index]
    median_value_l2 = sorted_l2[median_index]
    
    return median_value_l1, median_value_l2
def swap_dict_items(data, key1, key2):
    # Swap dictionary entries.
    if key1 not in data or key2 not in data:
        raise KeyError("Index Error")
    items = list(data.items())
    index1 = next(i for i, (k, v) in enumerate(items) if k == key1)
    index2 = next(i for i, (k, v) in enumerate(items) if k == key2)
    items[index1], items[index2] = items[index2], items[index1]
    new_data = dict(items)
    return new_data

def get_job_table_sample(workload_file_name, num_materialized_samples = 1000):
    tables = []
    samples = []
    # Load queries
    with open(workload_file_name + ".csv", 'r') as f:
        data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
        for row in data_raw:
            tables.append(row[0].split(','))

            if int(row[3]) < 1:
                print("Queries must have non-zero cardinalities")
                exit(1)
    print("Loaded queries with len ", len(tables))
    # Load bitmaps
    num_bytes_per_bitmap = int((num_materialized_samples + 7) >> 3)
    with open(workload_file_name + ".bitmaps", 'rb') as f:
        for i in range(len(tables)):
            four_bytes = f.read(4)
            if not four_bytes:
                print("Error while reading 'four_bytes'")
                exit(1)
            num_bitmaps_curr_query = int.from_bytes(four_bytes, byteorder='little')
            bitmaps = np.empty((num_bitmaps_curr_query, num_bytes_per_bitmap * 8), dtype=np.uint8)
            for j in range(num_bitmaps_curr_query):
                # Read bitmap
                bitmap_bytes = f.read(num_bytes_per_bitmap)
                if not bitmap_bytes:
                    print("Error while reading 'bitmap_bytes'")
                    exit(1)
                bitmaps[j] = np.unpackbits(np.frombuffer(bitmap_bytes, dtype=np.uint8))
            samples.append(bitmaps)
    print("Loaded bitmaps")
    table_sample = []
    for ts, ss in zip(tables,samples):
        d = {}
        for t, s in zip(ts,ss):
            tf = t.split(' ')[0] # remove alias
            d[tf] = s
        table_sample.append(d)
    return table_sample

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
            ]  # Support table.alias.column names in histogram files.
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
        """Extract SQL subqueries and save each as a separate file."""
        # Read the SQL file.
        with open(self.sql_file_path, 'r') as file:
            self.original_sql = file.read()
        
        # Create the output directory.
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract subqueries.
        self.processed_sql, _ = self._process_subqueries(self.original_sql, "main")
        
        # Save extracted subqueries to files.
        for subquery_id, (sql, parent, is_correlated) in self.subqueries.items():
            file_name = f"{subquery_id}.sql"
            file_path = os.path.join(self.output_dir, file_name)
            
            with open(file_path, 'w') as file:
                # Add a comment identifying the subquery source.
                file.write(f"-- 来源: {parent}\n")
                file.write(f"-- 是否相关子查询: {is_correlated}\n")
                file.write(sql)
        
        # Save the processed main query.
        main_query_path = os.path.join(self.output_dir, "main_query.sql")
        with open(main_query_path, 'w') as file:
            file.write(self.processed_sql)
            
        print(f"Extracted {len(self.subqueries)} subqueries to directory: {self.output_dir}")
        
    def _process_subqueries(self, sql, parent_name):
        """Recursively process SQL subqueries."""
        # Match parenthesized SELECT statements.
        subquery_pattern = re.compile(
            r'\((\s*SELECT\s+(?:(?!\);?\s*\)).)*\s*)\)', 
            re.IGNORECASE | re.DOTALL
        )
        
        # Replace all subqueries.
        processed_sql = sql
        matches = list(subquery_pattern.finditer(sql))
        
        # Process replacements backwards so earlier indices remain valid.
        for match in reversed(matches):
            subquery_text = match.group(1).strip()
            is_correlated = self._is_correlated_subquery(subquery_text, sql)
            
            # Generate a subquery ID.
            subquery_type = "correlated" if is_correlated else "non_correlated"
            self.subquery_counter[subquery_type] += 1
            subquery_id = f"{subquery_type}_{self.subquery_counter[subquery_type]}"
            
            # Recursively process nested subqueries.
            processed_subquery, _ = self._process_subqueries(subquery_text, subquery_id)
            
            # Save the subquery.
            self.subqueries[subquery_id] = (processed_subquery, parent_name, is_correlated)
            
            # Replace the subquery in the original SQL with a placeholder.
            placeholder = f"/* SUBQUERY_{subquery_id} */"
            processed_sql = processed_sql[:match.start()] + placeholder + processed_sql[match.end():]
        
        return processed_sql, matches
    
    def _is_correlated_subquery(self, subquery, outer_sql):
        """Check whether a subquery is correlated."""
        # Check whether the subquery references outer-query table aliases.
        # This simplified check does not cover every SQL correlation pattern.
        table_aliases = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+AS\s+([a-zA-Z_][a-zA-Z0-9_]*)\b', outer_sql)
        table_aliases = [alias for _, alias in table_aliases]
        
        # Check whether the subquery uses these aliases.
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
    """Load per-database features from a .npz file.

    Expected format (simple & robust):
        np.savez(npz_path,
                 **{f"{db}_hist": hist_vec, f"{db}_bitmap": bitmap_vec, ...})

    Returns:
        bank: dict[str, dict[str, np.ndarray]] where bank[db]['hist'/'bitmap'] are 1D vectors.
    """
    if (npz_path is None) or (not os.path.exists(npz_path)):
        return None

    data = np.load(npz_path, allow_pickle=True)
    bank = {}
    keys = list(data.keys())

    # If user provides db list, only load those; otherwise infer from key prefixes.
    if databases is None:
        # infer db names from "<db>_hist" keys
        dbs = sorted({k[:-5] for k in keys if k.endswith("_hist")})
    else:
        dbs = list(databases)

    for db in dbs:
        hist_key = f"{db}_hist"
        bmp_key  = f"{db}_bitmap"
        if hist_key in data:
            bank.setdefault(db, {})['hist'] = make_writable_np(np.array(data[hist_key], copy=True)).astype(np.float32)
        if bmp_key in data:
            bank.setdefault(db, {})['bitmap'] = make_writable_np(np.array(data[bmp_key], copy=True)).astype(np.float32)

    return bank

def get_db_features(bank, db: str, hist_dim: int = 0, bitmap_dim: int = 0):
    """Fetch (hist, bitmap) vectors for a database; return zero vectors when missing."""
    if bank is None or db not in bank:
        hist = np.zeros((hist_dim,), dtype=np.float32) if hist_dim > 0 else None
        bmp  = np.zeros((bitmap_dim,), dtype=np.float32) if bitmap_dim > 0 else None
        return hist, bmp

    hist = bank[db].get('hist', None)
    bmp  = bank[db].get('bitmap', None)

    # Defensive: enforce fixed length expected by model.
    if hist_dim > 0:
        if hist is None:
            hist = np.zeros((hist_dim,), dtype=np.float32)
        else:
            hist = np.asarray(hist, dtype=np.float32).reshape(-1)
            if hist.shape[0] != hist_dim:
                # truncate or pad to fit
                if hist.shape[0] > hist_dim:
                    hist = hist[:hist_dim]
                else:
                    pad = np.zeros((hist_dim - hist.shape[0],), dtype=np.float32)
                    hist = np.concatenate([hist, pad], axis=0)
    else:
        hist = None

    if bitmap_dim > 0:
        if bmp is None:
            bmp = np.zeros((bitmap_dim,), dtype=np.float32)
        else:
            bmp = np.asarray(bmp, dtype=np.float32).reshape(-1)
            if bmp.shape[0] != bitmap_dim:
                if bmp.shape[0] > bitmap_dim:
                    bmp = bmp[:bitmap_dim]
                else:
                    pad = np.zeros((bitmap_dim - bmp.shape[0],), dtype=np.float32)
                    bmp = np.concatenate([bmp, pad], axis=0)
    else:
        bmp = None

    return make_writable_np(np.array(hist, copy=True)) if hist is not None else None, make_writable_np(np.array(bmp, copy=True)) if bmp is not None else None

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
_table_sample_cache = {}
_pickle_sample_cache = {}
def load_histogram_json_as_df(database: str, hist_json_dir: str):
    """Load collect.py-produced <db>_histogram_string.json into a DataFrame.
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
    if df is None:
        return np.zeros((hist_dim,), dtype=np.float32)
    hist_filter_dict = {'colId': [], 'opId': [], 'val': [], 'dtype': []}  # Encode only filters compatible with numeric histograms.
    for col_id, op_id, val, dtype in zip(
        (filter_dict or {}).get('colId', []),
        (filter_dict or {}).get('opId', []),
        (filter_dict or {}).get('val', []),
        (filter_dict or {}).get('dtype', []),
    ):
        if int(dtype) != 0:
            continue
        hist_filter_dict['colId'].append(int(col_id))
        hist_filter_dict['opId'].append(int(op_id))
        hist_filter_dict['val'].append(float(val))
        hist_filter_dict['dtype'].append(int(dtype))
        if len(hist_filter_dict['colId']) >= 3:
            break
    out = np.zeros((hist_dim,), dtype=np.float32)
    if hist_filter_dict['colId']:
        try:
            h3 = filterDict2Hist(df, hist_filter_dict, encoding).astype(np.float32).reshape(-1)
            out[:min(hist_dim, len(h3))] = h3[:hist_dim]
        except Exception:
            pass

    if getattr(config, 'mode', None) == 'TPCDS':
        fact_columns = (
            'store_sales.ss_sold_date_sk', 'catalog_sales.cs_sold_date_sk',
            'web_sales.ws_sold_date_sk', 'store_returns.sr_returned_date_sk',
            'catalog_returns.cr_returned_date_sk', 'web_returns.wr_returned_date_sk',
            'inventory.inv_date_sk',
        )
        offset = 3 * (int(config.db_hist_bins_per_col) - 1)
        needed = offset + 5 * len(fact_columns)
        if hist_dim < needed:
            raise ValueError('TPCDS db_hist_dim must be at least {}'.format(needed))
        for index, column in enumerate(fact_columns):
            matched = df.loc[df['table_column'] == column, 'year_fractions']
            if len(matched) != 1:
                raise ValueError('TPCDS fact-date histogram missing: {} in {}'.format(column, database))
            fractions = np.asarray(matched.iloc[0], dtype=np.float32)
            if fractions.shape != (5,):
                raise ValueError('Invalid TPCDS year fractions: {} in {}'.format(column, database))
            out[offset + 5 * index:offset + 5 * (index + 1)] = fractions
    return out

def load_workload_table_samples(workload_file_name: str, sample_size: int = 1000):
    """Load QueryFormer-style per-query table sample bitmaps from <workload>.bitmaps.
    Returns: list[dict[table_name -> np.uint8[sample_size]]]
    """
    key = (workload_file_name, int(sample_size))
    if key in _table_sample_cache:
        return _table_sample_cache[key]
    if os.path.exists(workload_file_name + '.bitmaps') and os.path.exists(workload_file_name + '.csv'):
        table_sample = get_job_table_sample(workload_file_name, num_materialized_samples=int(sample_size))
        _table_sample_cache[key] = table_sample
        return table_sample
    _table_sample_cache[key] = None
    return None

def load_pickle_sample_bitmaps(database: str, bitmap_pickle_dir: str):
    """Fallback: load collect.py-produced <db>_sample.bitmap (pickle)."""
    key = (database, bitmap_pickle_dir)
    if key in _pickle_sample_cache:
        return _pickle_sample_cache[key]
    path = os.path.join(bitmap_pickle_dir, f"{database}_sample.bitmap")
    if not os.path.exists(path):
        _pickle_sample_cache[key] = None
        return None
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    _pickle_sample_cache[key] = data
    return data

def build_query_bitmap_vector(database: str, query_id, encoding, config):
    """Build fixed-length bitmap feature for current query.
    Priority:
      1) QueryFormer <workload>.bitmaps (via get_job_table_sample)
      2) collect.py <db>_sample.bitmap (pickle) fallback
    Output length is config.db_bitmap_dim (pad/trunc).
    """
    bmp_dim = int(getattr(config, 'db_bitmap_dim', 0))
    if bmp_dim <= 0:
        return np.zeros((0,), dtype=np.float32)

    sample_size = int(getattr(config, 'sample_size', 1000))
    workload = getattr(config, 'workload_file_name', './experiment/JOBRand/synthetic')
    # allow workload_file_name to be dict: {db: prefix}
    if isinstance(workload, dict):
        workload = workload.get(database, None)
    table_samples = load_workload_table_samples(workload, sample_size) if workload else None

    # Try the full query ID first, then its numeric prefix for JOB IDs such as 14a.
    qidx_candidates = []
    try:
        qidx_candidates.append(int(query_id))
    except Exception:
        pass
    if isinstance(query_id, str):
        prefix = ''
        for ch in query_id:
            if ch.isdigit():
                prefix += ch
            else:
                break
        if prefix:
            qidx_candidates.append(int(prefix))
    qidx_candidates = [qidx for i, qidx in enumerate(qidx_candidates) if qidx >= 0 and qidx not in qidx_candidates[:i]]

    # Decide source for this query
    q_dict = None
    if table_samples is not None:
        for qidx in qidx_candidates:
            if 0 <= qidx < len(table_samples):
                q_dict = table_samples[qidx]
                break
    if q_dict is None:
        pickles = load_pickle_sample_bitmaps(database, getattr(config, 'bitmap_pickle_dir', './experiment/sample'))
        if pickles is not None:
            for qidx in qidx_candidates:
                if 0 <= qidx < len(pickles):
                    q_dict = pickles[qidx]
                    break

    if q_dict is None:
        return np.zeros((bmp_dim,), dtype=np.float32)

    # Flatten into a fixed layout: [tablenum, sample_size] -> [tablenum*sample_size]
    # Use encoding.table2idx to place each table's bitmap.
    tablenum = int(getattr(config, 'tablenum', 0))
    if tablenum <= 0:
        # fallback: just concatenate available tables then pad/trunc
        flat = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1) for v in q_dict.values()]) if len(q_dict) else np.zeros((0,), dtype=np.float32)
        out = np.zeros((bmp_dim,), dtype=np.float32)
        out[:min(bmp_dim, flat.shape[0])] = flat[:min(bmp_dim, flat.shape[0])]
        return out

    mat = np.zeros((tablenum, sample_size), dtype=np.float32)
    for tname, bits in q_dict.items():
        base = tname.split(' ')[0]
        tid = encoding.table2idx.get(base, 0)
        if tid <= 0 or tid >= tablenum:
            continue
        b = np.asarray(bits, dtype=np.float32).reshape(-1)
        if b.shape[0] != sample_size:
            b = b[:sample_size] if b.shape[0] > sample_size else np.pad(b, (0, sample_size - b.shape[0]))
        mat[tid, :] = b
    flat = mat.reshape(-1)
    # enforce bmp_dim
    if flat.shape[0] == bmp_dim:
        return flat
    out = np.zeros((bmp_dim,), dtype=np.float32)
    out[:min(bmp_dim, flat.shape[0])] = flat[:min(bmp_dim, flat.shape[0])]
    return out    
if __name__ == "__main__":
    
    sql_file = "input.sql"  
    extractor = SQLSubqueryExtractor(sql_file)
    extractor.extract_subqueries()
