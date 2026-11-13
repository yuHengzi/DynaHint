"""Database and plan-tree utilities."""
import numpy as np
from collections import deque
from shared.util import filterDict2Hist
import math
import os
import re
JOINTYPE = ["Nested Loop", "Hash Join", "Merge Join"]  # Join types.
CONDTYPE = ['Hash Cond','Join Filter','Index Cond','Merge Cond','Filter','Recheck Cond']  # Predicate types.
SCANTYPE = ['Index Only Scan', 'Seq Scan', 'Index Scan', 'Bitmap Heap Scan','Tid Scan']  # Scan types.
BINOP = [' >= ',' <= ',' = ',' > ',' < ']  # Binary operators.

LOGICAL_JOIN_COND_TYPES = ('Hash Cond', 'Merge Cond', 'Join Filter', 'Index Cond', 'Filter')
_JOIN_COLUMN_REF = r'(?:(?P<{prefix}_alias>[A-Za-z_][\w$]*)\.)?(?P<{prefix}_column>[A-Za-z_][\w$]*)'
_JOIN_PREDICATE_RE = re.compile(
    _JOIN_COLUMN_REF.format(prefix='left')
    + r'(?:\s*::\s*[A-Za-z_][\w$]*(?:\[\])?)?\s*'
    + r'(?P<operator>>=|<=|<>|=|>|<)\s*'
    + _JOIN_COLUMN_REF.format(prefix='right')
    + r'(?:\s*::\s*[A-Za-z_][\w$]*(?:\[\])?)?',
    flags=re.IGNORECASE,
)


def extract_logical_join_graph(plan, alias2table):
    """Extract alias-level column-to-column predicates from a PostgreSQL plan.

    The result is independent of the physical join-node ordering.  Relation aliases
    are sorted to give every query a deterministic, one-based relation id.  Exact
    duplicate predicates can be emitted by more than one plan condition (notably a
    parameterized Index Cond); those duplicates are removed, while distinct
    predicates between the same relation pair are preserved.
    """
    aliases = sorted(alias for alias, table in alias2table.items() if alias and table)
    alias_to_rel_id = {alias: idx + 1 for idx, alias in enumerate(aliases)}
    reverse_operator = {'<': '>', '<=': '>=', '>': '<', '>=': '<='}
    edges = []
    seen = set()

    def visit(node, inherited_alias=None):
        node_alias = node.get('Alias') or inherited_alias
        for cond_type in LOGICAL_JOIN_COND_TYPES:
            condition = node.get(cond_type)
            if not isinstance(condition, str):
                continue
            for match in _JOIN_PREDICATE_RE.finditer(condition.replace('"', '')):
                left_alias = match.group('left_alias') or node_alias
                right_alias = match.group('right_alias') or node_alias
                if (
                    left_alias not in alias_to_rel_id
                    or right_alias not in alias_to_rel_id
                    or left_alias == right_alias
                ):
                    continue
                left_column = match.group('left_column')
                right_column = match.group('right_column')
                operator = match.group('operator')
                left_id = alias_to_rel_id[left_alias]
                right_id = alias_to_rel_id[right_alias]
                if left_id > right_id:
                    left_alias, right_alias = right_alias, left_alias
                    left_column, right_column = right_column, left_column
                    left_id, right_id = right_id, left_id
                    operator = reverse_operator.get(operator, operator)
                edge = {
                    'left_rel_id': left_id,
                    'right_rel_id': right_id,
                    'left_alias': left_alias,
                    'right_alias': right_alias,
                    'left_column': f'{alias2table[left_alias]}.{left_column}',
                    'right_column': f'{alias2table[right_alias]}.{right_column}',
                    'operator': operator,
                }
                dedup_key = (
                    edge['left_rel_id'], edge['right_rel_id'], edge['left_column'],
                    edge['right_column'], edge['operator'],
                )
                if dedup_key not in seen:
                    seen.add(dedup_key)
                    edges.append(edge)
        for child in node.get('Plans', []):
            visit(child, node_alias)

    visit(plan)
    return {
        'aliases': aliases,
        'alias_to_rel_id': alias_to_rel_id,
        'base_tables': [alias2table[alias] for alias in aliases],
        'edges': edges,
    }

STATS_ALIAS={'comments':'c','posthistory':'ph','votes':'v','badges':'b','users':'u','postlinks':'pl','posts':'p','tags':'t'}
def GetParam(config):
    import json
    with open(config.auto_config, 'r', encoding='utf-8') as json_file:  # Open the file with UTF-8 encoding.
        data = json.load(json_file)  # Load JSON data.
    bins_per_col = data.get("db_hist_bins_per_col", getattr(config, "db_hist_bins_per_col", 51))
    data["node_hist_dim"] = data.get("node_hist_dim", 3 * (bins_per_col - 1))
    if "maxjoins" in data and "filtmaxnum" in data:
        data["num_node_feature"] = 7 + data["maxjoins"] + 5 * data["filtmaxnum"] + data["node_hist_dim"]
    return data

def bfs(N, pc_dict, rel_pos_max): 
    # Use breadth-first search to construct the distance matrix.
    distance_matrix = np.full((N, N), True)
    for start_node in range(N):
        queue = deque([(start_node, 0)])  # node, distance
        while queue:
            node, distance = queue.popleft()
            for end_node in pc_dict[node]:
                if distance + 1 < rel_pos_max:
                    distance_matrix[start_node][end_node] = False
                queue.append((end_node, distance + 1))
        distance_matrix[start_node][start_node] = False
    return distance_matrix

def node2feature(node, config, encoding=None, hist_file=None):
    # Convert a node to its feature vector.
    if config.AutoGetParam and os.path.exists(config.auto_config):
        data = GetParam(config)
        config.filtmaxnum = data["filtmaxnum"]
        config.node_hist_dim = data.get("node_hist_dim", getattr(config, "node_hist_dim", 0))
    filtmaxnum = config.filtmaxnum
    node_hist_dim = int(getattr(config, "node_hist_dim", 0))
    num_filter = len(node.filterDict['colId'])
    if num_filter > filtmaxnum:
        print(f'Query:{node.query_id} has {num_filter} filters')
    kept_filter_num = min(num_filter, filtmaxnum)
    pad = np.zeros((4, filtmaxnum - kept_filter_num))
    filt_values = [np.asarray(v[:kept_filter_num]) for v in node.filterDict.values()]
    filts = np.array(filt_values)  #cols, ops, vals, dtype
    filts = np.concatenate((filts, pad), axis=1).flatten()
    mask = np.zeros(filtmaxnum)
    mask[:kept_filter_num] = 1
    type_join = np.array([node.typeId] + node.join)
    if (encoding is not None) and (hist_file is not None) and node_hist_dim > 0 and num_filter > 0:
        try:
            hist_filter_dict = {k: v[:min(num_filter, 3)] for k, v in node.filterDict.items()}
            hists = filterDict2Hist(hist_file, hist_filter_dict, encoding).astype(np.float32).reshape(-1)
            if hists.shape[0] > node_hist_dim:
                hists = hists[:node_hist_dim]
            elif hists.shape[0] < node_hist_dim:
                hists = np.concatenate([hists, np.zeros((node_hist_dim - hists.shape[0],), dtype=np.float32)], axis=0)
        except Exception:
            hists = np.zeros((node_hist_dim,), dtype=np.float32)
    else:
        hists = np.zeros((node_hist_dim,), dtype=np.float32)
    table = np.array([node.table_id])
    pos = np.array([node.pos])
    db_est = np.array(node.db_est)
    return np.concatenate((type_join, filts, mask, hists, pos, table, db_est))

def pad_1d_unsqueeze(x, padlen):
    # Pad a one-dimensional array.
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    # Pad a two-dimensional array.
    xlen, xdim = x.size()
    # x = x + 1 # pad id = 0
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) # + 1
        new_x[:xlen, :] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen, alpha):
    # Pad the attention-bias matrix.
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(alpha)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = alpha
        x = new_x
    return x

def processCond(json_node, alias, alias2table):
    # Parse predicate operators.
    join = []
    filters = set()
    for condtype in CONDTYPE:
        if condtype in json_node:
            condition = json_node[condtype]
            if ' AND ' in condition:
                condition = condition[1:-1]
            cond_list = condition.split(' AND ')
            for cond in cond_list:
                cond = cond[1:-1]
                if condtype == 'Filter' or '::text' in cond or cond[-1].isnumeric():
                    filters.add((cond))
                else:
                    for op in BINOP:
                        if op in cond:
                            twoCol = [col.split(' ')[0].strip('() ') for col in cond.split(op)]
                            onejoin = [op]
                            for col in twoCol:
                                col_split = col.split('.')
                                if len(col_split) == 1:
                                    onejoin.append(alias2table[alias] + '.' + col_split[0])
                                else:
                                    if col_split[1][-2:] == '),':
                                        col_split[1] = col_split[1][:-2]
                                    if col_split[0] not in alias2table:
                                        # Record the invalid alias and its context.
                                        print(f"[Error] Missing alias '{col_split[0]}' in alias2table.")
                                        print(f"Alias2Table Content: {alias2table}")
                                    onejoin.append(alias2table[col_split[0]] + '.' + col_split[1])
                            join.append(onejoin)
                            break
    planrows    = math.log10(1 + int(json_node['Plan Rows']))
    totalcost   = math.log10(1 + int(json_node['Total Cost']))
    planwidth   = math.log10(1 + int(json_node["Plan Width"]))
    startupcost = math.log10(1 + int(json_node['Startup Cost']))
    db_est = [planrows, totalcost, planwidth, startupcost]
    return join, list(filters), db_est

class TreeNode:
    # Plan-tree node.
    def __init__(self, nodeType,table,table_id ,typeId, filt, join,
                 filterDict, db_est, pos):
        self.nodeType = nodeType    # Node type.
        self.typeId = typeId        # Node-type index.
        self.filter = filt          # Filters.

        self.table = table
        self.table_id = table_id    # Table index.
        self.query_id = None        # Query ID.
        self.join = join            # Join.
        self.children = []          # Child nodes.
        self.rounds = 0             # Iteration.

        self.filterDict = filterDict  # Filter dictionary.
        self.db_est = db_est          # Database estimates.
        self.pos = pos                # Position.
        self.alias = None             # Alias.
        self.parent = None            # Parent node.

        self.feature = None           # Features.

    def addChild(self, treeNode):
        # Add a child node.
        self.children.append(treeNode)

    def __str__(self):
        # Return the node's string representation.
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter,
                                                    self.join_str,
                                                    len(self.children))

    def __repr__(self):
        # Return the node's string representation.
        return self.__str__()

    @staticmethod
    def print_nested(node, indent=0):
        # Print nested nodes.
        print('--' * indent + '{} with {} and {}, {} childs'.format(
            node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children:
            TreeNode.print_nested(k, indent + 1)
