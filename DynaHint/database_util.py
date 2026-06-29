import numpy as np
from collections import deque
from util import filterDict2Hist
import math
import os
JOINTYPE = ["Nested Loop", "Hash Join", "Merge Join"]
CONDTYPE = ['Hash Cond','Join Filter','Index Cond','Merge Cond','Filter','Recheck Cond']
SCANTYPE = ['Index Only Scan', 'Seq Scan', 'Index Scan', 'Bitmap Heap Scan','Tid Scan']
BINOP = [' >= ',' <= ',' = ',' > ',' < ']

STATS_ALIAS={'comments':'c','posthistory':'ph','votes':'v','badges':'b','users':'u','postlinks':'pl','posts':'p','tags':'t'}
def GetParam(config):
    import json
    with open(config.auto_config, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    bins_per_col = data.get("db_hist_bins_per_col", getattr(config, "db_hist_bins_per_col", 51))
    data["node_hist_dim"] = data.get("node_hist_dim", 3 * (bins_per_col - 1))
    if "maxjoins" in data and "filtmaxnum" in data:
        data["num_node_feature"] = 7 + data["maxjoins"] + 5 * data["filtmaxnum"] + data["node_hist_dim"]
    return data

def bfs(N, pc_dict, rel_pos_max): 
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
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    # x = x + 1 # pad id = 0
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype) # + 1
        new_x[:xlen, :] = x
        x = new_x
    return x#.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen, alpha):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(alpha)
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = alpha
        x = new_x
    return x

def processCond(json_node, alias, alias2table):
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
    def __init__(self, nodeType,table,table_id ,typeId, filt, join,
                 filterDict, db_est, pos):
        self.nodeType = nodeType
        self.typeId = typeId
        self.filter = filt

        self.table = table
        self.table_id = table_id
        self.query_id = None
        self.join = join
        self.children = []
        self.rounds = 0

        self.filterDict = filterDict
        self.db_est = db_est
        self.pos = pos
        self.alias = None
        self.parent = None

        self.feature = None

    def addChild(self, treeNode):
        self.children.append(treeNode)

    def __str__(self):
        return '{} with {}, {}, {} children'.format(self.nodeType, self.filter,
                                                    self.join_str,
                                                    len(self.children))

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def print_nested(node, indent=0):
        print('--' * indent + '{} with {} and {}, {} childs'.format(
            node.nodeType, node.filter, node.join_str, len(node.children)))
        for k in node.children:
            TreeNode.print_nested(k, indent + 1)
