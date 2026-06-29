import torch
import numpy as np
from gymnasium import spaces
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from config import Config
import json
import os
config = Config()
config.ConfirmPath()
DB_STATE_FEATURE_KEYS = {
    'db_hist',
    'db_table_stats',
    'query_table_mask',
    'db_global_stats',
    'query_col_stats',
    'query_feat',
}


def parse_feature_mask(mask):
    if mask is None or mask == 'all':
        return set(DB_STATE_FEATURE_KEYS)
    if isinstance(mask, (list, tuple, set)):
        return {str(item).strip() for item in mask if str(item).strip()}
    return {item.strip() for item in str(mask).split(',') if item.strip()}


def GetParam(param_config):
    # print(param_config.auto_config)
    with open(param_config.auto_config, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    bins_per_col = data.get("db_hist_bins_per_col", getattr(param_config, "db_hist_bins_per_col", 51))
    data["node_hist_dim"] = data.get("node_hist_dim", 3 * (bins_per_col - 1))
    if "maxjoins" in data and "filtmaxnum" in data:
        data["num_node_feature"] = 7 + data["maxjoins"] + 5 * data["filtmaxnum"] + data["node_hist_dim"]
    return data


class QueryConditionedDataAttention(nn.Module):
    """Encode DB-state tokens, then let query/plan tokens attend to them."""

    def __init__(self, param_config, hidden_dim: int, dropout: float = 0.05):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.db_table_feat_dim = int(getattr(param_config, 'db_table_feat_dim', 6))
        self.db_global_feat_dim = int(getattr(param_config, 'db_global_feat_dim', 4))
        self.db_col_feat_dim = int(getattr(param_config, 'db_col_feat_dim', 4))
        self.db_hist_dim = int(getattr(param_config, 'db_hist_dim', 0))
        self.query_feat_dim = int(getattr(param_config, 'query_feat_dim', 0))
        self.max_tables = int(getattr(param_config, 'max_tables_in_schema', 256))
        self.kv_feature_mask = parse_feature_mask(getattr(param_config, 'cross_attn_kv_feature_mask', 'all'))
        self.q_feature_mask = parse_feature_mask(getattr(param_config, 'cross_attn_q_feature_mask', 'all'))
        self.q_use_query_feat = bool(getattr(param_config, 'cross_attn_q_use_query_feat', True))
        self.q_use_plan_nodes = bool(getattr(param_config, 'cross_attn_q_use_plan_nodes', True))

        self.table_proj = nn.Linear(self.db_table_feat_dim, hidden_dim)
        self.table_mask_proj = nn.Linear(1, hidden_dim)
        self.table_pos_emb = nn.Embedding(self.max_tables, hidden_dim)
        self.global_proj = nn.Linear(self.db_global_feat_dim, hidden_dim)
        self.col_proj = nn.Linear(self.db_col_feat_dim, hidden_dim)
        self.hist_proj = nn.Linear(self.db_hist_dim, hidden_dim) if self.db_hist_dim > 0 else None
        self.query_feat_proj = nn.Linear(self.query_feat_dim, hidden_dim) if self.query_feat_dim > 0 else None
        self.empty_memory = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        requested_heads = int(getattr(param_config, 'scorer_cross_attn_heads', 3))
        self.num_heads = self._compatible_heads(hidden_dim, requested_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.data_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(getattr(param_config, 'scorer_cross_attn_layers', 1)),
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_ln = nn.LayerNorm(hidden_dim)
        self.ffn_ln = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.out_ln = nn.LayerNorm(hidden_dim)

    @staticmethod
    def _compatible_heads(hidden_dim, requested_heads):
        requested_heads = max(1, int(requested_heads))
        if hidden_dim % requested_heads == 0:
            return requested_heads
        for candidate in range(requested_heads, 0, -1):
            if hidden_dim % candidate == 0:
                return candidate
        return 1

    @staticmethod
    def _as_2d(feature, key, batch_size, width, device):
        value = feature.get(key, None) if isinstance(feature, dict) else None
        if value is None:
            return torch.zeros((batch_size, width), device=device)
        value = value.to(device).float()
        if value.dim() == 1:
            value = value.unsqueeze(0)
        if value.size(0) == 1 and batch_size > 1:
            value = value.expand(batch_size, -1)
        return value

    def _build_data_tokens(self, feature, batch_size, device):
        tokens = []
        db_table = feature.get('db_table_stats', None) if isinstance(feature, dict) and 'db_table_stats' in self.kv_feature_mask else None
        if db_table is not None:
            db_table = db_table.to(device).float()
            if db_table.dim() == 2:
                db_table = db_table.unsqueeze(0)
            if db_table.size(0) == 1 and batch_size > 1:
                db_table = db_table.expand(batch_size, -1, -1)
            _, table_count, _ = db_table.shape
            pos = torch.arange(table_count, device=device).clamp(max=self.max_tables - 1)
            table_tokens = self.table_proj(db_table) + self.table_pos_emb(pos).unsqueeze(0)
            query_mask = feature.get('query_table_mask', None) if isinstance(feature, dict) and 'query_table_mask' in self.kv_feature_mask else None
            if query_mask is None:
                query_mask = torch.zeros((batch_size, table_count), device=device)
            else:
                query_mask = query_mask.to(device).float()
                if query_mask.dim() == 1:
                    query_mask = query_mask.unsqueeze(0)
                if query_mask.size(0) == 1 and batch_size > 1:
                    query_mask = query_mask.expand(batch_size, -1)
                if query_mask.size(1) != table_count:
                    query_mask = query_mask[:, :table_count]
                    if query_mask.size(1) < table_count:
                        pad = torch.zeros((batch_size, table_count - query_mask.size(1)), device=device)
                        query_mask = torch.cat([query_mask, pad], dim=1)
            table_tokens = table_tokens + self.table_mask_proj(query_mask.unsqueeze(-1))
            tokens.append(table_tokens)

        if 'db_global_stats' in self.kv_feature_mask:
            db_global = self._as_2d(feature, 'db_global_stats', batch_size, self.db_global_feat_dim, device)
            tokens.append(self.global_proj(db_global).unsqueeze(1))

        if 'query_col_stats' in self.kv_feature_mask:
            query_col = self._as_2d(feature, 'query_col_stats', batch_size, self.db_col_feat_dim, device)
            tokens.append(self.col_proj(query_col).unsqueeze(1))

        if self.hist_proj is not None and 'db_hist' in self.kv_feature_mask:
            db_hist = self._as_2d(feature, 'db_hist', batch_size, self.db_hist_dim, device)
            tokens.append(self.hist_proj(db_hist).unsqueeze(1))

        if len(tokens) == 0:
            return self.empty_memory.expand(batch_size, -1, -1)
        data_tokens = torch.cat(tokens, dim=1)
        return self.data_encoder(data_tokens)

    def encode_data_memory(self, feature, batch_size, device):
        return self._build_data_tokens(feature, batch_size, device)

    def _build_query_tokens(self, feature, plan_global, plan_nodes):
        tokens = [plan_global.unsqueeze(1)]
        if self.q_use_plan_nodes:
            tokens.append(plan_nodes)
        if self.q_use_query_feat and self.query_feat_proj is not None and 'query_feat' in self.q_feature_mask:
            query_feat = self._as_2d(
                feature,
                'query_feat',
                plan_global.size(0),
                self.query_feat_dim,
                plan_global.device,
            )
            tokens.append(self.query_feat_proj(query_feat).unsqueeze(1))
        return torch.cat(tokens, dim=1)

    @staticmethod
    def _masked_mean(nodes, node_mask):
        if nodes is None or nodes.size(1) == 0:
            return None
        if node_mask is None:
            return nodes.mean(dim=1)
        mask = node_mask.to(nodes.device).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (nodes * mask).sum(dim=1) / denom

    def attend_query_tokens(self, query_tokens, data_memory):
        attended, _ = self.cross_attn(query_tokens, data_memory, data_memory, need_weights=False)
        query_tokens = self.cross_ln(query_tokens + self.cross_dropout(attended))
        query_tokens = query_tokens + self.cross_dropout(self.ffn(self.ffn_ln(query_tokens)))
        return query_tokens

    def forward(self, feature, plan_global, plan_nodes, node_mask):
        data_memory = self.encode_data_memory(feature, plan_global.size(0), plan_global.device)
        query_tokens = self._build_query_tokens(feature, plan_global, plan_nodes)
        query_tokens = self.attend_query_tokens(query_tokens, data_memory)

        cross_global = query_tokens[:, 0, :]
        node_count = plan_nodes.size(1)
        cross_nodes = query_tokens[:, 1:1 + node_count, :]
        node_ctx = self._masked_mean(cross_nodes, node_mask)
        if node_ctx is None:
            node_ctx = torch.zeros_like(plan_global)
        return self.out_ln(plan_global + cross_global + node_ctx)

class FeatureEmbed(nn.Module):
    def __init__(self, param_config, embed_size = 32, tables = config.tablenum, types = config.types, columns = config.columns, \
                 ops = config.opsnum, pos = 8):
        super(FeatureEmbed, self).__init__()
        self.filtmaxnum = config.filtmaxnum
        self.maxjoins   = config.maxjoins
        self.node_hist_dim = int(getattr(config, "node_hist_dim", 0))
        if os.path.exists(param_config.auto_config):
            data = GetParam(param_config)
            if data["AutoGetParam"]==1:
                tables = data["tablenum"]
                types = data["types"]
                columns = data["columns"]
                ops = data["opsnum"]
                self.maxjoins = data["maxjoins"]
                pos = data["maxpos"]
                self.filtmaxnum = data["filtmaxnum"]
                self.node_hist_dim = data.get("node_hist_dim", self.node_hist_dim)


        # print(types)
        self.typeEmbed      = nn.Embedding(types, embed_size)
        self.tableEmbed     = nn.Embedding(tables, embed_size)
        self.columnEmbed    = nn.Embedding(columns, 2 * embed_size)
        self.opEmbed        = nn.Embedding(ops, embed_size // 8)
        self.dtypeEmbed     = nn.Embedding(4, embed_size // 4)
        self.posEmbed       = nn.Embedding(pos, embed_size // 8)
        self.linearFilter2  = nn.Linear(2 * embed_size  + 3 * embed_size // 8 + 1,
                                        2 * embed_size  + 3 * embed_size // 8 + 1)
        self.linearFilter   = nn.Linear(2 * embed_size  + 3 * embed_size // 8 + 1,
                                        2 * embed_size  + 3 * embed_size // 8 + 1) #        
        self.linearJoin1    = nn.Linear(2 * self.maxjoins  * embed_size, 
                                        3 * embed_size)
        self.linearJoin2    = nn.Linear(3 * embed_size,  3 * embed_size)
        self.histProject    = nn.Sequential(
            nn.Linear(self.node_hist_dim, embed_size),
            nn.LeakyReLU(),
            nn.Linear(embed_size, embed_size),
        )
        self.linearest      = nn.Linear(4, embed_size // 2)
        self.project_in_dim = embed_size * 8 + 8 * (embed_size // 8) + 1
        self.project_out_dim = embed_size * 7 + 8 * (embed_size // 8) + 1
        self.project        = nn.Linear(self.project_in_dim, self.project_out_dim)
        # self.project        = nn.Linear(embed_size * 5 + 5 * (embed_size // 8),
        #                                 embed_size * 5 + 5 * (embed_size // 8))
    def forward(self, feature):
        typeId, join, filtersId, filtersMask, histId, posId,table,db_est = torch.split(
                feature, (1, self.maxjoins, self.filtmaxnum*4, self.filtmaxnum, self.node_hist_dim, 1, 1, 4), dim = -1)
        typeEmb     = self.getType(typeId)
        joinEmb     = self.getJoin(join)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        histEmb     = self.getHist(histId)
        dbest       = self.linearest(db_est)
        tableEmb    = self.getTable(table)
        posEmb      = self.getPos(posId)
        # final       = torch.cat((typeEmb, joinEmb, tableEmb, posEmb, dbest),dim=1)
        final       = torch.cat((typeEmb, filterEmbed, joinEmb, histEmb, tableEmb, posEmb, dbest),dim=1)
        temp        = self.project(final)
        final       = F.leaky_relu(temp)       
        return final

    def getType(self, typeId):
        typeId = typeId.long()
        emb = self.typeEmbed(typeId).squeeze(1)
        return emb

    def getTable(self, table_sample):
        table = table_sample.long()
        # print(f'table: {table.max().item()}, tableEmbed.num_embeddings: {self.tableEmbed.num_embeddings}')
        emb = self.tableEmbed(table).squeeze(1)
        return emb
    
    def getJoin(self, joins):
        joins = joins.long()
        joins_embed = self.columnEmbed(joins)
        joins_embed = torch.cat([joins_embed[:, i, :] for i in range(self.maxjoins)], dim=-1)
        concat = F.leaky_relu(self.linearJoin1(joins_embed))
        concat = F.leaky_relu(self.linearJoin2(concat))
        concat = concat.squeeze(1)
        return concat
    
    def getPos(self, posId):  
        posId = posId.long()
        if posId.max().item() >= self.posEmbed.num_embeddings:
            raise ValueError("Index out of range in posEmbed")
   
        emb = self.posEmbed(posId).squeeze(1)
        return emb

    def getHist(self, histId):
        histId = histId.float()
        return self.histProject(histId)
    
    def getFilter(self, filtersId, filtersMask):
        filterExpand = filtersId.view(-1, 4, self.filtmaxnum).transpose(1, 2)
        colsId = filterExpand[:, :, 0].long()
        opsId = filterExpand[:, :, 1].long()
        vals = filterExpand[:, :, 2].unsqueeze(-1)  # b by 3 by 1
        dtypeId = filterExpand[:, :, 3].long()

        # b by 3 by embed_dim
        col = self.columnEmbed(colsId)
        op = self.opEmbed(opsId)
        dtype = self.dtypeEmbed(dtypeId)

        concat = torch.cat((col, op, vals, dtype), dim = -1)
        # concat = torch.cat((col, op), dim=-1)
        concat = F.leaky_relu(self.linearFilter(concat))
        concat = F.leaky_relu(self.linearFilter2(concat))
        concat[~filtersMask.bool()] = 0.
        num_filters = torch.sum(filtersMask, dim=1) + 1e-10
        total = torch.sum(concat, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg
    
class PlanNetwork(nn.Module):
    def __init__(self, param_config, emb_size = config.emb_size ,ffn_dim = config.ffn_dim, \
                 head_size = config.head_size, dropout = config.dropout, \
                 attention_dropout_rate = config.dropout, n_layers = config.num_layers):

        super(PlanNetwork, self).__init__()
        if os.path.exists(param_config.auto_config):
            data = GetParam(param_config)
            if data["AutoGetParam"]==1:
                config.heightsize = data["heightsize"]
                config.num_node_feature = data["num_node_feature"]
                config.node_hist_dim = data.get("node_hist_dim", getattr(config, "node_hist_dim", 0))
                config.db_hist_dim = data["db_hist_dim"]
                config.query_feat_dim = data["query_feat_dim"]

        self.hidden_dim     = config.hidden_dim
        self.head_size      = head_size
        self.emb_size       = emb_size
        self.height_size    = emb_size // 2
        # self.structure_size = emb_size // 2
        self.height_encoder = nn.Embedding(config.heightsize, self.height_size , padding_idx=0)
        # self.structure_encoder = nn.Embedding(config.structuresize, self.structure_size , padding_idx=0)
        self.input_dropout  = nn.Dropout(dropout)
        encoders = [
            EncoderLayer(self.hidden_dim, ffn_dim, dropout, attention_dropout_rate,
                         head_size) for _ in range(n_layers)
        ]
        self.layers         = nn.ModuleList(encoders)
        self.final_ln       = nn.LayerNorm(self.hidden_dim)
        self.embbed_layer   = FeatureEmbed(param_config, embed_size = emb_size)
        # ===== Optional: drift features (DB-level + query-level + DB-meta) =====
        # 1) DB-level histogram features + lightweight query features
        self.use_db_features = bool(getattr(config, "use_db_features", False))
        self.db_hist_dim = int(getattr(config, "db_hist_dim", 0))
        self.query_feat_dim = int(getattr(config, "query_feat_dim", 0))

        self.db_hist_proj = None
        self.query_feat_proj = None

        if self.use_db_features:
            # Each branch maps -> hidden_dim so we can sum/broadcast and also concatenate for fuse_proj.
            if self.db_hist_dim > 0:
                self.db_hist_proj = nn.Sequential(
                    nn.Linear(self.db_hist_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            if self.query_feat_dim > 0:
                self.query_feat_proj = nn.Sequential(
                    nn.Linear(self.query_feat_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )

        # 2) DB meta features (table/column statistics) with a small Transformer encoder
        self.enable_db_meta = bool(getattr(config, "enable_db_meta", False))
        if self.enable_db_meta:
            self.db_table_feat_dim = int(getattr(config, "db_table_feat_dim", 6))
            self.db_global_feat_dim = int(getattr(config, "db_global_feat_dim", 4))
            self.db_col_feat_dim = int(getattr(config, "db_col_feat_dim", 4))
            self.db_token_hidden = int(getattr(config, "db_token_hidden_dim", 128))

            self.db_pos_emb = nn.Embedding(int(getattr(config, "max_tables_in_schema", 256)), self.db_token_hidden)
            self.db_cls = nn.Parameter(torch.zeros(1, 1, self.db_token_hidden))
            self.db_proj = nn.Linear(self.db_table_feat_dim, self.db_token_hidden)

            nhead = int(getattr(config, "db_token_nhead", 4))
            nlayer = int(getattr(config, "db_token_layers", 2))
            ffn_mult = int(getattr(config, "db_token_ffn_mult", 4))
            enc_layer = nn.TransformerEncoderLayer(
                d_model=self.db_token_hidden,
                nhead=nhead,
                dim_feedforward=self.db_token_hidden * ffn_mult,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.db_encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayer)

            # map meta embedding -> hidden_dim so it can be fused with plan embedding
            self.db_to_hidden = nn.Linear(self.db_token_hidden, self.hidden_dim)
            self.query_meta_proj = nn.Linear(self.db_global_feat_dim + self.db_col_feat_dim + 2, self.hidden_dim)
            self.db_fuse_ln = nn.LayerNorm(self.hidden_dim)

        self.fuse_proj = None
        fuse_in = self.hidden_dim
        if self.db_hist_proj is not None:
            fuse_in += self.hidden_dim
        if self.query_feat_proj is not None:
            fuse_in += self.hidden_dim
        if self.enable_db_meta:
            fuse_in += self.hidden_dim

        if fuse_in != self.hidden_dim:
            self.fuse_proj = nn.Sequential(
                nn.Linear(fuse_in, self.hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )

    def forward(self, batched_data, return_nodes: bool = False, use_global_drift_fusion: bool = True, drift_feature_mask=None):
        """Encode a (batched) plan into embeddings.

        Args:
            batched_data: dict containing at least:
                - 'attn_bias': [B, N+1, N+1] or compatible
                - 'x':         [B, N, num_node_feature]
                - 'heights':   [B, N]
              Extra keys (e.g. 'steps', 'action_mask') are ignored.
            return_nodes: if True, also return per-node embeddings and a padding mask.

        Returns:
            If return_nodes=False:
                global_emb: [B, hidden_dim]
            If return_nodes=True:
                global_emb: [B, hidden_dim]
                node_emb:   [B, N, hidden_dim]
                node_mask:  [B, N] bool (True for valid nodes)
        """
        attn_bias, x = batched_data['attn_bias'], batched_data['x']
        heights = batched_data['heights'].long()

        n_batch, n_nodes = x.size()[:2]

        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        tree_attn_bias = tree_attn_bias[:, :, 1:, 1:]  # [B, head, N, N]

        x_view = x.contiguous().view(-1, config.num_node_feature)
        node_feature = self.embbed_layer(x_view).view(
            n_batch, -1, self.hidden_dim - self.height_size
        )
        height_feature = self.height_encoder(heights)

        node_feature = torch.cat([node_feature, height_feature], dim=2)  # [B, N, hidden_dim]

        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)
        output = self.final_ln(output)  # [B, N, hidden_dim]

        global_emb = output[:, 0, :]

        if use_global_drift_fusion:
            drift_feature_mask = parse_feature_mask('all' if drift_feature_mask is None else drift_feature_mask)
            # ===== Fuse drift features (optional) =====
            # These features help the shared hidden layers learn data/query drift patterns.
            parts = [global_emb]
            ctx_total = torch.zeros_like(global_emb)

            # ---- db_hist / query_feat ----
            if self.db_hist_proj is not None:
                if 'db_hist' in drift_feature_mask:
                    db_hist = batched_data.get("db_hist", None)
                    if db_hist is not None:
                        h = self.db_hist_proj(db_hist.to(global_emb.device).float())
                    else:
                        h = torch.zeros_like(global_emb)
                else:
                    h = torch.zeros_like(global_emb)
                parts.append(h)
                ctx_total = ctx_total + h

            if self.query_feat_proj is not None:
                if 'query_feat' in drift_feature_mask:
                    query_feat = batched_data.get("query_feat", None)
                    if query_feat is not None:
                        qf = self.query_feat_proj(query_feat.to(global_emb.device).float())
                    else:
                        qf = torch.zeros_like(global_emb)
                else:
                    qf = torch.zeros_like(global_emb)
                parts.append(qf)
                ctx_total = ctx_total + qf

            # ---- db meta branch ----
            if self.enable_db_meta:
                meta_ctx = torch.zeros_like(global_emb)

                db_table = batched_data.get("db_table_stats", None) if 'db_table_stats' in drift_feature_mask else None
                if db_table is not None:
                    db_table = db_table.to(global_emb.device).float()
                    B, T, _ = db_table.shape

                    db_global = batched_data.get("db_global_stats", None)
                    if 'db_global_stats' not in drift_feature_mask:
                        db_global = torch.zeros((B, self.db_global_feat_dim), device=global_emb.device)
                    elif db_global is None:
                        db_global = torch.zeros((B, self.db_global_feat_dim), device=global_emb.device)
                    else:
                        db_global = db_global.to(global_emb.device).float()

                    q_col = batched_data.get("query_col_stats", None) if 'query_col_stats' in drift_feature_mask else None
                    if q_col is None:
                        q_col = torch.zeros((B, self.db_col_feat_dim), device=global_emb.device)
                    else:
                        q_col = q_col.to(global_emb.device).float()

                    q_mask = batched_data.get("query_table_mask", None) if 'query_table_mask' in drift_feature_mask else None
                    if q_mask is None:
                        q_mask = torch.zeros((B, T), device=global_emb.device)
                    else:
                        q_mask = q_mask.to(global_emb.device).float()

                    # ---- table tokens + position embedding ----
                    pos = torch.arange(T, device=global_emb.device).clamp(max=self.db_pos_emb.num_embeddings - 1)
                    tokens = self.db_proj(db_table) + self.db_pos_emb(pos).unsqueeze(0)  # [B, T, Hdb]

                    # ---- transformer encode (add CLS token) ----
                    cls = self.db_cls.expand(B, -1, -1)      # [B, 1, Hdb]
                    db_in = torch.cat([cls, tokens], dim=1)  # [B, T+1, Hdb]
                    db_out = self.db_encoder(db_in)          # [B, T+1, Hdb]
                    db_full = db_out[:, 0, :]

                    # ---- query-focus: masked mean over encoded tokens ----
                    mask = q_mask.unsqueeze(-1)              # [B, T, 1]
                    den = mask.sum(dim=1).clamp(min=1.0)     # [B, 1]
                    db_focus = (db_out[:, 1:, :] * mask).sum(dim=1) / den  # [B, Hdb]

                    db_vec = self.db_to_hidden(db_full) + self.db_to_hidden(db_focus)  # [B, hidden_dim]

                    # ---- query meta ----
                    tables_used = q_mask.sum(dim=1, keepdim=True)         # [B, 1]
                    table_ratio = tables_used / (float(T) + 1e-6)         # [B, 1]
                    q_meta = torch.cat([db_global, q_col, torch.log1p(tables_used), table_ratio], dim=1)
                    q_meta = self.query_meta_proj(q_meta)

                    meta_ctx = self.db_fuse_ln(db_vec + q_meta)

                parts.append(meta_ctx)
                ctx_total = ctx_total + meta_ctx

            # ---- final fuse (concat -> hidden_dim) ----
            if self.fuse_proj is not None:
                global_emb = self.fuse_proj(torch.cat(parts, dim=-1))

            # Optional: let every node representation "see" the drift context too
            output = output + ctx_total.unsqueeze(1)

        if not return_nodes:
            return global_emb

        mask1 = (heights != 0)
        mask2 = (x.abs().sum(dim=-1) != 0)
        node_mask = (mask1 & mask2)

        return global_emb, output, node_mask

class FeedForwardNetwork(nn.Module):

    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

        q = q.transpose(1, 2)  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            attn_bias = attn_bias
            #x = x + attn_bias
            x = x * attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x

class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size,
                                                 attention_dropout_rate,
                                                 head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x) 
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x

def create_custom_model(param_config, hid=256, card_loss_w=0.2):
    """Create an RLlib TorchModelV2 with two branches:
    - action logits (masked)
    - final-cardinality prediction (log1p(card)) as an auxiliary task
    The auxiliary loss is injected via custom_loss().
    """

    effective_card_loss_w = float(card_loss_w) if bool(getattr(param_config, 'use_cardinality_aux_loss', True)) else 0.0

    class FinalCardLog1pHuberLoss(nn.Module):
        def __init__(self, beta: float = 1.0, reduction: str = "mean"):
            super().__init__()
            self.loss = nn.SmoothL1Loss(beta=beta, reduction=reduction)

        def forward(self, pred_log: torch.Tensor, y_card: torch.Tensor) -> torch.Tensor:
            y_card = y_card.to(pred_log.device).float()
            valid_mask = y_card >= 0
            if not torch.any(valid_mask):
                return pred_log.float().sum() * 0.0
            y_log = torch.log1p(y_card[valid_mask])
            pred_valid = pred_log.float().view_as(y_card)[valid_mask]
            return self.loss(pred_valid, y_log)

    class CustomModel(TorchModelV2, nn.Module):
        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)

            self.embedmodel = PlanNetwork(param_config)
            d = self.embedmodel.hidden_dim
            self.planner_data_fusion_mode = getattr(param_config, 'planner_data_fusion_mode', 'concat')
            if self.planner_data_fusion_mode not in ('concat', 'cross_attention'):
                raise ValueError("planner_data_fusion_mode must be 'concat' or 'cross_attention'")
            self.planner_cross_attn_enabled = self.planner_data_fusion_mode == 'cross_attention'
            self.planner_cross_attn_actor_only = bool(getattr(param_config, 'planner_cross_attn_actor_only', True))
            self.planner_cross_attn_share_with_scorer = bool(getattr(param_config, 'planner_cross_attn_share_with_scorer', False))
            self.cross_attn_use_local_plan_q = bool(getattr(param_config, 'cross_attn_use_local_plan_q', True))
            self.cross_attn_q_feature_mask = getattr(param_config, 'cross_attn_q_feature_mask', 'all')
            self.actor_data_cross_attn = QueryConditionedDataAttention(
                param_config,
                d,
                dropout=float(getattr(param_config, 'dropout', 0.05)),
            ) if self.planner_cross_attn_enabled else None

            self.card_loss_w = effective_card_loss_w
            self.card_loss_fn = FinalCardLog1pHuberLoss(beta=1.0)

            # Action branch (logits)
            self.action_head = nn.Sequential(
                nn.Linear(d + 1, hid),
                nn.LeakyReLU(),
                nn.Linear(hid, hid),
                nn.LeakyReLU(),
                nn.Linear(hid, num_outputs),
            )

            # Value branch
            self.value_head = nn.Sequential(
                nn.Linear(d + 1, hid),
                nn.LeakyReLU(),
                nn.Linear(hid, 1),
            )

            # Final-cardinality branch (predict log1p(card))
            self.card_head = nn.Sequential(
                nn.Linear(d, hid),
                nn.LeakyReLU(),
                nn.Linear(hid, 1),
            )

            self._last_rep = None
            self._last_actor_rep = None
            self._last_critic_rep = None
            self._last_card_pred_log = None
            self._last_card_label = None

        def _split_obs(self, obs_dict):
            action_mask = obs_dict.get("action_mask", None)
            steps = obs_dict.get("steps", None)
            card_label = obs_dict.get("card_label", None)

            # Do NOT mutate RLlib's input_dict in-place.
            plan_obs = {k: v for k, v in obs_dict.items()
                        if k not in ["action_mask", "steps", "card_label"]}

            return plan_obs, action_mask, steps, card_label

        def forward(self, input_dict, state, seq_lens):
            obs = input_dict["obs"]
            plan_obs, action_mask, steps, card_label = self._split_obs(obs)

            if self.planner_cross_attn_enabled:
                critic_rep, plan_nodes, node_mask = self.embedmodel(
                    plan_obs,
                    return_nodes=True,
                    use_global_drift_fusion=not self.cross_attn_use_local_plan_q,
                    drift_feature_mask=self.cross_attn_q_feature_mask,
                )
                actor_rep = self.actor_data_cross_attn(plan_obs, critic_rep, plan_nodes, node_mask)
                if not self.planner_cross_attn_actor_only:
                    critic_rep = actor_rep
            else:
                critic_rep = self.embedmodel(plan_obs)  # [B, d]
                actor_rep = critic_rep

            # Step feature: ensure [B, 1]
            if steps is None:
                steps_feat = torch.zeros((actor_rep.size(0), 1), device=actor_rep.device, dtype=actor_rep.dtype)
            else:
                steps_feat = steps.to(actor_rep.device).float()
                if steps_feat.dim() == 1:
                    steps_feat = steps_feat.unsqueeze(-1)

            actor_rep_step = torch.cat([actor_rep, steps_feat], dim=-1)  # [B, d+1]
            critic_rep_step = torch.cat([critic_rep, steps_feat.to(critic_rep.device, dtype=critic_rep.dtype)], dim=-1)
            self._last_rep = critic_rep_step
            self._last_actor_rep = actor_rep
            self._last_critic_rep = critic_rep

            # Action logits
            logits = self.action_head(actor_rep_step)  # [B, num_outputs]

            # Mask invalid actions: action_mask should be 0/1 (or >=1 means valid)
            if action_mask is not None:
                am = action_mask.to(logits.device)
                inf_mask = torch.where(
                    am > 0,
                    torch.zeros_like(am, dtype=logits.dtype),
                    torch.full_like(am, FLOAT_MIN, dtype=logits.dtype),
                )
                logits = logits + inf_mask

            # Aux final-card prediction (log1p(card)), independent of steps
            self._last_card_pred_log = self.card_head(actor_rep).squeeze(-1)  # [B]
            self._last_card_label = card_label

            return logits, state

        def value_function(self):
            assert self._last_rep is not None, "value_function() called before forward()"
            return self.value_head(self._last_rep).squeeze(-1)

        def custom_loss(self, policy_loss, loss_inputs):
            if self.card_loss_w <= 0:
                return policy_loss
            # If no label provided, skip aux loss.
            if self._last_card_label is None:
                return policy_loss

            y = self._last_card_label
            if y is None:
                return policy_loss

            card_loss = self.card_loss_fn(self._last_card_pred_log, y)

            # policy_loss can be a list (multi-GPU) or a tensor
            if isinstance(policy_loss, (list, tuple)):
                return [pl + self.card_loss_w * card_loss for pl in policy_loss]
            return policy_loss + self.card_loss_w * card_loss

    CustomModel.card_loss_w_default = effective_card_loss_w
    return CustomModel
