import torch
import numpy as np
from gymnasium import spaces
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2 as RLlibTorchModel
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.torch_utils import FLOAT_MIN
from shared.config import Config
import json  # JSON support.
import os
config = Config()
config.ConfirmPath()
DB_STATE_FEATURE_KEYS = {
    'db_hist',
    'db_bitmap',
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
    with open(param_config.auto_config, 'r', encoding='utf-8') as json_file:  # Open the file with UTF-8 encoding.
        data = json.load(json_file)  # Load JSON data.
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
        self.db_bitmap_dim = int(getattr(param_config, 'db_bitmap_dim', 0))
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
        self.bitmap_proj = nn.Linear(self.db_bitmap_dim, hidden_dim) if self.db_bitmap_dim > 0 else None
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

        if self.bitmap_proj is not None and 'db_bitmap' in self.kv_feature_mask:
            db_bitmap = self._as_2d(feature, 'db_bitmap', batch_size, self.db_bitmap_dim, device)
            tokens.append(self.bitmap_proj(db_bitmap).unsqueeze(1))

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
# Feature embeddings.
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
        # Define embedding layers.
        self.typeEmbed      = nn.Embedding(types, embed_size)
        self.tableEmbed     = nn.Embedding(tables, embed_size)
        self.columnEmbed    = nn.Embedding(columns, 2 * embed_size)
        self.opEmbed        = nn.Embedding(ops, embed_size // 8)
        self.dtypeEmbed     = nn.Embedding(4, embed_size // 4)
        self.posEmbed       = nn.Embedding(pos, embed_size // 8)
        # Define linear layers.
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
        # Split features.
        typeId, join, filtersId, filtersMask, histId, posId,table,db_est = torch.split(
                feature, (1, self.maxjoins, self.filtmaxnum*4, self.filtmaxnum, self.node_hist_dim, 1, 1, 4), dim = -1)
        typeEmb     = self.getType(typeId)
        joinEmb     = self.getJoin(join)
        filterEmbed = self.getFilter(filtersId, filtersMask)
        histEmb     = self.getHist(histId)
        dbest       = self.linearest(db_est)
        tableEmb    = self.getTable(table)
        posEmb      = self.getPos(posId)
        # Concatenate embeddings.
        # final       = torch.cat((typeEmb, joinEmb, tableEmb, posEmb, dbest),dim=1)
        final       = torch.cat((typeEmb, filterEmbed, joinEmb, histEmb, tableEmb, posEmb, dbest),dim=1)
        temp        = self.project(final)
        final       = F.leaky_relu(temp)       
        return final

    def getType(self, typeId):
        # Get type embeddings.
        typeId = typeId.long()
        emb = self.typeEmbed(typeId).squeeze(1)
        return emb

    def getTable(self, table_sample):
        # Get table embeddings.
        table = table_sample.long()
        # print(f'table: {table.max().item()}, tableEmbed.num_embeddings: {self.tableEmbed.num_embeddings}')
        emb = self.tableEmbed(table).squeeze(1)
        return emb
    
    def getJoin(self, joins):
        # Get join embeddings.
        joins = joins.long()
        joins_embed = self.columnEmbed(joins)
        joins_embed = torch.cat([joins_embed[:, i, :] for i in range(self.maxjoins)], dim=-1)
        concat = F.leaky_relu(self.linearJoin1(joins_embed))
        concat = F.leaky_relu(self.linearJoin2(concat))
        concat = concat.squeeze(1)
        return concat
    
    def getPos(self, posId):  
        # Get position embeddings.
        posId = posId.long()
        if posId.max().item() >= self.posEmbed.num_embeddings:
            raise ValueError("Index out of range in posEmbed")
   
        emb = self.posEmbed(posId).squeeze(1)
        return emb

    def getHist(self, histId):
        # Get node-level histogram embeddings.
        histId = histId.float()
        return self.histProject(histId)
    
    def getFilter(self, filtersId, filtersMask):
        # Get filter embeddings.
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
        # Average filter embeddings.
        num_filters = torch.sum(filtersMask, dim=1) + 1e-10
        total = torch.sum(concat, dim=1)
        avg = total / num_filters.view(-1, 1)
        return avg
    
# Plan network.
class PlanNetwork(nn.Module):
    def __init__(self, param_config, emb_size = config.emb_size ,ffn_dim = config.ffn_dim, \
                 head_size = config.head_size, dropout = config.dropout, \
                 attention_dropout_rate = config.dropout, n_layers = config.num_layers):

        super(PlanNetwork, self).__init__()
        self.global_context_disabled = bool(getattr(param_config, 'ablation_disable_global_context', False))
        if os.path.exists(param_config.auto_config):
            data = GetParam(param_config)
            if data["AutoGetParam"]==1:
                config.heightsize = data["heightsize"]
                config.num_node_feature = data["num_node_feature"]
                config.max_logical_edges = data["max_logical_edges"]
                config.node_hist_dim = data.get("node_hist_dim", getattr(config, "node_hist_dim", 0))
                config.db_hist_dim = data["db_hist_dim"]
                config.db_bitmap_dim = data["db_bitmap_dim"]
                config.query_feat_dim = data["query_feat_dim"]

        self.hidden_dim     = config.hidden_dim
        self.head_size      = head_size
        self.emb_size       = emb_size
        self.height_size    = emb_size // 2
        # self.structure_size = emb_size // 2
        self.height_encoder = nn.Embedding(config.heightsize, self.height_size , padding_idx=0)
        # self.structure_encoder = nn.Embedding(config.structuresize, self.structure_size , padding_idx=0)
        self.input_dropout  = nn.Dropout(dropout)
        # Define encoder layers.
        encoders = [
            EncoderLayer(self.hidden_dim, ffn_dim, dropout, attention_dropout_rate,
                         head_size) for _ in range(n_layers)
        ]
        self.layers         = nn.ModuleList(encoders)
        self.final_ln       = nn.LayerNorm(self.hidden_dim)
        self.embbed_layer   = FeatureEmbed(param_config, embed_size = emb_size)
        # ===== Optional: drift features (DB-level + query-level + DB-meta) =====
        # 1) DB-level features (histograms / sample bitmaps) + lightweight query features
        # PlanHelper injects db_hist, db_bitmap and query_feat into observations.
        self.use_db_features = bool(getattr(config, "use_db_features", False))
        self.db_hist_dim = int(getattr(config, "db_hist_dim", 0))
        self.db_bitmap_dim = int(getattr(config, "db_bitmap_dim", 0))
        self.query_feat_dim = int(getattr(config, "query_feat_dim", 0))

        self.db_hist_proj = None
        self.db_bitmap_proj = None
        self.query_feat_proj = None

        if self.use_db_features:
            # Each branch maps -> hidden_dim so we can sum/broadcast and also concatenate for fuse_proj.
            if self.db_hist_dim > 0:
                self.db_hist_proj = nn.Sequential(
                    nn.Linear(self.db_hist_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                )
            if self.db_bitmap_dim > 0:
                self.db_bitmap_proj = nn.Sequential(
                    nn.Linear(self.db_bitmap_dim, self.hidden_dim),
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
        # PlanHelper injects table/database statistics and query table/column features.
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

        # Project concatenated features into hidden_dim so the shared representation learns drift signals.
        self.fuse_proj = None
        fuse_in = self.hidden_dim
        if self.db_hist_proj is not None:
            fuse_in += self.hidden_dim
        if self.db_bitmap_proj is not None:
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
        # Get attention bias and input features.
        attn_bias, x = batched_data['attn_bias'], batched_data['x']
        # Get height embeddings.
        heights = batched_data['heights'].long()

        n_batch, n_nodes = x.size()[:2]

        # Construct tree-attention bias.
        tree_attn_bias = attn_bias.clone()
        tree_attn_bias = tree_attn_bias.unsqueeze(1).repeat(1, self.head_size, 1, 1)
        tree_attn_bias = tree_attn_bias[:, :, 1:, 1:]  # [B, head, N, N]

        x_view = x.contiguous().view(-1, config.num_node_feature)
        node_feature = self.embbed_layer(x_view).view(
            n_batch, -1, self.hidden_dim - self.height_size
        )
        height_feature = self.height_encoder(heights)

        # Combine node and height features.
        node_feature = torch.cat([node_feature, height_feature], dim=2)  # [B, N, hidden_dim]

        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, tree_attn_bias)  # Apply an encoder layer.
        output = self.final_ln(output)  # [B, N, hidden_dim]

        global_emb = output[:, 0, :]  # Use token zero as the global representation; inputs must supply a CLS or root token.

        if use_global_drift_fusion and not self.global_context_disabled:
            drift_feature_mask = parse_feature_mask('all' if drift_feature_mask is None else drift_feature_mask)
            # ===== Fuse drift features (optional) =====
            # These features help the shared hidden layers learn data/query drift patterns.
            parts = [global_emb]
            ctx_total = torch.zeros_like(global_emb)

            # ---- db_hist / db_bitmap / query_feat ----
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

            if self.db_bitmap_proj is not None:
                if 'db_bitmap' in drift_feature_mask:
                    db_bitmap = batched_data.get("db_bitmap", None)
                    if db_bitmap is not None:
                        b = self.db_bitmap_proj(db_bitmap.to(global_emb.device).float())
                    else:
                        b = torch.zeros_like(global_emb)
                else:
                    b = torch.zeros_like(global_emb)
                parts.append(b)
                ctx_total = ctx_total + b

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
                    db_full = db_out[:, 0, :]                # Database-wide CLS representation.

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

        # Infer padding from zero heights, then use all-zero node features as a fallback.
        mask1 = (heights != 0)
        mask2 = (x.abs().sum(dim=-1) != 0)
        node_mask = (mask1 & mask2)

        return global_emb, output, node_mask

# Feed-forward network.
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
        self.scale = att_size ** -0.5  # Scaling factor.

        # Define linear projections.
        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()  # Save the original query-tensor shape.

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # Compute attention.
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
            visible = attn_bias > 0
            # Padding query rows can be fully masked. Give them only a self-edge
            # so softmax remains finite while valid rows use a true hard mask.
            has_visible_key = visible.any(dim=-1, keepdim=True)
            diagonal = torch.eye(
                visible.size(-2), visible.size(-1),
                dtype=torch.bool, device=visible.device,
            ).view(1, 1, visible.size(-2), visible.size(-1))
            visible = visible | (~has_visible_key & diagonal)
            x = x.masked_fill(~visible, torch.finfo(x.dtype).min)

        x = torch.softmax(x, dim=3)  # Compute attention weights.
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.head_size * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size  # Require output and input shapes to match.
        return x

# Encoder layer.
class EncoderLayer(nn.Module):

    def __init__(self, hidden_size, ffn_size, dropout_rate,
                 attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)  # Self-attention layer normalization.
        self.self_attention = MultiHeadAttention(hidden_size,
                                                 attention_dropout_rate,
                                                 head_size)  # Self-attention module.
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)  # Feed-forward layer normalization.
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)  # Feed-forward network.
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, attn_bias)  # Self-attention.
        y = self.self_attention_dropout(y)
        x = x + y  # Residual connection.

        y = self.ffn_norm(x) 
        y = self.ffn(y)  # Feed-forward network.
        y = self.ffn_dropout(y)
        x = x + y  # Residual connection.
        return x


class LogicalJoinGraphEncoder(nn.Module):
    """Encode an alias-level logical join graph with shared table/column/op embeddings."""

    def __init__(self, param_config, hidden_dim, embed_size):
        super().__init__()
        self.global_context_disabled = bool(getattr(param_config, 'ablation_disable_global_context', False))
        self.hidden_dim = hidden_dim
        self.embed_size = embed_size
        self.db_table_feat_dim = int(getattr(param_config, 'db_table_feat_dim', 6))
        max_slots = int(getattr(param_config, 'max_tables_in_schema', 256))
        self.slot_embedding = nn.Embedding(max_slots, embed_size)
        self.relation_proj = nn.Sequential(
            nn.Linear(embed_size * 2 + self.db_table_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        edge_input_dim = 4 * embed_size + max(1, embed_size // 8)
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.message_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(2)
        ])
        self.relation_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(2)])
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _masked_mean(values, mask):
        weights = mask.to(values.device).float().unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _gather_nodes(nodes, indices):
        safe = indices.clamp(min=0, max=max(0, nodes.size(1) - 1))
        return torch.gather(nodes, 1, safe.unsqueeze(-1).expand(-1, -1, nodes.size(-1)))

    def forward(self, feature, feature_embed, batch_size, device):
        rel_table_ids = feature.get('logical_rel_table_ids')
        if rel_table_ids is None:
            rel_table_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        else:
            rel_table_ids = rel_table_ids.to(device).long()
            if rel_table_ids.dim() == 1:
                rel_table_ids = rel_table_ids.unsqueeze(0)
        batch_size, relation_slots = rel_table_ids.shape
        rel_mask = feature.get('logical_rel_mask')
        rel_mask = (
            (rel_table_ids > 0).float() if rel_mask is None
            else rel_mask.to(device).float().view(batch_size, relation_slots)
        )
        table_ids = rel_table_ids.clamp(min=0, max=feature_embed.tableEmbed.num_embeddings - 1)
        table_emb = feature_embed.tableEmbed(table_ids)
        slot_ids = torch.arange(relation_slots, device=device).clamp(max=self.slot_embedding.num_embeddings - 1)
        slot_emb = self.slot_embedding(slot_ids).unsqueeze(0).expand(batch_size, -1, -1)

        table_stats = torch.zeros(
            (batch_size, relation_slots, self.db_table_feat_dim), device=device, dtype=table_emb.dtype
        )
        db_table_stats = None if self.global_context_disabled else feature.get('db_table_stats')
        db_positions = feature.get('logical_rel_db_positions')
        if db_table_stats is not None and db_positions is not None:
            db_table_stats = db_table_stats.to(device).float()
            db_positions = db_positions.to(device).long()
            safe_pos = (db_positions - 1).clamp(min=0, max=max(0, db_table_stats.size(1) - 1))
            table_stats = torch.gather(
                db_table_stats,
                1,
                safe_pos.unsqueeze(-1).expand(-1, -1, db_table_stats.size(-1)),
            )
            table_stats = table_stats * (db_positions > 0).unsqueeze(-1)
        relations = self.relation_proj(torch.cat([table_emb, slot_emb, table_stats], dim=-1))
        relations = relations * rel_mask.unsqueeze(-1)

        edge_index = feature.get('logical_edge_index')
        edge_columns = feature.get('logical_edge_columns')
        edge_ops = feature.get('logical_edge_ops')
        edge_mask = feature.get('logical_edge_mask')
        if edge_index is None:
            edge_index = torch.zeros((batch_size, 1, 2), dtype=torch.long, device=device)
            edge_columns = torch.zeros((batch_size, 1, 2), dtype=torch.long, device=device)
            edge_ops = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            edge_mask = torch.zeros((batch_size, 1), dtype=torch.float, device=device)
        else:
            edge_index = edge_index.to(device).long()
            edge_columns = edge_columns.to(device).long()
            edge_ops = edge_ops.to(device).long()
            edge_mask = edge_mask.to(device).float()
        edge_columns = edge_columns.clamp(min=0, max=feature_embed.columnEmbed.num_embeddings - 1)
        edge_ops = edge_ops.clamp(min=0, max=feature_embed.opEmbed.num_embeddings - 1)
        left_col = feature_embed.columnEmbed(edge_columns[:, :, 0])
        right_col = feature_embed.columnEmbed(edge_columns[:, :, 1])
        op_emb = feature_embed.opEmbed(edge_ops)
        edges = self.edge_proj(torch.cat([left_col, right_col, op_emb], dim=-1))
        edges = edges * edge_mask.unsqueeze(-1)

        src = (edge_index[:, :, 0] - 1).clamp(min=0, max=max(0, relation_slots - 1))
        dst = (edge_index[:, :, 1] - 1).clamp(min=0, max=max(0, relation_slots - 1))
        for message_layer, relation_norm in zip(self.message_layers, self.relation_norms):
            src_rel = self._gather_nodes(relations, src)
            dst_rel = self._gather_nodes(relations, dst)
            forward_message = message_layer(torch.cat([src_rel, dst_rel, edges], dim=-1))
            reverse_message = message_layer(torch.cat([dst_rel, src_rel, edges], dim=-1))
            mask3 = edge_mask.unsqueeze(-1)
            aggregate = torch.zeros_like(relations)
            degree = torch.zeros((batch_size, relation_slots, 1), device=device, dtype=relations.dtype)
            aggregate.scatter_add_(1, dst.unsqueeze(-1).expand_as(forward_message), forward_message * mask3)
            aggregate.scatter_add_(1, src.unsqueeze(-1).expand_as(reverse_message), reverse_message * mask3)
            degree.scatter_add_(1, dst.unsqueeze(-1), mask3)
            degree.scatter_add_(1, src.unsqueeze(-1), mask3)
            relations = relation_norm(relations + aggregate / degree.clamp(min=1.0))
            relations = relations * rel_mask.unsqueeze(-1)

        src_rel = self._gather_nodes(relations, src)
        dst_rel = self._gather_nodes(relations, dst)
        edges = self.edge_update(torch.cat([src_rel, dst_rel, edges], dim=-1)) * edge_mask.unsqueeze(-1)
        graph_global = self.graph_proj(torch.cat([
            self._masked_mean(relations, rel_mask),
            self._masked_mean(edges, edge_mask),
        ], dim=-1))
        return graph_global, relations, edges, rel_mask, edge_index, edge_mask


class LogicalActionEncoder(nn.Module):
    """Build embeddings for the configured join-order, SET/operator, and STOP actions."""

    def __init__(self, param_config, hidden_dim, num_outputs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_outputs = num_outputs
        self.action_mode = getattr(param_config, 'action_mode', 'set_then_join_order')
        if self.action_mode not in ('set_then_join_order', 'join_order_and_operator'):
            raise ValueError(f'Unsupported action_mode: {self.action_mode}')
        max_slots = int(getattr(param_config, 'max_tables_in_schema', 256))
        self.position_embedding = nn.Embedding(max_slots, hidden_dim)
        self.action_type_embedding = nn.Embedding(3, hidden_dim)
        self.base_action_embedding = nn.Embedding(num_outputs, hidden_dim)
        self.set_action_size = int(getattr(param_config, 'set_size', 64)) if self.action_mode == 'set_then_join_order' else 0
        self.set_feature_width = len(getattr(param_config, 'set_hint_gucs', ()))
        if self.action_mode == 'set_then_join_order' and self.set_action_size != 2 ** self.set_feature_width:
            raise ValueError('set_size must cover every SET GUC bit combination')
        self.swap_proj = nn.Sequential(
            nn.Linear(hidden_dim * 7 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if self.action_mode == 'set_then_join_order':
            self.set_proj = nn.Sequential(
                nn.Linear(hidden_dim + self.set_feature_width, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.set_context_proj = nn.Sequential(
                nn.Linear(self.set_feature_width, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.target_operator_embedding = nn.Embedding(3, hidden_dim)
            self.operator_proj = nn.Sequential(
                nn.Linear(hidden_dim * 6 + 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.stop_embedding = nn.Parameter(torch.zeros(hidden_dim))
        relation_slots = None
        for candidate in range(1, max_slots + 1):
            secondary_actions = (
                self.set_action_size if self.action_mode == 'set_then_join_order'
                else 3 * (candidate - 1)
            )
            expected = candidate * (candidate - 1) // 2 + secondary_actions + 1
            if expected == num_outputs:
                relation_slots = candidate
                break
        self.relation_slots = relation_slots
        if relation_slots is not None:
            pairs = [(left, right) for left in range(relation_slots - 1)
                     for right in range(left + 1, relation_slots)]
            permutations = torch.arange(relation_slots).repeat(len(pairs), 1)
            for action_idx, (left, right) in enumerate(pairs):
                permutations[action_idx, left] = right
                permutations[action_idx, right] = left
            self.register_buffer('swap_left_positions', torch.tensor([p[0] for p in pairs], dtype=torch.long))
            self.register_buffer('swap_right_positions', torch.tensor([p[1] for p in pairs], dtype=torch.long))
            self.register_buffer('swap_permutations', permutations.long())
        else:
            self.register_buffer('swap_left_positions', torch.zeros((0,), dtype=torch.long))
            self.register_buffer('swap_right_positions', torch.zeros((0,), dtype=torch.long))
            self.register_buffer('swap_permutations', torch.zeros((0, 0), dtype=torch.long))
        if self.action_mode == 'set_then_join_order':
            set_bits = [
                [int(bit) for bit in bin(hint_set)[2:].zfill(self.set_feature_width)]
                for hint_set in range(self.set_action_size)
            ]
            self.register_buffer('set_bits', torch.tensor(set_bits, dtype=torch.float32))

    @staticmethod
    def _gather_relations(relations, relation_ids):
        safe = (relation_ids - 1).clamp(min=0, max=max(0, relations.size(1) - 1))
        gathered = torch.gather(
            relations, 1, safe.unsqueeze(-1).expand(-1, -1, relations.size(-1))
        )
        return gathered * (relation_ids > 0).unsqueeze(-1)

    @staticmethod
    def _pool_edges(edges, mask):
        weights = mask.float().unsqueeze(-1)
        return (edges * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _crossing_mask(order, join_position, edge_index, edge_mask):
        right = order[:, join_position]
        prefix = order[:, :join_position]
        src, dst = edge_index[:, :, 0], edge_index[:, :, 1]
        src_in_prefix = (src.unsqueeze(-1) == prefix.unsqueeze(1)).any(dim=-1)
        dst_in_prefix = (dst.unsqueeze(-1) == prefix.unsqueeze(1)).any(dim=-1)
        crosses = ((src == right.unsqueeze(1)) & dst_in_prefix) | ((dst == right.unsqueeze(1)) & src_in_prefix)
        return crosses & edge_mask.bool() & (right > 0).unsqueeze(1)

    def forward(self, feature, relations, edges, edge_index, edge_mask, plan_nodes=None):
        device = relations.device
        batch_size, relation_slots, hidden_dim = relations.shape
        order = feature.get('join_order_rel_ids')
        if order is None:
            order = torch.zeros((batch_size, relation_slots), dtype=torch.long, device=device)
        else:
            order = order.to(device).long().view(batch_size, relation_slots)
        action_ids = torch.arange(self.num_outputs, device=device)
        actions = self.base_action_embedding(action_ids).unsqueeze(0).expand(batch_size, -1, -1).clone()
        stop_action_idx = self.num_outputs - 1
        if self.relation_slots != relation_slots:
            # Non-DynaHint synthetic action spaces used by a few isolated tests have no
            # complete swap/operator layout.  They retain learned per-action identity.
            actions[:, stop_action_idx, :] += self.stop_embedding + self.action_type_embedding.weight[2]
            return actions

        swap_count = self.swap_left_positions.numel()
        src, dst = edge_index[:, :, 0], edge_index[:, :, 1]
        swapped_order = torch.gather(
            order.unsqueeze(1).expand(-1, swap_count, -1),
            2,
            self.swap_permutations.unsqueeze(0).expand(batch_size, -1, -1),
        )
        left_ids = order[:, self.swap_left_positions]
        right_ids = order[:, self.swap_right_positions]
        left_rel = self._gather_relations(relations, left_ids)
        right_rel = self._gather_relations(relations, right_ids)

        direct_mask = (
            ((src.unsqueeze(1) == left_ids.unsqueeze(-1)) & (dst.unsqueeze(1) == right_ids.unsqueeze(-1)))
            | ((src.unsqueeze(1) == right_ids.unsqueeze(-1)) & (dst.unsqueeze(1) == left_ids.unsqueeze(-1)))
        ) & edge_mask.bool().unsqueeze(1)
        direct_weight = direct_mask.float()
        direct_edges = torch.matmul(direct_weight, edges) / direct_weight.sum(dim=-1, keepdim=True).clamp(min=1.0)

        src_match = swapped_order.unsqueeze(2) == src.unsqueeze(1).unsqueeze(-1)
        dst_match = swapped_order.unsqueeze(2) == dst.unsqueeze(1).unsqueeze(-1)
        src_position = src_match.float().argmax(dim=-1)
        dst_position = dst_match.float().argmax(dim=-1)
        later_position = torch.maximum(src_position, dst_position)
        start_position = torch.maximum(
            self.swap_left_positions,
            torch.ones_like(self.swap_left_positions),
        ).view(1, swap_count, 1)
        end_position = self.swap_right_positions.view(1, swap_count, 1)
        valid_edge = edge_mask.bool().unsqueeze(1) & (src > 0).unsqueeze(1) & (dst > 0).unsqueeze(1)
        affected_mask = valid_edge & (later_position >= start_position) & (later_position <= end_position)
        affected_weight = affected_mask.float()
        affected_edges = torch.matmul(affected_weight, edges) / affected_weight.sum(dim=-1, keepdim=True).clamp(min=1.0)

        edge_steps = F.one_hot(later_position.clamp(max=relation_slots - 1), num_classes=relation_slots).bool()
        edge_steps = (edge_steps & valid_edge.unsqueeze(-1)).any(dim=2)
        positions = torch.arange(relation_slots, device=device).view(1, 1, relation_slots)
        affected_steps = (positions >= start_position) & (positions <= end_position)
        valid_steps = affected_steps & (swapped_order > 0)
        connected_steps = valid_steps & edge_steps
        cartesian_steps = valid_steps & ~edge_steps
        valid_step_count = valid_steps.float().sum(dim=-1).clamp(min=1.0)
        scalar = torch.stack([
            torch.log1p(affected_weight.sum(dim=-1)),
            connected_steps.float().sum(dim=-1) / valid_step_count,
            cartesian_steps.float().sum(dim=-1) / valid_step_count,
        ], dim=-1)
        pos_left = self.position_embedding(self.swap_left_positions).unsqueeze(0).expand(batch_size, -1, -1)
        pos_right = self.position_embedding(self.swap_right_positions).unsqueeze(0).expand(batch_size, -1, -1)
        action_type = self.action_type_embedding.weight[0].view(1, 1, -1).expand(batch_size, swap_count, -1)
        swap_semantic = self.swap_proj(torch.cat([
            left_rel, right_rel, direct_edges, affected_edges, pos_left, pos_right, action_type, scalar,
        ], dim=-1))
        if self.action_mode == 'set_then_join_order':
            current_set_bits = feature.get('set_hint_bits')
            if current_set_bits is None:
                current_set_bits = torch.ones(
                    (batch_size, self.set_feature_width), device=device, dtype=relations.dtype
                )
            else:
                current_set_bits = current_set_bits.to(device=device, dtype=relations.dtype).view(
                    batch_size, self.set_feature_width
                )
            swap_semantic = swap_semantic + self.set_context_proj(current_set_bits).unsqueeze(1)
        actions[:, :swap_count, :] += swap_semantic

        if self.action_mode == 'set_then_join_order':
            set_start = swap_count
            set_bits = self.set_bits.to(device).unsqueeze(0).expand(batch_size, -1, -1)
            set_type = self.action_type_embedding.weight[1].view(1, 1, -1).expand(
                batch_size, self.set_action_size, -1
            )
            set_semantic = self.set_proj(torch.cat([set_type, set_bits], dim=-1))
            actions[:, set_start:set_start + self.set_action_size, :] += set_semantic
        else:
            join_count = relation_slots - 1
            ordered_relations = self._gather_relations(relations, order)
            ordered_mask = (order > 0).float().unsqueeze(-1)
            cumulative_relations = (ordered_relations * ordered_mask).cumsum(dim=1)
            cumulative_counts = ordered_mask.cumsum(dim=1).clamp(min=1.0)
            prefix_mean = cumulative_relations[:, :join_count, :] / cumulative_counts[:, :join_count, :]
            right_rel = ordered_relations[:, 1:, :]
            src_position = (order.unsqueeze(1) == src.unsqueeze(-1)).float().argmax(dim=-1)
            dst_position = (order.unsqueeze(1) == dst.unsqueeze(-1)).float().argmax(dim=-1)
            later_position = torch.maximum(src_position, dst_position)
            join_positions = torch.arange(1, relation_slots, device=device).view(1, join_count, 1)
            crossing = (
                (later_position.unsqueeze(1) == join_positions)
                & edge_mask.bool().unsqueeze(1)
                & (src > 0).unsqueeze(1)
                & (dst > 0).unsqueeze(1)
            )
            crossing_weight = crossing.float()
            crossing_edges = torch.matmul(crossing_weight, edges) / crossing_weight.sum(
                dim=-1, keepdim=True
            ).clamp(min=1.0)
            crossing_count = crossing_weight.sum(dim=-1)
            operator_scalar = torch.stack([
                torch.log1p(crossing_count), (crossing_count == 0).float(),
            ], dim=-1)
            if plan_nodes is None:
                plan_nodes = relations.new_zeros((batch_size, 1, hidden_dim))
            join_node_ids = feature.get('physical_join_node_indices')
            if join_node_ids is None:
                join_node_ids = torch.zeros((batch_size, join_count), dtype=torch.long, device=device)
            else:
                join_node_ids = join_node_ids.to(device).long()[:, :join_count]
            safe_node = (join_node_ids - 1).clamp(min=0, max=max(0, plan_nodes.size(1) - 1))
            plan_node = torch.gather(
                plan_nodes, 1, safe_node.unsqueeze(-1).expand(-1, -1, hidden_dim)
            ) * (join_node_ids > 0).unsqueeze(-1)
            action_type = self.action_type_embedding.weight[1].view(1, 1, 1, -1).expand(
                batch_size, join_count, 3, -1
            )
            target_ops = self.target_operator_embedding.weight.view(1, 1, 3, -1).expand(
                batch_size, join_count, -1, -1
            )
            def expand_ops(value):
                return value.unsqueeze(2).expand(-1, -1, 3, -1)
            operator_semantic = self.operator_proj(torch.cat([
                expand_ops(plan_node), expand_ops(prefix_mean), expand_ops(right_rel),
                expand_ops(crossing_edges), target_ops, action_type,
                expand_ops(operator_scalar),
            ], dim=-1))
            join_ids = torch.arange(join_count, device=device).view(join_count, 1)
            operator_ids = torch.arange(3, device=device).view(1, 3)
            action_indices = stop_action_idx - (3 * join_ids + operator_ids + 1)
            actions[:, action_indices.reshape(-1), :] += operator_semantic.reshape(
                batch_size, -1, hidden_dim
            )

        actions[:, stop_action_idx, :] = (
            actions[:, stop_action_idx, :]
            + self.stop_embedding
            + self.action_type_embedding.weight[2]
        )
        return actions

def create_custom_model(param_config, hid=256, card_loss_w=0.2):
    """Create an RLlib RLlibTorchModel with two branches:
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

    class CustomModel(RLlibTorchModel, nn.Module):
        def __init__(self, obs_space, action_space, num_outputs, model_config, name):
            RLlibTorchModel.__init__(self, obs_space, action_space, num_outputs, model_config, name)
            nn.Module.__init__(self)

            self.embedmodel = PlanNetwork(param_config)
            d = self.embedmodel.hidden_dim
            self.planner_data_fusion_mode = getattr(param_config, 'planner_data_fusion_mode', 'concat')
            if self.planner_data_fusion_mode not in ('concat', 'cross_attention'):
                raise ValueError("planner_data_fusion_mode must be 'concat' or 'cross_attention'")
            self.planner_cross_attn_enabled = (
                self.planner_data_fusion_mode == 'cross_attention'
                and not bool(getattr(param_config, 'ablation_disable_global_context', False))
            )
            self.planner_cross_attn_actor_only = bool(getattr(param_config, 'planner_cross_attn_actor_only', True))
            self.cross_attn_use_local_plan_q = bool(getattr(param_config, 'cross_attn_use_local_plan_q', True))
            self.cross_attn_q_feature_mask = getattr(param_config, 'cross_attn_q_feature_mask', 'all')
            self.actor_data_cross_attn = QueryConditionedDataAttention(
                param_config,
                d,
                dropout=float(getattr(param_config, 'dropout', 0.05)),
            ) if self.planner_cross_attn_enabled else None

            self.card_loss_w = effective_card_loss_w
            self.card_loss_fn = FinalCardLog1pHuberLoss(beta=1.0)

            self.logical_graph_encoder = LogicalJoinGraphEncoder(param_config, d, self.embedmodel.emb_size)
            self.logical_action_encoder = LogicalActionEncoder(param_config, d, num_outputs)
            self.graph_state_fuse = nn.Sequential(
                nn.Linear(d * 2, d),
                nn.GELU(),
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )
            self.action_scorer = nn.Sequential(
                nn.Linear(d * 2 + 1, hid),
                nn.LeakyReLU(),
                nn.Linear(hid, hid),
                nn.LeakyReLU(),
                nn.Linear(hid, 1),
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

            # Use query-conditioned data attention in the actor; retain concatenated critic features for PPO stability.
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
                critic_rep, plan_nodes, node_mask = self.embedmodel(plan_obs, return_nodes=True)  # [B, d]
                actor_rep = critic_rep

            graph_global, relation_embeddings, edge_embeddings, _, edge_index, edge_mask = (
                self.logical_graph_encoder(
                    plan_obs,
                    self.embedmodel.embbed_layer,
                    actor_rep.size(0),
                    actor_rep.device,
                )
            )
            share_actor_critic_rep = critic_rep is actor_rep
            actor_rep = self.graph_state_fuse(torch.cat([actor_rep, graph_global], dim=-1))
            if share_actor_critic_rep:
                critic_rep = actor_rep
            else:
                critic_rep = self.graph_state_fuse(torch.cat([critic_rep, graph_global], dim=-1))

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

            action_embeddings = self.logical_action_encoder(
                plan_obs,
                relation_embeddings,
                edge_embeddings,
                edge_index,
                edge_mask,
                plan_nodes,
            )
            state_for_actions = actor_rep_step.unsqueeze(1).expand(-1, action_embeddings.size(1), -1)
            logits = self.action_scorer(
                torch.cat([state_for_actions, action_embeddings], dim=-1)
            ).squeeze(-1)

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
