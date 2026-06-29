
from model import PlanNetwork, QueryConditionedDataAttention
import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader


class FinalCardHead(nn.Module):
    """Predict log1p(final_card) from a candidate plan embedding."""

    def __init__(self, d: int, hid: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hid),
            nn.LeakyReLU(),
            nn.Linear(hid, 1),
        )

    def forward(self, global_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(global_emb).squeeze(-1)


class RiskHead(nn.Module):
    """Predict whether a candidate is likely to be non-improving / risky."""

    def __init__(self, d: int, hid: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d, hid),
            nn.LeakyReLU(),
            nn.Linear(hid, 1),
        )

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(fused_emb).squeeze(-1)


class RelativeScoreHead(nn.Module):
    """Score a candidate with baseline/candidate/delta context."""

    def __init__(self, d: int, hid: int = 256, use_steps: bool = True):
        super().__init__()
        self.use_steps = use_steps
        self.base_in_dim = d * 4
        if use_steps:
            self.base_in_dim += 2
        self.mlp = nn.Sequential(
            nn.Linear(self.base_in_dim, hid),
            nn.LeakyReLU(),
            nn.Linear(hid, hid // 2),
            nn.LeakyReLU(),
            nn.Linear(hid // 2, 1),
        )

    @staticmethod
    def _ensure_step(steps: torch.Tensor, batch_size: int, device, dtype) -> torch.Tensor:
        if steps is None:
            return torch.zeros((batch_size, 1), device=device, dtype=dtype)
        s = steps.to(device).float()
        if s.dim() == 1:
            s = s.unsqueeze(-1)
        if s.size(0) == 1 and batch_size > 1:
            s = s.expand(batch_size, -1)
        return s

    def build_fused(self, cand_emb: torch.Tensor, base_emb: torch.Tensor, cand_steps: torch.Tensor = None,
                    base_steps: torch.Tensor = None) -> torch.Tensor:
        batch_size = cand_emb.size(0)
        fused = [cand_emb, base_emb, cand_emb - base_emb, cand_emb * base_emb]
        if self.use_steps:
            cand_step = self._ensure_step(cand_steps, batch_size, cand_emb.device, cand_emb.dtype)
            base_step = self._ensure_step(base_steps, batch_size, cand_emb.device, cand_emb.dtype)
            fused.extend([cand_step, cand_step - base_step])
        return torch.cat(fused, dim=-1)

    def forward(self, fused_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(fused_emb).squeeze(-1)


class PlanScorer(nn.Module):

    def __init__(self, param_config, hid_units: int = 256, use_steps: bool = True):
        super().__init__()
        self.embed = PlanNetwork(param_config)
        d = self.embed.hidden_dim
        self.data_fusion_mode = getattr(param_config, 'scorer_data_fusion_mode', 'concat')
        if self.data_fusion_mode not in ('concat', 'cross_attention'):
            raise ValueError("scorer_data_fusion_mode must be 'concat' or 'cross_attention'")
        self.cross_attn_enabled = self.data_fusion_mode == 'cross_attention'
        self.cross_attn_use_local_plan_q = bool(getattr(param_config, 'cross_attn_use_local_plan_q', True))
        self.cross_attn_q_feature_mask = getattr(param_config, 'cross_attn_q_feature_mask', 'all')
        self.cross_attn_q_use_query_feat = bool(getattr(param_config, 'cross_attn_q_use_query_feat', True))
        self.cross_attn_q_use_plan_nodes = bool(getattr(param_config, 'cross_attn_q_use_plan_nodes', True))
        self.data_cross_attn = QueryConditionedDataAttention(
            param_config,
            d,
            dropout=float(getattr(param_config, 'dropout', 0.05)),
        ) if self.cross_attn_enabled else None
        self.score_head = RelativeScoreHead(d, hid=hid_units, use_steps=use_steps)
        self.card_head = FinalCardHead(d, hid=hid_units)
        self.risk_head = RiskHead(self.score_head.base_in_dim, hid=hid_units)
        self.use_baseline_context = getattr(param_config, 'scorer_use_baseline_context', True)
        self.use_risk_head = bool(getattr(param_config, 'scorer_use_risk_head', True))

    def _split_feature(self, feature):
        if isinstance(feature, dict) and 'candidate' in feature:
            candidate = feature['candidate']
            baseline = feature.get('baseline', feature['candidate'])
        else:
            candidate = feature
            baseline = None
        return candidate, baseline

    @staticmethod
    def _zero_if_none(value, like):
        return torch.zeros_like(like) if value is None else value

    def _encode_local_plan_feature(self, feature):
        return self.embed(feature, return_nodes=True, use_global_drift_fusion=False)

    def _query_feat_token(self, feature, batch_size, device):
        if (not self.cross_attn_q_use_query_feat) or self.data_cross_attn.query_feat_proj is None:
            return None
        query_feat = self.data_cross_attn._as_2d(
            feature,
            'query_feat',
            batch_size,
            self.data_cross_attn.query_feat_dim,
            device,
        )
        return self.data_cross_attn.query_feat_proj(query_feat).unsqueeze(1)

    def _encode_comparative_features(self, candidate, baseline):
        cand_global, cand_nodes, cand_mask = self._encode_local_plan_feature(candidate)
        base_global, base_nodes, base_mask = self._encode_local_plan_feature(baseline)
        query_token_list = [
            cand_global,
            base_global,
            cand_global - base_global,
            cand_global * base_global,
        ]
        if self.cross_attn_q_use_plan_nodes:
            cand_ctx = self._zero_if_none(self.data_cross_attn._masked_mean(cand_nodes, cand_mask), cand_global)
            base_ctx = self._zero_if_none(self.data_cross_attn._masked_mean(base_nodes, base_mask), base_global)
            query_token_list.extend([cand_ctx, base_ctx, cand_ctx - base_ctx])
        query_tokens = torch.stack(query_token_list, dim=1)
        query_feat_token = self._query_feat_token(candidate, cand_global.size(0), cand_global.device)
        if query_feat_token is not None:
            query_tokens = torch.cat([query_tokens, query_feat_token], dim=1)
        data_memory = self.data_cross_attn.encode_data_memory(candidate, cand_global.size(0), cand_global.device)
        query_tokens = self.data_cross_attn.attend_query_tokens(query_tokens, data_memory)
        comparative_ctx = query_tokens.mean(dim=1)
        cand_final = self.data_cross_attn.out_ln(cand_global + query_tokens[:, 0, :] + comparative_ctx)
        base_final = self.data_cross_attn.out_ln(base_global + query_tokens[:, 1, :] + comparative_ctx)
        return cand_final, base_final

    def _encode_feature(self, feature):
        if self.cross_attn_enabled:
            plan_global, plan_nodes, node_mask = self.embed(
                feature,
                return_nodes=True,
                use_global_drift_fusion=not self.cross_attn_use_local_plan_q,
                drift_feature_mask=self.cross_attn_q_feature_mask,
            )
            return self.data_cross_attn(feature, plan_global, plan_nodes, node_mask)
        return self.embed(feature)

    def forward(self, feature):
        candidate, baseline = self._split_feature(feature)
        if (
            self.cross_attn_enabled
            and self.cross_attn_use_local_plan_q
            and baseline is not None
            and self.use_baseline_context
        ):
            cand_global, base_global = self._encode_comparative_features(candidate, baseline)
            base_steps = baseline.get('steps', None)
        else:
            cand_global = self._encode_feature(candidate)
            if baseline is not None and self.use_baseline_context:
                base_global = self._encode_feature(baseline)
                base_steps = baseline.get('steps', None)
            else:
                base_global = torch.zeros_like(cand_global)
                base_steps = None
        cand_steps = candidate.get('steps', None) if isinstance(candidate, dict) else None
        fused_emb = self.score_head.build_fused(cand_global, base_global, cand_steps, base_steps)
        score = self.score_head(fused_emb)
        card_pred_log = self.card_head(cand_global)
        if self.use_risk_head:
            risk_logit = self.risk_head(fused_emb)
        else:
            risk_logit = torch.zeros_like(score)
        return score, card_pred_log, risk_logit

    def get_embed(self, plan_feature):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            embeddings = self._encode_feature(plan_feature)
        if was_training:
            self.train()
        return embeddings


class MyDataset(Dataset):

    def __init__(self, inputs, labels, weights):
        self.inputs = inputs
        self.s_labels = labels[0]
        self.c_labels = labels[1]
        self.r_labels = labels[2] if len(labels) > 2 else [0.0 for _ in inputs]
        self.weights = weights

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.s_labels[idx], self.c_labels[idx], self.r_labels[idx], self.weights[idx]

    @property
    def labels(self):
        return self.s_labels


class FinalCardLog1pHuberLoss(nn.Module):
    """Huber loss on log1p(card)."""

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


class PointTrainer:
    def __init__(self, genConfig, device=None):
        self.config = genConfig
        self.alpha = self.config.alpha
        self.device = self.config.device if device is None else device

        self.seed = self.config.seed
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self._net = PlanScorer(genConfig).to(self.device)
        self.optimizer = torch.optim.Adam(self._net.parameters(), lr=self.config.scorer_lr)
        self.score_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.card_loss_fn = FinalCardLog1pHuberLoss(beta=1.0)
        self.risk_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.score_scale = 1.0
        self.card_scale = 1.0
        self.risk_scale = 1.0
        self.use_cardinality_aux_loss = bool(getattr(self.config, 'use_cardinality_aux_loss', True))
        self.use_risk_head = bool(getattr(self.config, 'scorer_use_risk_head', True))

    @staticmethod
    def _as_tensor(v, device):
        if isinstance(v, torch.Tensor):
            return v.to(device)
        if isinstance(v, np.ndarray):
            if not v.flags.writeable:
                v = v.copy()
            return torch.as_tensor(v, device=device)
        return torch.as_tensor(v, device=device)

    def _weighted_score_loss(self, pred_score, target_score, weights):
        loss = self.score_loss_fn(pred_score.float(), target_score.float())
        return (loss * weights.float()).mean()

    def _weighted_risk_loss(self, risk_logit, risk_label, weights):
        loss = self.risk_loss_fn(risk_logit.float(), risk_label.float())
        return (loss * weights.float()).mean()

    def fit(self, dataset, valdataset=None, testdataset=None, mybatch_size=128, epochs=15, writer=None,
            writer_prefix='Others/Scorer', global_step_base=0):
        dataloader = DataLoader(dataset, batch_size=mybatch_size, shuffle=True)
        self._net.train()
        warmup_epochs = 3
        history = {
            'total_loss': [],
            'score_loss': [],
            'card_loss': [],
            'risk_loss': [],
            'val_mse': [],
            'val_reward': [],
            'test_mse': [],
            'test_reward': [],
        }

        for epoch in range(epochs):
            cur_score_loss = 0.0
            cur_card_loss = 0.0
            cur_risk_loss = 0.0
            cur_total_loss = 0.0
            num_ = 0

            for batch_inputs, batch_score_labels, batch_cardinality_labels, batch_risk_labels, batch_weights in dataloader:
                batch_inputs = self._move_feature_to_device(batch_inputs)
                batch_weights = batch_weights.to(self.device).float()
                batch_score_labels = batch_score_labels.to(self.device).float()
                batch_cardinality_labels = batch_cardinality_labels.to(self.device)
                batch_risk_labels = batch_risk_labels.to(self.device).float()

                pred_score, card_pred_log, risk_logit = self._net(batch_inputs)
                score_loss = self._weighted_score_loss(pred_score, batch_score_labels, batch_weights)
                if self.use_cardinality_aux_loss:
                    card_loss = self.card_loss_fn(card_pred_log, batch_cardinality_labels)
                else:
                    card_loss = score_loss * 0.0
                if self.use_risk_head:
                    risk_loss = self._weighted_risk_loss(risk_logit, batch_risk_labels, batch_weights)
                else:
                    risk_loss = score_loss * 0.0

                cur_score_loss += score_loss.detach().cpu().item() * len(batch_score_labels)
                cur_card_loss += card_loss.detach().cpu().item() * len(batch_score_labels)
                cur_risk_loss += risk_loss.detach().cpu().item() * len(batch_score_labels)
                num_ += len(batch_score_labels)

                self.score_scale = 0.9 * self.score_scale + 0.1 * score_loss.detach().item()
                if self.use_cardinality_aux_loss:
                    self.card_scale = 0.9 * self.card_scale + 0.1 * card_loss.detach().item()
                if self.use_risk_head:
                    self.risk_scale = 0.9 * self.risk_scale + 0.1 * risk_loss.detach().item()
                normalized_score_loss = score_loss / (self.score_scale + 1e-8)
                normalized_card_loss = card_loss / (self.card_scale + 1e-8)
                normalized_risk_loss = risk_loss / (self.risk_scale + 1e-8)
                w = min(1.0, epoch / warmup_epochs)
                aux_loss = score_loss * 0.0
                if self.use_cardinality_aux_loss:
                    aux_loss = aux_loss + (1 - self.alpha) * normalized_card_loss * w
                if self.use_risk_head:
                    aux_loss = aux_loss + float(getattr(self.config, 'scorer_risk_loss_weight', 0.2)) * normalized_risk_loss
                total_loss = self.alpha * normalized_score_loss + aux_loss
                cur_total_loss += total_loss.detach().cpu().item() * len(batch_score_labels)

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            avg_total_loss = cur_total_loss / max(num_, 1)
            avg_score_loss = cur_score_loss / max(num_, 1)
            avg_card_loss = cur_card_loss / max(num_, 1)
            avg_risk_loss = cur_risk_loss / max(num_, 1)
            history['total_loss'].append(avg_total_loss)
            history['score_loss'].append(avg_score_loss)
            history['card_loss'].append(avg_card_loss)
            history['risk_loss'].append(avg_risk_loss)
            if writer is not None:
                global_step = global_step_base + epoch
                writer.add_scalar(f'{writer_prefix}/Total_Loss', avg_total_loss, global_step)
                writer.add_scalar(f'{writer_prefix}/Score_Loss', avg_score_loss, global_step)
                writer.add_scalar(f'{writer_prefix}/Card_Loss', avg_card_loss, global_step)
                writer.add_scalar(f'{writer_prefix}/Risk_Loss', avg_risk_loss, global_step)

            print(
                f"Epoch {epoch+1}/{epochs}, Total_Loss: {avg_total_loss:.6f}, "
                f"Score_Loss: {avg_score_loss:.6f}, Card_Loss: {avg_card_loss:.6f}, Risk_Loss: {avg_risk_loss:.6f}"
            )

            if valdataset is not None:
                valmse, valreward = self.test_dataset(valdataset)
                history['val_mse'].append(valmse)
                history['val_reward'].append(valreward)
                if writer is not None:
                    writer.add_scalar(f'{writer_prefix}/Reward_Val', valreward, global_step_base + epoch)
                    writer.add_scalar(f'{writer_prefix}/MSE_Val', valmse, global_step_base + epoch)

            if testdataset is not None:
                testmse, testreward = self.test_dataset(testdataset)
                history['test_mse'].append(testmse)
                history['test_reward'].append(testreward)
                if writer is not None:
                    writer.add_scalar(f'{writer_prefix}/Reward_Test', testreward, global_step_base + epoch)
                    writer.add_scalar(f'{writer_prefix}/MSE_Test', testmse, global_step_base + epoch)
        return history

    def test_dataset(self, dataset):
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        self._net.eval()
        total_sqerr = 0.0
        total_reward = 0.0
        total_count = 0

        for batch_inputs, batch_score_labels, _, _, batch_weights in dataloader:
            batch_inputs = self._move_feature_to_device(batch_inputs)
            with torch.no_grad():
                pred_score, _, _ = self._net(batch_inputs)
                diff = pred_score.cpu() - batch_score_labels.float()
                total_sqerr += torch.sum(diff * diff).item()
                reward = -torch.abs(diff) * batch_weights.float()
                total_reward += reward.sum().item()
                total_count += len(batch_score_labels)

        mse = total_sqerr / max(total_count, 1)
        avg_reward = total_reward / max(total_count, 1)
        print(f"Scorer MSE:{mse:.4f}  Average Reward:{avg_reward:.4f}")
        self._net.train()
        return mse, avg_reward

    def get_embed(self, plan_feature):
        return self._net.get_embed(plan_feature)

    def _move_feature_to_device(self, batch_inputs):
        if isinstance(batch_inputs, dict):
            return {k: self._move_feature_to_device(v) for k, v in batch_inputs.items()}
        return batch_inputs.to(self.device)

    def _prep_feature(self, feature):
        out = {}
        for k, v in feature.items():
            if k == 'action_mask':
                continue
            if isinstance(v, dict):
                out[k] = self._prep_feature(v)
                continue
            t = self._as_tensor(v, self.device)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            out[k] = t.unsqueeze(0)
        return out

    def _batch_plan_features(self, features):
        if len(features) == 0:
            return {}
        keys = list(features[0].keys())
        batch = {}
        for k in keys:
            if isinstance(features[0][k], dict):
                batch[k] = self._batch_plan_features([feature[k] for feature in features])
            else:
                batch[k] = torch.cat([feature[k] for feature in features], dim=0)
        return batch

    def _predict_components_from_prepped(self, prepped_features):
        if len(prepped_features) == 0:
            return [], [], []
        batch = self._batch_plan_features(prepped_features)
        with torch.no_grad():
            raw_scores, _, risk_logits = self._net(batch)
        risk_prob = torch.sigmoid(risk_logits)
        adjusted_scores = raw_scores - float(getattr(self.config, 'scorer_risk_inference_weight', 0.0)) * risk_prob
        return (
            adjusted_scores.detach().cpu().tolist(),
            raw_scores.detach().cpu().tolist(),
            risk_prob.detach().cpu().tolist(),
        )

    def _safe_select_enabled_for_phase(self, phase):
        if phase == 'train':
            return bool(getattr(self.config, 'safe_select_enable_train', False))
        if phase == 'test':
            return bool(getattr(self.config, 'safe_select_enable_test', getattr(self.config, 'safe_select_enable', False)))
        return bool(getattr(self.config, 'safe_select_enable', False))

    def _ensure_wrapped_feature(self, feature, baseline_feature=None):
        if isinstance(feature, dict) and 'candidate' in feature:
            return feature
        if baseline_feature is None:
            return {'candidate': feature, 'baseline': feature}
        return {'candidate': feature, 'baseline': baseline_feature}

    def predict_scores(self, features):
        self._net.eval()
        wrapped = [self._ensure_wrapped_feature(feature) for feature in features]
        prepped = [self._prep_feature(feature) for feature in wrapped]
        adjusted_scores, _, _ = self._predict_components_from_prepped(prepped)
        self._net.train()
        return adjusted_scores

    def predict_score(self, feature):
        self._net.eval()
        wrapped = self._ensure_wrapped_feature(feature)
        prepped = self._prep_feature(wrapped)
        with torch.no_grad():
            raw_score, card_pred_log, risk_logit = self._net(prepped)
        risk_prob = float(torch.sigmoid(risk_logit).view(-1)[0].detach().cpu().item())
        score_value = float(raw_score.view(-1)[0].detach().cpu().item())
        score_value -= float(getattr(self.config, 'scorer_risk_inference_weight', 0.0)) * risk_prob
        card_pred_value = float(card_pred_log.view(-1)[0].detach().cpu().item())
        self._net.train()
        return score_value, card_pred_value

    def compare_features(self, left, right):
        self._net.eval()
        left_prepped = self._prep_feature(self._ensure_wrapped_feature(left))
        right_prepped = self._prep_feature(self._ensure_wrapped_feature(right))
        left_scores, _, _ = self._predict_components_from_prepped([left_prepped])
        right_scores, _, _ = self._predict_components_from_prepped([right_prepped])
        left_score_value = float(left_scores[0])
        right_score_value = float(right_scores[0])
        score_margin = right_score_value - left_score_value
        better_idx = 1 if right_score_value > left_score_value else 0
        self._net.train()
        return better_idx, left_score_value, right_score_value, score_margin

    def compare_feature_probabilities(self, left, right):
        better_idx, _, _, score_margin = self.compare_features(left, right)
        prob_right = torch.sigmoid(torch.tensor(score_margin)).detach().cpu().item()
        return better_idx, [1.0 - prob_right, prob_right]

    def compare_feature_list(self, inputs):
        if len(inputs) == 0:
            return []
        better_indices = []
        self._net.eval()
        for inp in inputs:
            better_idx, _ = self.compare_feature_probabilities(inp['left'], inp['right'])
            better_indices.append(better_idx)
        self._net.train()
        return better_indices

    def predict_epi(self, hint_feature, phase='test'):
        return self.predict_group_with_metadata(hint_feature, phase=phase)

    def predict_epi_parallel(self, hint_feature, phase='test'):
        return self.predict_group_with_metadata(hint_feature, phase=phase)

    def predict_group(self, hint_feature):
        optimal_hint, optimal_feature, _ = self.predict_group_with_metadata(hint_feature)
        return optimal_hint, optimal_feature

    def predict_group_with_metadata(self, hint_feature, phase='test'):
        self._net.eval()
        hint_norepeat = []
        raw_features = []
        extras = []
        baseline_feature = None
        if len(hint_feature) > 0:
            baseline_feature = hint_feature[0][1]
        for item in hint_feature:
            hint, feature = item[0], item[1]
            extra = item[2] if len(item) >= 3 else {}
            if hint not in hint_norepeat:
                hint_norepeat.append(hint)
                raw_features.append(self._ensure_wrapped_feature(feature, baseline_feature=baseline_feature))
                extras.append(extra if isinstance(extra, dict) else {'planning_time_cum_ms': float(extra)})

        if len(raw_features) == 0:
            self._net.train()
            return '', {}, {
                'selected_score': None,
                'selected_adjusted_score': None,
                'selected_risk': None,
                'data_fusion_mode': getattr(self._net, 'data_fusion_mode', 'concat'),
                'cross_attn_enabled': bool(getattr(self._net, 'cross_attn_enabled', False)),
                'score_margin_top1_top2': 0.0,
                'candidate_scores': [],
                'selected_planning_time_cum_ms': 0.0,
            }

        preinputs = [self._prep_feature(feature) for feature in raw_features]
        adjusted_scores, raw_scores, risk_probs = self._predict_components_from_prepped(preinputs)
        score_array = np.asarray(adjusted_scores, dtype=np.float32)
        raw_optimal_idx = int(np.argmax(score_array))
        optimal_idx = raw_optimal_idx
        safe_fallback_to_baseline = False
        baseline_idx = 0
        if (
            self._safe_select_enabled_for_phase(phase)
            and len(score_array) > 1
            and baseline_idx < len(score_array)
        ):
            selected_risk = float(risk_probs[raw_optimal_idx])
            baseline_score = float(score_array[baseline_idx])
            selected_margin_vs_baseline = float(score_array[raw_optimal_idx] - baseline_score)
            if (
                raw_optimal_idx != baseline_idx
                and selected_risk >= float(getattr(self.config, 'safe_select_risk_threshold', 0.65))
                and selected_margin_vs_baseline < float(getattr(self.config, 'safe_select_min_adjusted_margin', 0.05))
                and baseline_score >= float(getattr(self.config, 'safe_select_baseline_score_eps', -0.02))
            ):
                optimal_idx = baseline_idx
                safe_fallback_to_baseline = True
        optimal_hint = hint_norepeat[optimal_idx]
        optimal_feature = {
            k: v.squeeze(0).cpu().numpy() if not isinstance(v, dict) else {
                kk: vv.squeeze(0).cpu().numpy() for kk, vv in v.items()
            }
            for k, v in preinputs[optimal_idx].items()
        }
        candidate_view = optimal_feature['candidate'] if 'candidate' in optimal_feature else optimal_feature
        sorted_scores = sorted([float(score) for score in adjusted_scores], reverse=True)
        score_margin_top1_top2 = 0.0 if len(sorted_scores) <= 1 else float(sorted_scores[0] - sorted_scores[1])
        selected_extra = extras[optimal_idx] if optimal_idx < len(extras) else {}
        metadata = {
            'selected_score': float(raw_scores[optimal_idx]),
            'selected_adjusted_score': float(adjusted_scores[optimal_idx]),
            'selected_risk': float(risk_probs[optimal_idx]),
            'raw_selected_score': float(raw_scores[raw_optimal_idx]),
            'raw_selected_adjusted_score': float(adjusted_scores[raw_optimal_idx]),
            'raw_selected_risk': float(risk_probs[raw_optimal_idx]),
            'safe_fallback_to_baseline': bool(safe_fallback_to_baseline),
            'safe_select_phase': phase,
            'data_fusion_mode': getattr(self._net, 'data_fusion_mode', 'concat'),
            'cross_attn_enabled': bool(getattr(self._net, 'cross_attn_enabled', False)),
            'score_margin_top1_top2': score_margin_top1_top2,
            'candidate_scores': [
                {
                    'hint': hint_norepeat[idx],
                    'score': float(raw_scores[idx]),
                    'adjusted_score': float(adjusted_scores[idx]),
                    'risk': float(risk_probs[idx]),
                }
                for idx in range(len(hint_norepeat))
            ],
            'selected_planning_time_cum_ms': float(selected_extra.get('planning_time_cum_ms', 0.0)),
        }
        self._net.train()
        return optimal_hint, candidate_view, metadata

    def retrainmodel(self):
        del self._net
        self._net = PlanScorer(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(self._net.parameters(), lr=self.config.scorer_lr)

    def save_model(self, model_path):
        model_dir = os.path.dirname(model_path)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        tmp_model_path = f"{model_path}.tmp.{os.getpid()}"
        try:
            with open(tmp_model_path, 'wb') as file_obj:
                torch.save(self._net.state_dict(), file_obj)
                file_obj.flush()
                os.fsync(file_obj.fileno())
            os.replace(tmp_model_path, model_path)
        finally:
            if os.path.exists(tmp_model_path):
                os.remove(tmp_model_path)

    def load_model(self, model_path):
        self._net.load_state_dict(torch.load(model_path, map_location=self.device))

    def try_load_model(self, model_path, load_tag='checkpoint'):
        if not os.path.exists(model_path):
            print(f"[Warning] safe load skipped: file missing | tag={load_tag} | path={model_path}")
            return False
        try:
            if os.path.getsize(model_path) <= 0:
                print(f"[Warning] safe load skipped: empty checkpoint | tag={load_tag} | path={model_path}")
                return False
        except OSError as exc:
            print(f"[Warning] safe load failed: {type(exc).__name__} | tag={load_tag} | path={model_path} | err={exc}")
            return False
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self._net.load_state_dict(state_dict)
            print(f"[Info] safe load success: updated scorer | tag={load_tag} | path={model_path}")
            return True
        except (EOFError, RuntimeError, OSError, ValueError, pickle.UnpicklingError) as exc:
            print(f"[Warning] safe load failed: {type(exc).__name__} | tag={load_tag} | path={model_path} | err={exc}")
            return False
