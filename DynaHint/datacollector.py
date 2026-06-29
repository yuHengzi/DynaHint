import os
import csv
import math
import pandas as pd
from collections import defaultdict


class DataColletor:
    def __init__(self, genConfig):
        self.planJsonPool = []
        self.planVecPool = {}
        self.planVecCurr = {}
        self.testPool = {}
        self.config = genConfig
        self.queryCardinality = {}
        self.queryCardinalityEvidence = defaultdict(lambda: defaultdict(int))

    def collect_testPool(self, queryID, feature, label, istimeout):
        if queryID not in self.testPool:
            self.testPool[queryID] = [[feature, label, istimeout]]
        else:
            self.testPool[queryID].append([feature, label, istimeout])

    def get_testData(self):
        testInputs = []
        testLabels = []
        testWeights = []
        for k, v in self.testPool.items():
            inputs, labels, weights = self.build_comparison_samples(v)
            testInputs.extend(inputs)
            testLabels.extend(labels)
            testWeights.extend(weights)
        return testInputs, testLabels, testWeights

    def clear_testPool(self):
        self.testPool.clear()

    def _is_valid_cardinality(self, cardinality):
        if cardinality is None:
            return False
        try:
            if pd.isna(cardinality):
                return False
        except Exception:
            pass
        return float(cardinality) >= 0

    def _canonical_cardinality(self, db_and_queryid):
        return self.queryCardinality.get(db_and_queryid, None)

    def _record_cardinality_evidence(self, db_and_queryid, cardinality):
        if not self._is_valid_cardinality(cardinality):
            return None
        card_key = int(cardinality) if float(cardinality).is_integer() else float(cardinality)
        self.queryCardinalityEvidence[db_and_queryid][card_key] += 1
        return card_key

    def _get_query_cardinality(self, db_and_queryid):
        canonical = self._canonical_cardinality(db_and_queryid)
        if canonical is not None:
            return canonical
        if db_and_queryid not in self.planVecPool:
            return None
        for record in self.planVecPool[db_and_queryid]:
            if self._is_valid_cardinality(record['cardinality']):
                canonical = self._record_cardinality_evidence(db_and_queryid, record['cardinality'])
                if canonical is not None:
                    self.queryCardinality[db_and_queryid] = canonical
                    return canonical
        return None

    def repair_planVecPool_cardinality(self, database, queryID, canonical_cardinality, source='runtime'):
        db_and_queryid = database + '|' + queryID
        if not self._is_valid_cardinality(canonical_cardinality):
            return 0
        if db_and_queryid not in self.planVecPool:
            return 0
        updated = 0
        mismatch = 0
        for idx, record in enumerate(self.planVecPool[db_and_queryid]):
            old_cardinality = record['cardinality']
            if old_cardinality != canonical_cardinality:
                if self._is_valid_cardinality(old_cardinality):
                    mismatch += 1
                self.planVecPool[db_and_queryid][idx]['cardinality'] = canonical_cardinality
                updated += 1
        if updated > 0:
            print(f"[Warning] planVecPool cardinality repaired: query={db_and_queryid}, source={source}, canonical={canonical_cardinality}, updated={updated}, mismatched_valid={mismatch}")
        return updated

    def _normalize_record(self, feature, label, istimeout, baseline_latency=None, step=None,
                          planning_time_cum=None, baseline_planning_time=None, db_and_queryid=None):
        latency, cardinality = label
        if step is None:
            step = int(round(float(feature.get('steps', [0])[0]) * self.config.maxsteps)) if 'steps' in feature else 0
        execution_latency = float(latency)
        baseline_execution_latency = None if baseline_latency is None else float(baseline_latency)
        planning_time_cum = 0.0 if planning_time_cum is None else float(planning_time_cum)
        baseline_planning_time = 0.0 if baseline_planning_time is None else float(baseline_planning_time)
        return {
            'feature': feature,
            'latency': execution_latency,
            'execution_latency': execution_latency,
            'cardinality': cardinality,
            'timeout': bool(istimeout),
            'baseline_latency': baseline_execution_latency,
            'baseline_execution_latency': baseline_execution_latency,
            'planning_time_cum': planning_time_cum,
            'baseline_planning_time': baseline_planning_time,
            'step': int(step),
            'db_and_queryid': db_and_queryid,
        }

    def _effective_latency(self, record):
        effective_latency = float(record.get('execution_latency', record['latency']))
        baseline_execution = record.get('baseline_execution_latency', record.get('baseline_latency'))
        if record.get('timeout', False) and baseline_execution is not None:
            effective_latency = max(effective_latency, float(baseline_execution) * self.config.timeoutcoeff)
        return effective_latency

    def _objective_value(self, record):
        effective_latency = self._effective_latency(record)
        if getattr(self.config, 'reward_use_total_time', False):
            return effective_latency + float(record.get('planning_time_cum', 0.0) or 0.0)
        return effective_latency

    def _baseline_objective_value(self, record):
        baseline_execution = record.get('baseline_execution_latency', record.get('baseline_latency'))
        if baseline_execution is None:
            return None
        if getattr(self.config, 'reward_use_total_time', False):
            return float(baseline_execution) + float(record.get('baseline_planning_time', 0.0) or 0.0)
        return float(baseline_execution)

    def _find_baseline_record(self, records):
        baseline_candidates = [record for record in records if int(record.get('step', 0)) == 0]
        if len(baseline_candidates) > 0:
            return min(baseline_candidates, key=lambda record: self._objective_value(record))
        fallback = [record for record in records if self._baseline_objective_value(record) is not None]
        if len(fallback) == 0:
            return None
        return min(fallback, key=lambda record: self._objective_value(record))

    def _wrap_pointwise_feature(self, record, baseline_record):
        baseline_feature = baseline_record['feature'] if baseline_record is not None else record['feature']
        return {
            'candidate': record['feature'],
            'baseline': baseline_feature,
        }

    def collect_planVecPool(self, database, queryID, feature, label, istimeout, baseline_latency=None, step=None,
                            planning_time_cum=None, baseline_planning_time=None):
        db_and_queryid = database + '|' + queryID
        canonical_cardinality = self._get_query_cardinality(db_and_queryid)
        latency, cardinality = label
        if self._is_valid_cardinality(cardinality):
            observed_cardinality = self._record_cardinality_evidence(db_and_queryid, cardinality)
            if canonical_cardinality is None:
                self.queryCardinality[db_and_queryid] = observed_cardinality
                self.repair_planVecPool_cardinality(database, queryID, observed_cardinality, source='new_valid_sample')
                canonical_cardinality = observed_cardinality
            elif canonical_cardinality != observed_cardinality:
                self.queryCardinality[db_and_queryid] = observed_cardinality
                self.repair_planVecPool_cardinality(database, queryID, observed_cardinality, source='new_valid_sample')
                canonical_cardinality = observed_cardinality
                cardinality = canonical_cardinality
            else:
                self.queryCardinality[db_and_queryid] = canonical_cardinality
                cardinality = canonical_cardinality
        elif canonical_cardinality is not None:
            cardinality = canonical_cardinality
        record = self._normalize_record(
            feature,
            (latency, cardinality),
            istimeout,
            baseline_latency=baseline_latency,
            step=step,
            planning_time_cum=planning_time_cum,
            baseline_planning_time=baseline_planning_time,
            db_and_queryid=db_and_queryid,
        )
        if db_and_queryid not in self.planVecPool:
            self.planVecPool[db_and_queryid] = [record]
        else:
            self.planVecPool[db_and_queryid].append(record)

    def collect_planJsonPool(self, queryno, planjson, label, istimeout):
        self.planJsonPool.append([queryno, planjson, label, istimeout])

    def backfill_baseline_info(self, baseline_info):
        if baseline_info is None:
            return 0
        updated = 0
        for db_and_queryid, records in self.planVecPool.items():
            base_meta = baseline_info.get(db_and_queryid, None)
            if base_meta is None:
                continue
            base_exec = float(base_meta.get('execution_time', 0.0))
            base_plan = float(base_meta.get('planning_time', 0.0))
            for record in records:
                if record.get('baseline_execution_latency', None) is None:
                    record['baseline_latency'] = base_exec
                    record['baseline_execution_latency'] = base_exec
                    updated += 1
                if float(record.get('baseline_planning_time', 0.0) or 0.0) == 0.0:
                    record['baseline_planning_time'] = base_plan
        if updated > 0:
            print(f"[Info] backfilled baseline info for pointwise samples: updated_records={updated}")
        return updated

    def get_featuresPool(self):
        features = []
        for db_queryId in self.planVecPool:
            db_queryId_len = len(self.planVecPool[db_queryId])
            for k in range(db_queryId_len):
                features.append(self.planVecPool[db_queryId][k]['feature'])
        features = pd.Series(features)
        return features

    def wirte_planJsonPool(self, path):
        if not os.path.exists(path):
            with open(path, mode='w', newline='', encoding='utf8') as cf:
                wf = csv.writer(cf)
                wf.writerow(['queryno', 'planjson', 'latency', 'istimeout'])
                for i in self.planJsonPool:
                    wf.writerow(i)
        else:
            with open(path, mode='a', newline='', encoding='utf8') as cfa:
                wf = csv.writer(cfa)
                for i in self.planJsonPool:
                    wf.writerow(i)
        self.planJsonPool.clear()

    def _build_pointwise_samples(self, records):
        inputs = []
        score_labels = []
        card_labels = []
        risk_labels = []
        weights = []
        eps = 1e-6
        baseline_record = self._find_baseline_record(records)
        if baseline_record is None:
            return inputs, score_labels, card_labels, risk_labels, weights
        baseline_objective = self._baseline_objective_value(baseline_record)
        if baseline_objective is None or baseline_objective <= 0:
            return inputs, score_labels, card_labels, risk_labels, weights
        record_objectives = []
        for idx, record in enumerate(records):
            candidate_objective = self._objective_value(record)
            record_objectives.append((idx, record, candidate_objective))
        if len(record_objectives) == 0:
            return inputs, score_labels, card_labels, risk_labels, weights
        best_objective = min(obj for _, _, obj in record_objectives)
        topk = max(int(getattr(self.config, 'distill_best_plan_topk', 0) or 0), 0)
        best_indices = {
            idx for idx, _, _ in sorted(record_objectives, key=lambda item: item[2])[:topk]
        }
        risk_margin = float(getattr(self.config, 'risk_negative_margin', 1.05))
        negative_weight = float(getattr(self.config, 'negative_sample_weight', 1.0))
        high_cost_log_base = max(float(getattr(self.config, 'high_cost_weight_log_base', 100.0)), eps)
        high_cost_weight_cap = max(float(getattr(self.config, 'high_cost_weight_cap', 1.0)), 1.0)
        high_cost_weight = 1.0 + min(
            math.log1p(baseline_objective) / math.log1p(high_cost_log_base),
            high_cost_weight_cap - 1.0,
        )
        for idx, record, candidate_objective in record_objectives:
            if candidate_objective <= 0:
                continue
            target_score = math.log((baseline_objective + eps) / (candidate_objective + eps))
            target_score = max(-4.0, min(4.0, target_score))
            card_label = record['cardinality'] if self._is_valid_cardinality(record['cardinality']) else -1
            is_baseline_record = int(record.get('step', -1)) == 0 and abs(candidate_objective - baseline_objective) <= eps
            risk_label = 1.0 if (not is_baseline_record and candidate_objective > baseline_objective * risk_margin) else 0.0
            sample_weight = (1.0 + abs(target_score)) * high_cost_weight
            if risk_label > 0.0:
                sample_weight *= negative_weight
            if getattr(self.config, 'dynamic_hard_weight', False):
                regret_ratio = max(candidate_objective - best_objective, 0.0) / max(baseline_objective, eps)
                sample_weight *= 1.0 + min(
                    regret_ratio,
                    float(getattr(self.config, 'dynamic_hard_weight_cap', 2.0)),
                )
            if idx in best_indices:
                sample_weight *= float(getattr(self.config, 'distill_best_plan_weight', 1.0))
            inputs.append(self._wrap_pointwise_feature(record, baseline_record))
            score_labels.append(target_score)
            card_labels.append(card_label)
            risk_labels.append(risk_label)
            weights.append(sample_weight)
        return inputs, score_labels, card_labels, risk_labels, weights

    def get_samples(self):
        inputs = []
        score_labels = []
        card_labels = []
        risk_labels = []
        weights = []
        db_sample_counts = defaultdict(int)
        db_payloads = {}
        for db_and_queryid, records in self.planVecPool.items():
            db = db_and_queryid.split('|')[0]
            query_inputs, query_scores, query_cards, query_risks, query_weights = self._build_pointwise_samples(records)
            if len(query_inputs) == 0:
                continue
            db_payloads.setdefault(db, {'inputs': [], 'scores': [], 'cards': [], 'risks': [], 'weights': []})
            db_payloads[db]['inputs'].extend(query_inputs)
            db_payloads[db]['scores'].extend(query_scores)
            db_payloads[db]['cards'].extend(query_cards)
            db_payloads[db]['risks'].extend(query_risks)
            db_payloads[db]['weights'].extend(query_weights)
            db_sample_counts[db] += len(query_inputs)
        if len(db_payloads) == 0:
            return inputs, (score_labels, card_labels, risk_labels), weights
        avg_db_samples = sum(db_sample_counts.values()) / max(len(db_sample_counts), 1)
        for db in sorted(db_payloads.keys()):
            payload = db_payloads[db]
            db_reweight = 1.0
            if getattr(self.config, 'balanced_db_sampling', False):
                db_reweight = avg_db_samples / max(db_sample_counts[db], 1)
            inputs.extend(payload['inputs'])
            score_labels.extend(payload['scores'])
            card_labels.extend(payload['cards'])
            risk_labels.extend(payload['risks'])
            weights.extend([weight * db_reweight for weight in payload['weights']])
        return inputs, (score_labels, card_labels, risk_labels), weights

    def build_comparison_samples_full(self, old_fea_latency):
        return [], [], [], []

    def build_comparison_samples(self, old_fea_latency):
        old_length = len(old_fea_latency)
        inputs = []
        labels = []
        weights = []
        if old_length == 1:
            return inputs, labels, weights

        to_save = []
        to_delete = []
        for i in range(old_length):
            if old_fea_latency[i][2]:
                to_delete.append(i)
            else:
                to_save.append(i)
        new_fea_latency = []

        for i in to_save:
            for j in to_delete:
                comparison_input = {'left': old_fea_latency[j][0], 'right': old_fea_latency[i][0]}
                latency_values = [old_fea_latency[j][1], old_fea_latency[i][1]]
                label = 0
                for l, p in enumerate(self.config.splitpoint):
                    if (latency_values[0] - latency_values[1]) / latency_values[0] >= p:
                        label = self.config.classNum - l
                        break
                if label != 0:
                    labels.append(label)
                    weights.append(math.log10(1 + (max(latency_values) - min(latency_values))))
                    inputs.append(comparison_input)
            new_fea_latency.append(old_fea_latency[i])

        new_length = len(new_fea_latency)
        for i in range(new_length - 1):
            for j in range(i + 1, new_length):
                comparison_input = {'left': new_fea_latency[i][0], 'right': new_fea_latency[j][0]}
                latency_values = [new_fea_latency[i][1], new_fea_latency[j][1]]
                ratio = (latency_values[0] - latency_values[1]) / latency_values[0]
                label = 0
                for l, p in enumerate(self.config.splitpoint):
                    if ratio >= p:
                        label = len(self.config.splitpoint) - l
                        break
                labels.append(label)
                inputs.append(comparison_input)
                weights.append(math.log10(1 + (max(latency_values) - min(latency_values))))
        return inputs, labels, weights
