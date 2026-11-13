"""Template/state nearest-neighbor latency prediction, without regression fallback."""
import bisect
import datetime
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from shared.latency_gate import LatencyGate, template

PREDICATE = re.compile(r"\b(\w+)\.(\w+)\s*(<=|>=|=|<|>)\s*('(?:[^']|'')*'|[-+]?\d+(?:\.\d+)?)")


@lru_cache(maxsize=8192)
def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        try:
            return (datetime.datetime.fromisoformat(value) - datetime.datetime(1970, 1, 1)).total_seconds()
        except (ValueError, TypeError):
            return None


def _cdf(value, histogram):
    if value <= histogram[0]:
        return 0.
    if value >= histogram[-1]:
        return 1.
    i = bisect.bisect_right(histogram, value)
    lo, hi = histogram[i - 1], histogram[i]
    return (i - 1 + (value - lo) / (hi - lo)) / (len(histogram) - 1)


def _selectivity(stats, operator, literal, rows):
    histogram = [_number(v) for v in stats.get('histogram', [])]
    histogram = [v for v in histogram if v is not None and math.isfinite(v)]
    value = _number(literal)
    common = stats.get('mcv', [])
    frequencies = stats.get('frequencies', [])
    residual = max(0., 1 - stats.get('null_frac', 0.) - sum(frequencies))
    if operator == '=':
        if literal in common:
            return frequencies[common.index(literal)], False
        outside = bool(histogram and value is not None and (value < histogram[0] or value > histogram[-1]))
        ndv = stats.get('ndv', 0.)
        ndv = ndv if ndv >= 0 else -ndv * rows
        return (0. if outside else residual / max(1., ndv - len(common))), outside
    if value is None or len(histogram) < 2:
        return None, False
    fraction = _cdf(value, histogram)
    compare = {'<': lambda x: x < value, '<=': lambda x: x <= value,
               '>': lambda x: x > value, '>=': lambda x: x >= value}[operator]
    mass = sum(f for v, f in zip(common, frequencies) if _number(v) is not None and compare(_number(v)))
    fraction = (fraction if operator in ('<', '<=') else 1 - fraction) * residual + mass
    outside = (value < histogram[0] if operator in ('<', '<=') else value > histogram[-1]) and mass == 0
    return fraction, outside


def build_signature(dataset, sql, plan, table_stats, column_stats=None, date_stats=None):
    """Build execution-free state features; statistics are caller-managed snapshots.

    plan is a PG JSON Plan object. table_stats maps relation to
    [estimated rows, pages, dead tuples, modifications]. column_stats uses
    relation.column keys; date_stats uses the existing TPC-DS years/columns format.
    """
    dataset = dataset.upper()
    if dataset not in ('JOB', 'STATS', 'TPCDS'):
        raise ValueError('Unsupported dataset: ' + dataset)
    if isinstance(plan, list):
        if len(plan) != 1:
            raise ValueError('Expected one EXPLAIN result')
        plan = plan[0]
    if 'Plan' in plan:
        plan = plan['Plan']
    cost = float(plan['Total Cost'])
    if not math.isfinite(cost) or cost < 0:
        raise ValueError('Plan cost must be finite and nonnegative')
    counts, relations = Counter(), set()
    def walk(node):
        counts[node['Node Type']] += 1
        if 'Relation Name' in node:
            relations.add(node['Relation Name'])
        for child in node.get('Plans', []):
            walk(child)
    walk(plan)
    vector = {'cost': math.log1p(cost)}
    vector.update({'operator_' + kind: float(n) for kind, n in counts.items()})
    for relation in relations:
        state = table_stats.get(relation, [0., 0.])
        vector['rows_' + relation] = math.log1p(max(0., state[0]))
        vector['pages_' + relation] = math.log1p(max(0., state[1]))
    empty, coverage = {}, {}
    if dataset == 'TPCDS':
        years = re.findall(r'\bd_year\s*=\s*(\d{4})', sql.lower())
        if years and date_stats:
            bounds = date_stats['years'].get(years[0])
            for relation in relations:
                stats = date_stats['columns'].get(relation)
                if not stats:
                    continue
                # Missing statistics is unknown, not proof of empty support.
                histogram = stats.get('histogram', [])
                common = stats.get('mcv', [])
                if not histogram and not common:
                    continue
                fraction = 0.
                if bounds is not None:
                    lo, hi = bounds
                    if len(histogram) > 1:
                        fraction = max(0., _cdf(hi + .5, histogram) - _cdf(lo - .5, histogram))
                    fraction *= max(0., 1 - stats.get('null_frac', 0.) - sum(stats['frequencies']))
                    fraction += sum(f for v, f in zip(common, stats['frequencies']) if lo <= v <= hi)
                empty[relation] = fraction <= 1e-8
                vector['coverage_' + relation] = math.log1p(100 * fraction)
    else:
        aliases = {alias.lower(): relation.lower() for relation, alias in re.findall(r'\b(\w+)\s+as\s+(\w+)', sql, re.I)}
        for alias, column, operator, literal in PREDICATE.findall(sql):
            relation = aliases.get(alias.lower())
            stats = (column_stats or {}).get(str(relation) + '.' + column.lower())
            if stats is None:
                continue
            value = literal[1:-1].replace("''", "'") if literal.startswith("'") else literal
            fraction, outside = _selectivity(stats, operator, value, table_stats.get(relation, [0.])[0])
            if fraction is None:
                continue
            coverage[relation] = coverage.get(relation, 1.) * fraction
            empty[relation] = empty.get(relation, False) or outside
        vector.update({'selectivity_' + relation: math.log1p(100 * v) for relation, v in coverage.items()})
    return dict(cost=cost, vector=vector, empty=empty)


class NearestLatencyTAG:
    """JSON-serializable training exemplar bank plus a training-only TAG rule."""
    VERSION = 1

    def __init__(self, dataset, k=3, exponent=.5):
        self.dataset = dataset.upper()
        if self.dataset not in ('JOB', 'STATS', 'TPCDS'):
            raise ValueError('Unsupported dataset')
        if int(k) != k or k < 1 or not math.isfinite(exponent) or exponent < 0:
            raise ValueError('Invalid neighbor parameters')
        self.k, self.exponent = int(k), float(exponent)
        self.samples, self.gate = [], None

    @staticmethod
    def _validate_signature(signature):
        cost = float(signature['cost'])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError('Invalid plan cost')
        if not signature['vector'] or any(not math.isfinite(float(v)) for v in signature['vector'].values()):
            raise ValueError('Invalid feature vector')

    def fit(self, samples, training_databases, iteration=0, gate_rows=None):
        allowed = set(training_databases)
        if not samples or not allowed:
            raise ValueError('Empty training set')
        self.samples = []
        for row in samples:
            if row.get('phase') != 'train' or row['database'] not in allowed:
                raise ValueError('Only explicit training records may be fitted')
            value = float(row['baseline_exec'])
            if value <= 0 or not math.isfinite(value):
                raise ValueError('Training latency must be positive and finite')
            self._validate_signature(row['signature'])
            self.samples.append(dict(database=row['database'], query=row['query'],
                                     template=str(row.get('template', template(row['query']))),
                                     baseline_ms=value, signature=row['signature']))
        self.gate = LatencyGate()
        self.gate.fit(samples if gate_rows is None else gate_rows, allowed, iteration)
        return self

    def predict(self, query, signature, template_id=None):
        if not self.samples:
            raise ValueError('Model is not fitted')
        self._validate_signature(signature)
        group = str(template_id) if template_id is not None else template(query)
        candidates = [r for r in self.samples if r['query'] == query and r['template'] == group]
        level = 'query'
        if not candidates:
            candidates = [r for r in self.samples if r['template'] == group]
            level = 'template'
        if not candidates:
            candidates, level = self.samples, 'global'
        matched = [r for r in candidates if r['signature'].get('empty', {}) == signature.get('empty', {})]
        state_match = bool(matched)
        candidates = matched or candidates
        def distance(row):
            other = row['signature']['vector']
            keys = set(signature['vector']) | set(other)
            return sum((signature['vector'].get(key, 0.) - other.get(key, 0.)) ** 2 for key in keys) / len(keys)
        nearest = sorted(((distance(row), row) for row in candidates), key=lambda x: x[0])[:self.k]
        weights = [1 / (d + .01) for d, _ in nearest]
        values = [math.log(row['baseline_ms']) + self.exponent * math.log(max(signature['cost'], 1.) / max(row['signature']['cost'], 1.))
                  for _, row in nearest]
        predicted = math.exp(max(-20., min(30., sum(w * v for w, v in zip(weights, values)) / sum(weights))))
        return dict(predicted_ms=predicted, optimize=self.gate.allows(query, predicted),
                    match_level=level, state_match=state_match, distance=nearest[0][0],
                    neighbors=[dict(database=r['database'], query=r['query'], distance=d) for d, r in nearest])

    def save(self, path):
        Path(path).write_text(json.dumps(dict(version=self.VERSION, dataset=self.dataset, k=self.k,
            exponent=self.exponent, samples=self.samples, gate=self.gate.to_dict()), indent=2) + '\n')

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data['version'] != cls.VERSION:
            raise ValueError('Unsupported model version')
        model = cls(data['dataset'], data['k'], data['exponent'])
        model.samples = data['samples']
        if not model.samples:
            raise ValueError('Empty model')
        for row in model.samples:
            cls._validate_signature(row['signature'])
            if not math.isfinite(row['baseline_ms']) or row['baseline_ms'] <= 0:
                raise ValueError('Invalid model latency')
        model.gate = LatencyGate.from_dict(data['gate'])
        return model
