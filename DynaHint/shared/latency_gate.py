"""Execution-history gate fitted exclusively from explicit training records."""
import math
import re

# Infinity is an internal rejection signal, never a newly fitted final rule.
THRESHOLDS = [0, 50, 100, 200, 500, 1000, 2000, 5000, math.inf]


def template(query):
    match = re.search(r'\d+', str(query))
    return str(int(match[0])) if match else str(query)


class LatencyGate:
    def __init__(self, minimum=6):
        self.minimum = int(minimum)
        if self.minimum < 1:
            raise ValueError('Minimum template records must be positive')
        self.global_threshold = 0.
        self.templates = {}
        self.iteration = None
        self.databases = []

    @staticmethod
    def _threshold(rows):
        def loss(t):
            return (sum(r['optimized_exec'] + r['optimized_plan'] if r['baseline_exec'] >= t
                        else r['baseline_exec'] for r in rows), -t)

        threshold = min(THRESHOLDS, key=loss)
        if math.isinf(threshold):
            overhead = sum(r['optimized_plan'] for r in rows) / len(rows)
            # Positive-only benefit is a heuristic, not a safety guarantee.
            rho = sum(max(0., 1-r['optimized_exec']/r['baseline_exec'])
                      for r in rows) / len(rows)
            threshold = overhead / rho if rho > 0 else 300000.
        return threshold

    def fit(self, rows, training_databases, iteration):
        allowed = set(training_databases)
        if not rows:
            raise ValueError('Empty training calibration')
        groups = {}
        for row in rows:
            if row.get('phase') != 'train' or row['database'] not in allowed:
                raise ValueError('Non-training data supplied to latency gate')
            if any(not math.isfinite(float(row[k])) or float(row[k]) < 0
                   for k in ('baseline_exec', 'baseline_plan', 'optimized_exec', 'optimized_plan')):
                raise ValueError('Invalid calibration latency')
            if float(row['baseline_exec']) <= 0:
                raise ValueError('Baseline calibration latency must be positive')
            groups.setdefault(template(row['query']), []).append(row)
        self.global_threshold = self._threshold(rows)
        self.templates = {key: self._threshold(values) for key, values in groups.items()
                          if len(values) >= self.minimum}
        self.iteration = int(iteration)
        self.databases = sorted(allowed)

    def allows(self, query, baseline_ms):
        if baseline_ms is None or not math.isfinite(float(baseline_ms)) or baseline_ms <= 0:
            return True  # Missing estimates must not fabricate a fast-query classification.
        return baseline_ms >= self.templates.get(template(query), self.global_threshold)

    def to_dict(self):
        def encode(threshold):
            return None if math.isinf(threshold) else threshold

        return dict(version=1, minimum=self.minimum, iteration=self.iteration,
                    training_databases=self.databases, baseline_input='known_history_ms',
                    global_threshold_ms=encode(self.global_threshold),
                    template_thresholds_ms={k: encode(v) for k, v in self.templates.items()})

    @classmethod
    def from_dict(cls, value):
        if value['version'] != 1:
            raise ValueError('Unsupported gate version')
        def decode(threshold):
            return math.inf if threshold is None else float(threshold)

        gate = cls(value['minimum'])
        gate.global_threshold = decode(value['global_threshold_ms'])
        gate.templates = {k: decode(v) for k, v in value['template_thresholds_ms'].items()}
        gate.iteration = value['iteration']
        gate.databases = value['training_databases']
        return gate
