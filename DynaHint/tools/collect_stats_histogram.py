from __future__ import annotations

try:
    from DynaHint.tools.collect_histogram_common import extract_predicate_columns, run_collection
except ImportError:
    from collect_histogram_common import extract_predicate_columns, run_collection


DEFAULT_DATABASES = [
    'stats_201112',
    'stats_201206',
    'stats_201212',
    'stats_201306',
    'stats_201312',
    'stats'
]
DEFAULT_WORKLOAD_DIRS = [
    "./experiment/STATS/train",
    "./experiment/STATS/test",
]


if __name__ == "__main__":
    run_collection(DEFAULT_DATABASES, DEFAULT_WORKLOAD_DIRS)
