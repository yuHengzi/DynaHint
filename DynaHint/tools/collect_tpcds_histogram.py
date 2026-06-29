from __future__ import annotations

try:
    from DynaHint.tools.collect_histogram_common import extract_predicate_columns, run_collection
except ImportError:
    from collect_histogram_common import extract_predicate_columns, run_collection


DEFAULT_DATABASES = [
    "tpcds10g_199806",
    "tpcds10g_199812",
    "tpcds10g_199906",
    "tpcds10g_199912",
    "tpcds10g_200006",
    "tpcds10g_200012",
    "tpcds10g_200106",
    "tpcds10g_200112",
    "tpcds10g_200206",
    "tpcds10g",
]
DEFAULT_WORKLOAD_DIRS = [
    "./experiment/TPCDS/train",
    "./experiment/TPCDS/test",
]


if __name__ == "__main__":
    run_collection(DEFAULT_DATABASES, DEFAULT_WORKLOAD_DIRS)
