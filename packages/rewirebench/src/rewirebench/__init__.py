"""Shared machinery for rewire.it benchmarks.

The house rules in the repository README are enforced here rather than described:
`splits` refuses a grouping that does not match the stated claim, `metrics` refuses
an interval that too few independent groups can support, and `results` refuses a
result that omits coverage, timing or the group count.
"""
from rewirebench.splits import connected_components, naive_key_count, assign_groups
from rewirebench.metrics import precision_at_n, recall_at_n, point_metrics, paired_group_bootstrap
from rewirebench.results import BenchmarkResult, write_result

__all__ = [
    "connected_components", "naive_key_count", "assign_groups",
    "precision_at_n", "recall_at_n", "point_metrics", "paired_group_bootstrap",
    "BenchmarkResult", "write_result",
]

# Local execution and explicit reviewed contribution workflow.
from rewirebench.sdk import prepare, run, evaluate, export
from rewirebench.submission import submit
__all__ += ["prepare", "run", "evaluate", "export", "submit"]
