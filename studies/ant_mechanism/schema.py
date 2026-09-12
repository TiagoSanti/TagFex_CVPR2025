"""Stable identifiers and schema validation for ANT study artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "1.0.0"
COVERAGES = ("IV", "FS")
REFERENCES = ("GR", "AR")
BRANCHES = ("current", "kd")


@dataclass(frozen=True)
class ANTVariant:
    coverage: str
    reference: str
    detach: bool = False

    def __post_init__(self) -> None:
        if self.coverage not in COVERAGES:
            raise ValueError(f"coverage must be one of {COVERAGES}, got {self.coverage!r}")
        if self.reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, got {self.reference!r}")

    @property
    def name(self) -> str:
        suffix = "-D" if self.detach else ""
        return f"ANT-{self.coverage}-{self.reference}{suffix}"

    @property
    def symmetric_full(self) -> bool:
        return self.coverage == "FS"

    @property
    def max_global(self) -> bool:
        return self.reference == "GR"


ALL_ANT_VARIANTS = tuple(
    ANTVariant(coverage, reference, detach)
    for detach in (False, True)
    for coverage in COVERAGES
    for reference in REFERENCES
)


REQUIRED_METRIC_FIELDS = {
    "schema_version",
    "task",
    "epoch",
    "batch",
    "branch",
    "variant",
    "mode",
    "nce_loss",
    "ant_loss_raw",
    "ant_loss_adjusted",
    "active_ratio",
    "nce_grad_norm",
    "ant_grad_norm",
    "combined_grad_norm",
}


def validate_metric_record(record: dict[str, Any]) -> None:
    missing = REQUIRED_METRIC_FIELDS.difference(record)
    if missing:
        raise ValueError(f"ANT study record is missing fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {record['schema_version']!r}")
    if record["branch"] not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}, got {record['branch']!r}")
    if record["mode"] not in {"actual", "shadow"}:
        raise ValueError("mode must be 'actual' or 'shadow'")
