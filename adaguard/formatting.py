"""Strict parser for AdAGuard's multi-label generation contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

LABEL_PATTERN = r"[A-Za-z][A-Za-z0-9_-]{0,31}"
NO_RISK_LABEL = "NR"


_FULL_RESPONSE_RE = re.compile(
    rf"\A<analysis>\n(?P<analysis>.*?)\n</analysis>\n"
    rf"<label>(?P<label>{LABEL_PATTERN}(?:,{LABEL_PATTERN})*)</label>\Z",
    flags=re.DOTALL,
)
_STRUCTURAL_TAG_RE = re.compile(r"</?(?:analysis|label)>", flags=re.IGNORECASE)


@dataclass(frozen=True)
class ParsedGuardResponse:
    valid: bool
    analysis: str
    labels: tuple[str, ...]
    raw_label: str | None
    error: str | None = None

    @property
    def label(self) -> str | None:
        """Backward-compatible textual label accessor."""

        return self.raw_label


def canonical_label_text(labels: Iterable[str]) -> str:
    values = tuple(labels)
    return NO_RISK_LABEL if not values else ",".join(values)


def parse_guard_response(value: object) -> ParsedGuardResponse:
    """Parse exact ``analysis`` + ordered set label output, without whitespace repair."""

    if not isinstance(value, str):
        return ParsedGuardResponse(False, "", (), None, "response_not_string")
    match = _FULL_RESPONSE_RE.fullmatch(value)
    if match is None:
        return ParsedGuardResponse(False, "", (), None, "response_schema_mismatch")
    analysis = match.group("analysis")
    raw_label = match.group("label")
    if not analysis.strip():
        return ParsedGuardResponse(False, "", (), raw_label, "analysis_empty")
    if _STRUCTURAL_TAG_RE.search(analysis):
        return ParsedGuardResponse(False, "", (), raw_label, "nested_structural_tag")
    raw_labels = tuple(raw_label.split(","))
    if len(raw_labels) != len(set(raw_labels)):
        return ParsedGuardResponse(False, analysis, (), raw_label, "duplicate_label")
    if NO_RISK_LABEL in raw_labels:
        if raw_labels != (NO_RISK_LABEL,):
            return ParsedGuardResponse(False, analysis, (), raw_label, "nr_mixed_with_risk")
        labels: tuple[str, ...] = ()
    else:
        labels = raw_labels
    return ParsedGuardResponse(True, analysis, labels, raw_label)


def parse_reference_target(value: object) -> tuple[str, tuple[str, ...]]:
    parsed = parse_guard_response(value)
    if not parsed.valid:
        raise ValueError(f"invalid reference target: {parsed.error}")
    return parsed.analysis, parsed.labels
