"""Building the residual table Phase 2 consumes.

This is why the harness is not merely a report: ADR-010 says uncertainty is measured,
and this is the measurement. A backend that has not been through here cannot ship,
because `ResidualTable` fails closed on an unknown backend.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

from fitkit.domain.regions import BodyRegion
from fitkit.measurement.residuals import ResidualEntry, ResidualTable
from fitkit.evaluation.records import GroundTruthSample

#: Below this many samples a bucket's residual is not a measurement, it is noise.
MIN_SAMPLES_PER_BUCKET = 8


def build_residual_table(
    samples: Iterable[GroundTruthSample],
    *,
    version: str,
    bucket_bounds_cm: Sequence[float] = (85.0, float("inf")),
    scale_sigma_rel: float = 0.0,
) -> ResidualTable:
    """Fit a residual per backend, region and body-shape bucket.

    The shared scale component is removed first: a residual table that absorbed the
    calibration error would double-count it when the two are recombined downstream.
    """
    grouped: dict[tuple[str, BodyRegion, float], list[float]] = defaultdict(list)
    for sample in samples:
        bound = next(b for b in bucket_bounds_cm if sample.estimated_cm <= b)
        grouped[(sample.backend_id, sample.region, bound)].append(sample.error_cm)

    entries = []
    for (backend_id, region, bound), errors in grouped.items():
        if len(errors) < MIN_SAMPLES_PER_BUCKET:
            continue
        total = _rms(errors)
        shared = scale_sigma_rel * _mean(abs(e) for e in errors)
        residual = math.sqrt(max(total**2 - shared**2, 1e-6))
        entries.append(ResidualEntry(backend_id, region, bound, round(residual, 3)))

    if not entries:
        raise ValueError(
            f"no bucket reached {MIN_SAMPLES_PER_BUCKET} samples; the panel is too small to "
            "characterise this backend, and guessing a residual is what ADR-010 forbids"
        )
    return ResidualTable(version=version, entries=tuple(entries))


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values) / len(values))


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
