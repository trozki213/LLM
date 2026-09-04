"""Per-region penalty and classification. Pure functions over floats."""

from __future__ import annotations

from fitkit.domain.contracts.fit_assessment import FitClassification
from fitkit.domain.policy import EaseWindow

#: Inside the acceptable window, distance from the preferred ease still costs something,
#: but an order of magnitude less than being outside it.
_INSIDE_WEIGHT = 0.25


def penalty(delta_cm: float, window: EaseWindow, tightness_ratio: float) -> float:
    """Cost of this much ease. Continuous, and asymmetric: too tight hurts more."""
    if delta_cm < window.min_cm:
        return _INSIDE_WEIGHT * (window.preferred_cm - window.min_cm) + tightness_ratio * (
            window.min_cm - delta_cm
        )
    if delta_cm > window.max_cm:
        return _INSIDE_WEIGHT * (window.max_cm - window.preferred_cm) + (delta_cm - window.max_cm)
    return _INSIDE_WEIGHT * abs(delta_cm - window.preferred_cm)


def classify(delta_cm: float, window: EaseWindow) -> FitClassification:
    """Where this ease sits relative to the window. A closed vocabulary, by construction."""
    span = max(window.max_cm - window.min_cm, 1e-9)
    outer = span / 2.0
    near_preferred = max((window.preferred_cm - window.min_cm) / 2.0, 1e-9)

    if delta_cm < window.min_cm - outer:
        return FitClassification.MUCH_TOO_TIGHT
    if delta_cm < window.min_cm:
        return FitClassification.TIGHT
    if delta_cm > window.max_cm + outer:
        return FitClassification.MUCH_TOO_LOOSE
    if delta_cm > window.max_cm:
        return FitClassification.LOOSE
    if abs(delta_cm - window.preferred_cm) <= near_preferred:
        return FitClassification.AS_INTENDED
    return FitClassification.SNUG if delta_cm < window.preferred_cm else FitClassification.RELAXED


def is_uncertain(delta_cm: float, sigma_cm: float, window: EaseWindow) -> bool:
    """True when one standard deviation is enough to change the verbal answer.

    This needs no threshold of its own: the question "would we say something different
    if we were one sigma out?" is already the thing a threshold would be approximating.
    """
    here = classify(delta_cm, window)
    return (
        classify(delta_cm - sigma_cm, window) is not here
        or classify(delta_cm + sigma_cm, window) is not here
    )
