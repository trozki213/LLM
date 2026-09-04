"""Fixed-node Gauss-Hermite quadrature over a standard normal.

ADR-006: uncertainty is propagated by deterministic quadrature, not Monte Carlo. The
nodes and weights below are compile-time constants, so `assess()` is bit-reproducible
for identical inputs -- a seeded RNG inside a "deterministic engine" is a trap for the
next maintainer.

Nodes are the probabilists' form (z = sqrt(2) x) with weights normalised to sum to 1,
so `sum(w * f(z))` approximates E[f(Z)] for Z ~ N(0, 1).
"""

from __future__ import annotations

from typing import Final, Sequence

#: Outer grid, over the shared scale factor. Five nodes: the correlated component moves
#: every region together, so it decides the verdict and deserves the finer grid.
SCALE_NODES: Final[Sequence[tuple[float, float]]] = (
    (-2.856970013872805, 0.011257411327721),
    (-1.355626179974266, 0.222075922005613),
    (0.0, 0.533333333333333),
    (1.355626179974266, 0.222075922005613),
    (2.856970013872805, 0.011257411327721),
)

#: Inner grid, over each region's independent residual. Three nodes: independent error
#: partly averages out across regions, so it does not need the resolution the shared
#: component does.
RESIDUAL_NODES: Final[Sequence[tuple[float, float]]] = (
    (-1.732050807568877, 0.166666666666667),
    (0.0, 0.666666666666667),
    (1.732050807568877, 0.166666666666667),
)

