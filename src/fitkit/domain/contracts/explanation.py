"""The output side of the explanation boundary.

Prose lives here and nowhere else in the domain. `Explanation` is deliberately thin: it
carries the text, who produced it, and whether it is degraded -- because the degradation
flag is what makes C2 observable in production rather than merely true in principle.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Explanation:
    text: str
    renderer_id: str
    degraded: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("an explanation must not be empty")
        if not self.renderer_id.strip():
            raise ValueError("renderer_id must identify which renderer produced the text")
        object.__setattr__(self, "notes", tuple(self.notes))
