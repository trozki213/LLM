"""Where metric scale comes from.

C3: declared height today, device depth later, a reference object as a hedge. Each is an
implementation of one Protocol, so adding one touches nothing downstream -- backends
consume a `ScaleCalibration`, never the thing that produced it.
"""

from __future__ import annotations

from fitkit.domain.body import ScaleCalibration
from fitkit.domain.capture import CaptureBundle
from fitkit.domain.errors import InvalidDeclaredHeight


class DeclaredHeightCalibration:
    """Scale from the height the shopper typed in, with the error that implies.

    The relative uncertainty is the declared height's own sigma over its value, so a
    self-report error of 1.5 cm on 175 cm becomes a ~0.86% scale error that then
    propagates, correlated, into every circumference.
    """

    source_id = "declared-height/1"

    def calibrate(self, bundle: CaptureBundle) -> ScaleCalibration:
        height = bundle.declared_height
        if height.value_cm <= 0:
            raise InvalidDeclaredHeight(f"declared height must be positive, got {height.value_cm}")
        return ScaleCalibration(
            source_id=self.source_id,
            sigma_rel=height.sigma_cm / height.value_cm,
            reference=height,
        )
