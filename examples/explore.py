"""A knob for every input that changes the answer, so you can watch it change.

    python examples/explore.py                      # a body between two sizes
    python examples/explore.py --waist 84 --hip 98
    python examples/explore.py --residual 2.5       # a noisier backend: watch it abstain
    python examples/explore.py --stretch high       # the same body, elastic fabric
    python examples/explore.py --preference looser

Nothing here is production code. It exists so the behaviour that matters -- uncertainty
turning into abstention, and fabric changing the verdict -- is visible in one command.
"""

from __future__ import annotations

import argparse
import datetime as dt

from fitkit.catalog import CsvSpecImporter, GarmentSpecBuilder, InMemoryGarmentRepository
from fitkit.domain.body import BodyMeasurements, MeasurementProvenance
from fitkit.domain.capture import CaptureBundle, DeviceMetadata, GateVerdict, PhotoRef
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.policy import FitPolicy, FitPreference
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory
from fitkit.domain.units import Measure, MeasureSource
from fitkit.measurement import ResidualEntry, ResidualTable
from fitkit.orchestration import AdviceRequest, build_advisor

CSV = b"""size_label,region,value_cm,tolerance_cm
44,waist_flat,37.0,0.6
44,hip_flat,45.0,0.6
46,waist_flat,39.0,0.6
46,hip_flat,47.0,0.6
48,waist_flat,41.0,0.6
48,hip_flat,49.0,0.6
50,waist_flat,43.0,0.6
50,hip_flat,51.0,0.6
"""


class Store:
    def __init__(self) -> None:
        self._d: dict = {}

    def save(self, a) -> None:
        self._d[a.assessment_id] = a

    def load(self, aid):
        return self._d.get(aid)


class Backend:
    """Stands in for the measurement vendor: it returns the body you asked for."""

    backend_id = "explore"
    supported_regions = frozenset({BodyRegion.WAIST, BodyRegion.HIP})

    def __init__(self, waist_cm: float, hip_cm: float, residual_cm: float) -> None:
        self._waist, self._hip, self._residual = waist_cm, hip_cm, residual_cm

    def estimate(self, bundle, calibration) -> BodyMeasurements:
        return BodyMeasurements(
            residuals={
                BodyRegion.WAIST: Measure(self._waist, self._residual, MeasureSource.ESTIMATED),
                BodyRegion.HIP: Measure(self._hip, self._residual, MeasureSource.ESTIMATED),
            },
            scale_sigma_rel=calibration.sigma_rel,
            provenance=MeasurementProvenance(
                self.backend_id, "1", "residuals/explore", bundle.capture_id,
                calibration.source_id, dt.datetime.now(dt.UTC)),
        )


def capture_bundle(height_cm: float, height_sigma_cm: float) -> CaptureBundle:
    """Built directly rather than through the assembler: this demo has no images."""
    return CaptureBundle(
        capture_id="cap_explore",
        frontal=PhotoRef("memory://frontal", "a" * 64),
        lateral=PhotoRef("memory://lateral", "b" * 64),
        declared_height=Measure(height_cm, height_sigma_cm, MeasureSource.USER_DECLARED),
        declared_weight=None,
        device=DeviceMetadata("demo", "demo", "0"),
        gate_report=(GateVerdict("demo.all", True, 1.0, None),),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--waist", type=float, default=82.0, help="body waist girth in cm")
    p.add_argument("--hip", type=float, default=96.0, help="body hip girth in cm")
    p.add_argument("--height", type=float, default=175.0)
    p.add_argument("--height-sigma", type=float, default=1.5,
                   help="how well the shopper knows their own height; scales everything")
    p.add_argument("--residual", type=float, default=1.2,
                   help="the backend's measured per-region error in cm (C6)")
    p.add_argument("--stretch", choices=[s.value for s in StretchClass], default="low")
    p.add_argument("--preference", choices=[f.value for f in FitPreference], default="as_designed")
    args = p.parse_args()

    spec = (GarmentSpecBuilder()
            .with_identity("brand:trousers", version=1, category=GarmentCategory.TROUSERS,
                           size_system="EU", fit_intent=FitIntent.REGULAR)
            .with_fabric(FabricSpec(StretchClass(args.stretch), RecoveryClass.GOOD))
            .with_grading_tolerance(0.6)
            .with_rows(CsvSpecImporter().parse(CSV).rows)
            .build())

    policy = FitPolicy(
        policy_id="policy/explore", version=1, tau_single=0.65, tau_pair=0.85,
        max_critical_sigma_cm=2.5,
        region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8},
        critical_regions=frozenset({BodyRegion.WAIST, BodyRegion.HIP}),
        tightness_penalty_ratio=1.8)

    residuals = ResidualTable("residuals/explore", (
        ResidualEntry("explore", BodyRegion.WAIST, 999.0, args.residual),
        ResidualEntry("explore", BodyRegion.HIP, 999.0, args.residual)))

    advisor = build_advisor(
        garments=InMemoryGarmentRepository(spec), store=Store(), residuals=residuals,
        default_policy=policy, backend=Backend(args.waist, args.hip, args.residual))

    result = advisor.advise(AdviceRequest(
        capture=capture_bundle(args.height, args.height_sigma),
        garment_id="brand:trousers", merchant_id="explore",
        preference=FitPreference(args.preference), locale="en"))

    a = result.assessment
    rec = a.recommendation
    print(f"body     waist {args.waist} cm, hip {args.hip} cm "
          f"(backend residual {args.residual} cm, height {args.height}±{args.height_sigma} cm)")
    print(f"garment  {args.stretch} stretch, preference {args.preference}")
    print(f"verdict  {rec.verdict.value}")
    if rec.abstain is not None:
        detail = ", ".join(rec.abstain.detail_codes) or "-"
        print(f"         {rec.abstain.code.value} ({detail})")
    if rec.primary is not None:
        print(f"primary  {rec.primary.size_label} at {rec.primary.confidence:.0%}")
    if rec.alternate is not None:
        print(f"also     {rec.alternate.size_label} at {rec.alternate.confidence:.0%}")

    print("\nper size:")
    for size in a.sizes:
        print(f"  {size.size_label:>3}  confidence {size.confidence:.2f}")
        for r in size.regions:
            flag = "  <- uncertain" if r.uncertain else ""
            print(f"        {r.region.value:<6} {r.delta_cm:+6.1f} cm  +/-{r.delta_sigma_cm:.1f}"
                  f"  absorbed {r.stretch_absorbed_cm:.1f} cm  {r.classification.value}{flag}")

    print(f"\n{result.explanation.text}")
    if result.degradations:
        print("degradations:", [d.value for d in result.degradations])


if __name__ == "__main__":
    main()
