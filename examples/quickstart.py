"""A whole run of the pipeline, with no network, no database and no LLM.

The four classes at the top are the ports a real deployment supplies: a CV stack, object
storage, an assessment store and a measurement backend. Everything else is the library.
Run it with `python examples/quickstart.py`.
"""
import datetime as dt

from fitkit.acquisition import CaptureAssembler, RawCapture
from fitkit.catalog import CsvSpecImporter, GarmentSpecBuilder, InMemoryGarmentRepository
from fitkit.domain.body import BodyMeasurements, MeasurementProvenance
from fitkit.domain.capture import DeviceMetadata, FrameSignals, PhotoRef
from fitkit.domain.fabric import FabricSpec, RecoveryClass, StretchClass
from fitkit.domain.policy import FitPolicy
from fitkit.domain.regions import BodyRegion, FitIntent, GarmentCategory
from fitkit.domain.units import Measure, MeasureSource
from fitkit.measurement import ResidualEntry, ResidualTable
from fitkit.orchestration import AdviceRequest, build_advisor

# --- the three ports a deployment must supply -------------------------------------

class Analyzer:                      # the CV stack that scores a frame
    analyzer_id = "demo/1"
    def analyze(self, image, view):
        return FrameSignals(view=view, head_visible=True, feet_visible=True,
                            subject_frame_fraction=0.78, sharpness=0.9, exposure=0.7,
                            background_separability=0.8, arm_separation=0.8,
                            torso_verticality=0.95, device_pitch_deg=2.0,
                            clothing_tightness=0.8)

class Photos:                        # object storage
    def __init__(self): self._d = {}
    def put(self, capture_id, view, data):
        ref = PhotoRef(uri=f"memory://{capture_id}/{view}", sha256=f"{view:>064}".replace(" ", "0"))
        self._d[ref.uri] = data
        return ref
    def get(self, ref): return self._d[ref.uri]
    def delete(self, ref): self._d.pop(ref.uri, None)

class Store:                         # the assessment audit record
    def __init__(self): self._d = {}
    def save(self, a): self._d[a.assessment_id] = a
    def load(self, aid): return self._d.get(aid)

class Backend:                       # the measurement backend (C5: swappable)
    backend_id = "demo"
    supported_regions = frozenset({BodyRegion.WAIST, BodyRegion.HIP})
    def estimate(self, bundle, calibration):
        return BodyMeasurements(
            residuals={BodyRegion.WAIST: Measure(82.0, 1.2, MeasureSource.ESTIMATED),
                       BodyRegion.HIP: Measure(96.0, 1.2, MeasureSource.ESTIMATED)},
            scale_sigma_rel=calibration.sigma_rel,
            provenance=MeasurementProvenance("demo", "1", "residuals/2026-09",
                                             bundle.capture_id, calibration.source_id,
                                             dt.datetime.now(dt.UTC)))

# --- catalogue --------------------------------------------------------------------

CSV = b"""size_label,region,value_cm,tolerance_cm
46,waist_flat,39.0,0.6
46,hip_flat,47.0,0.6
48,waist_flat,41.0,0.6
48,hip_flat,49.0,0.6
50,waist_flat,43.0,0.6
50,hip_flat,51.0,0.6
"""
spec = (GarmentSpecBuilder()
        .with_identity("brand:sku-1", version=7, category=GarmentCategory.TROUSERS,
                       size_system="EU", fit_intent=FitIntent.REGULAR)
        .with_fabric(FabricSpec(StretchClass.LOW, RecoveryClass.GOOD))
        .with_grading_tolerance(0.6)
        .with_rows(CsvSpecImporter().parse(CSV).rows)
        .build())

policy = FitPolicy(policy_id="policy/demo", version=1, tau_single=0.65, tau_pair=0.85,
                   max_critical_sigma_cm=2.5,
                   region_weights={BodyRegion.WAIST: 1.0, BodyRegion.HIP: 0.8},
                   critical_regions=frozenset({BodyRegion.WAIST, BodyRegion.HIP}),
                   tightness_penalty_ratio=1.8)

residuals = ResidualTable("residuals/2026-09", (
    ResidualEntry("demo", BodyRegion.WAIST, 999.0, 1.2),
    ResidualEntry("demo", BodyRegion.HIP, 999.0, 1.2)))

# --- run --------------------------------------------------------------------------

capture = CaptureAssembler(Analyzer(), Photos()).assemble(
    "cap_001",
    RawCapture(frontal=b"<jpeg>", lateral=b"<jpeg>", declared_height_cm=175.0,
               declared_weight_kg=None,
               device=DeviceMetadata("ios", "iPhone15,2", "1.0.0")))

advisor = build_advisor(garments=InMemoryGarmentRepository(spec), store=Store(),
                        residuals=residuals, default_policy=policy, backend=Backend())

result = advisor.advise(AdviceRequest(capture=capture, garment_id="brand:sku-1",
                                      merchant_id="demo", locale="en"))

rec = result.assessment.recommendation
print(f"{rec.verdict.value}: size {rec.primary.size_label} (confidence {rec.primary.confidence:.0%})")
chosen = next(s for s in result.assessment.sizes if s.size_label == rec.primary.size_label)
for region in chosen.regions:
    print(f"  {region.region.value}: {region.delta_cm:+.1f} cm "
          f"(sigma {region.delta_sigma_cm:.1f}) {region.classification.value}")
print(result.explanation.text)
print("degradations:", [d.value for d in result.degradations])
