# Garment Size Prediction — Phase 1 Design

Status: **proposed, awaiting approval**. No production code has been written.
Date: 2026-09-03.

---

## 0. Executive summary

Three findings drive everything below.

**Finding 1 — the licensing chain kills the obvious technical answer.** The best-published
measurement research in this field is built on the SMPL family of body models and on the
BodyM dataset. Both are non-commercial. The SMPL licence additionally forbids using the
software *to train* networks for commercial use, which makes the encumbrance transitive:
"open" checkpoints trained against SMPL data are not automatically safe. Verified details
in §1.3.

**Finding 2 — capture protocol beats model choice, and it is measurable.** On Amazon's
BodyM benchmark, the same network moves from 31.93 mm waist MAE (single frontal view) to
18.71 mm (frontal + lateral + declared height) to 15.44 mm (+ weight). Meanwhile
general-purpose human-mesh-recovery methods evaluated on the same task produce waist errors
of 53–108 mm. The rigid capture protocol you already specified is worth more than any
backend swap, and *weight* is worth more than height. Verified numbers in §1.2.

**Finding 3 — the error budget is tighter than the size step, so abstention is not
optional.** Best realistic in-the-wild waist error is ~2 cm (1σ). A typical inter-size step
is ~4 cm. A 2 cm measurement error against a 4 cm step means a substantial fraction of
shoppers sit genuinely between sizes, and no amount of engineering removes that. Your C6 is
correct and this design makes uncertainty a type-level obligation rather than a policy.

I also want to register one disagreement with the framing, argued in §7: **body measurement
error is probably not the primary failure mode — garment specification error is.**
Manufacturing grading tolerance on a single size is frequently of the same order as our
measurement error, and unlike our error it is invisible to us. The design therefore puts
uncertainty on garment measurements too, not just body measurements.

**Recommendation:** integrate a commercial measurement vendor behind the `MeasurementBackend`
port for launch, build the evaluation harness *before* trusting it, and run a
commercially-licensed in-house track (Anny + MHR, both permissive) as the planned
replacement. Detail in §1.4.

---

## 1. Measurement backend selection

### 1.1 Evidence standard used here

Everything marked **[verified]** was read from the primary source during this session
(paper PDF, licence text, or repository file) and the numbers are quoted as published.
Everything marked **[vendor-claimed]** comes from marketing material and has not been
independently reproduced. Everything marked **[unverified]** I could not confirm and you
should not treat as fact. I have invented nothing.

### 1.2 Published accuracy against ground truth

**Amazon BMnet / BodyM** (Ruiz et al., *Human Body Measurement Estimation with Adversarial
Augmentation*, arXiv:2210.05667). Ground truth is 3D body scans of 2,505 real subjects with
14 tape-equivalent measurements. Test-A is lab-photographed; Test-B is photographed in a
less-controlled environment to simulate in-the-wild capture. **[verified]**

Table 2 — input ablation, errors in mm, BodyM Test-A:

| Inputs | Chest MAE | Hip MAE | Waist MAE |
|---|---|---|---|
| Single view | 33.95 | 31.03 | 31.93 |
| Multi-view (frontal + lateral) | 28.66 | 28.29 | 27.32 |
| Multi-view + height | 19.38 | 15.97 | 18.71 |
| Multi-view + weight | 15.22 | 10.54 | 13.69 |
| Multi-view + height + weight | 15.92 | 9.74 | 15.44 |

Table 7 — the same benchmark, comparing general human-mesh-recovery methods (single view,
no metadata), errors in mm:

| Method | Chest MAE | Hip MAE | Waist MAE |
|---|---|---|---|
| SPIN | 74.45 | 65.41 | 77.39 |
| STRAPS | 82.30 | 63.96 | 108.00 |
| Sengupta et al. | 53.07 | 47.43 | 53.20 |
| BMnet (single view, no metadata) | 33.95 | 31.03 | 31.93 |

Table 8 — Test-B (in-the-wild capture conditions), minimal-clothing subset, single view:
chest 22.67 mm, hip 16.41 mm, leg length 13.58 mm, waist 20.78 mm.

Table 9 — Test-A per-measurement, full training set: chest 18.84, hip 11.34, waist 15.78,
mean over 14 measurements 9.19 mm.

**SHAPY** (Choutas et al., CVPR 2022, arXiv:2206.07036), evaluated on the authors' HBW
dataset — 35 subjects with 3D-scan ground truth, in-the-wild photos. Table 3, MAE in mm:
**[verified]**

| Method | Height | Chest | Waist | Hips |
|---|---|---|---|---|
| SMPLR | 182 | 267 | 309 | 305 |
| STRAPS | 135 | 167 | 145 | 102 |
| SPIN | 59 | 92 | 78 | 101 |
| TUCH | 58 | 89 | 75 | 57 |
| Sengupta et al. | 82 | 133 | 107 | 63 |
| ExPose | 85 | 99 | 92 | 94 |
| **SHAPY** | **51** | **65** | **69** | **57** |

SHAPY is the best of that set and its waist error is still 6.9 cm. On the MMTS set
(Table 4) SHAPY reports height 71, chest 64, waist 98, hips 74 mm.

**Read this comparison carefully.** The two families are not measuring the same thing. HBW
is unconstrained single-image capture; BodyM Test-A is a rigid protocol with a declared
height. The 4–5x gap between them is the value of the protocol, not the value of the
network. That is the single most actionable fact in this document.

### 1.3 Licensing chain — blocking findings

| Asset | Licence | Commercial use | Status |
|---|---|---|---|
| SMPL / SMPL-H / SMPL-X / STAR / SUPR | MPG research licence | **No.** Commercial sub-licence only via Meshcapade | **BLOCKING** |
| SMPL licence, training clause | — | Prohibits using the software to train networks for commercial use of any kind | **BLOCKING, transitive** |
| BodyM dataset | CC BY-NC 4.0 | **No** | **BLOCKING** |
| Anny body model + code (NAVER) | Apache 2.0 | Yes | Clear |
| Meta MHR (`facebookresearch/MHR`) | Apache 2.0 **[unverified — README/search level, not read from the repo LICENSE]** | Yes, if confirmed | Verify before committing |
| Meta SAM 3D Body | "SAM License" — grants use, reproduction, modification, derivative works, royalty-free; adds ITAR / trade-control / no-reverse-engineering restrictions **[verified from the repo LICENSE]** | Yes, with an acceptable-use policy attached | Counsel review, not a blanket OSS grant |
| Commercial vendor APIs | Per contract | Yes | Contract + DPA |

Sources: `smpl.is.tue.mpg.de/modellicense.html`, `smpl-x.is.tue.mpg.de/modellicense.html`,
`awslabs/open-data-registry/datasets/bodym.yaml`, `arxiv.org/abs/2511.03589`,
`github.com/facebookresearch/sam-3d-body/blob/main/LICENSE`.

The transitive clause is the trap. A permissively-licensed *checkpoint* whose training
pipeline used SMPL meshes, SMPL-derived synthetic data, or a CC BY-NC dataset may still be
encumbered. **Every candidate weight file must have its training-data provenance
documented before it enters the build.** This is a legal-diligence task, not an
engineering one, and it belongs in Phase 2 planning as a hard gate.

### 1.4 Candidate comparison

| | **A. In-house silhouette regressor** (BMnet architecture, own data) | **B. SHAPY / SMPL-X research stack** | **C. Permissive open stack** (Anny + MHR / SAM 3D Body) | **D. Commercial vendor API** |
|---|---|---|---|---|
| Accuracy vs tape | Waist 15–19 mm lab, 20.8 mm in-the-wild **[verified]** | Waist 69–98 mm **[verified]** | **No published evidence for anthropometry.** Anny claims mm-accurate scan *fitting* and HMR parity — neither is tape accuracy **[verified claim, wrong metric]** | 1.5–2.2 cm **[vendor-claimed]**; one 2022 academic comparison reports bust/waist/shoulder discrepancies exceeding ANSUR allowable error **[unverified — secondary summary, paper not read in full]** |
| Frontal + lateral | Native, and the second view is worth ~4.6 mm on waist **[verified]** | Single image, no fusion | Single image; multi-view fusion is ours to build | Typically yes; matches our protocol |
| Scale calibration | Height is an explicit network input | None at inference | Fit a metric-scaled mesh; we own it | Height is a required field |
| Licensing | Architecture free; **weights and data blocked** | **Blocked** | Clear (pending MHR verification) | Contractual; plus a GDPR data-processing problem — full-body photos leave our control |
| Maintenance | Paper 2022, no maintained implementation **[unverified]** | Research repo, low activity **[unverified]** | Active, released Nov 2025 | Vendor SLA |
| Inference cost / latency | MNASNet-class backbone — cheap, tens of ms | GPU seconds | GPU seconds, self-hosted | Network round trip; **per-scan price not public [unverified]** |
| Integration effort | Very high — requires our own scanned + tape-measured dataset | Low to integrate, but unusable | High — we own measurement extraction, calibration and validation | **Low — days** |
| Verdict | Target architecture, blocked on data | **Reject** | In-house track | **Launch choice** |

### 1.5 Recommendation and rationale

**Launch on D (commercial vendor) behind the `MeasurementBackend` port. Build C in parallel
as the planned replacement. Reject B outright. Treat A as the destination if we ever fund
our own scan panel.**

Rationale, in order of weight:

1. **The evaluation harness must exist before the backend is trusted, and it can only be
   built against a working end-to-end system.** Nothing about size prediction is knowable
   until return outcomes flow back. A vendor gets us to that loop in days rather than
   quarters. We are buying measurement time, not measurement truth.
2. **No candidate has published accuracy on *our* protocol with *our* population.** Vendor
   claims are unverified; the open stack has no anthropometric evidence at all. So the
   choice is not "which is most accurate" — it is "which lets us find out fastest, at
   acceptable switching cost". C5 makes the answer cheap to revise.
3. **The open stack is the only path that is simultaneously commercially clean, cost-
   controlled at volume, and privacy-defensible** (images never leave our infrastructure).
   That makes it the right destination, but it is a build, not an integration, and it must
   be validated against tape before it can be trusted with a shopper's money.
4. **The vendor decision is reversible; the protocol decision is not.** Capture protocol
   changes require re-onboarding users. So we should over-invest in Phase 1 (acquisition)
   and under-commit on Phase 2 (measurement).

Consequences we accept: vendor per-scan cost at unknown unit economics; a DPA and probably
an Article 9 biometric-data assessment; and a hard dependency on the vendor's uptime
during launch, mitigated by the degradation ladder in §4.

### 1.6 What I could not verify

- Vendor per-scan pricing, latency SLAs, and whether any vendor will contractually commit
  to an accuracy figure. All are commercial questions for a procurement conversation.
- The 2022 academic comparison of mobile scanning apps — I have only a search-result
  summary of it. Do not cite it externally on my say-so.
- Meta MHR's repository LICENSE file. Search results say Apache 2.0; I did not read it.
- Whether BMnet's implementation was ever released.

---

## 2. System architecture

### 2.1 Module map and dependency direction

```
                        ┌───────────────────────────────┐
                        │  domain  (pure, zero deps)    │
                        │  Measure, BodyRegion,         │
                        │  BodyMeasurements, GarmentSpec│
                        │  FabricSpec, FitPolicy,       │
                        │  errors, PORTS (Protocols)    │
                        │  contracts/FitAssessment v1   │
                        └───────────────────────────────┘
                          ▲     ▲      ▲      ▲       ▲
          ┌───────────────┘     │      │      │       └────────────────┐
          │                     │      │      │                        │
  ┌───────────────┐  ┌──────────────┐ │ ┌──────────────┐   ┌───────────────────┐
  │ acquisition   │  │ measurement  │ │ │ catalog      │   │ explanation       │
  │ (gates)       │  │ (backends,   │ │ │ (specs,      │   │ (renderers,       │
  │               │  │  calibration)│ │ │  importers)  │   │  guards, LLM port)│
  └───────────────┘  └──────────────┘ │ └──────────────┘   └───────────────────┘
          ▲                  ▲        │        ▲                    ▲
          │                  │  ┌─────────────┐│                    │
          │                  │  │ fit_engine  ││                    │
          │                  │  │ (pure)      ││                    │
          │                  │  └─────────────┘│                    │
          │                  │        ▲        │                    │
          └──────────────────┴────────┴────────┴────────────────────┘
                                      │
                        ┌───────────────────────────────┐
                        │ orchestration                 │
                        │ (composition root, public API)│
                        └───────────────────────────────┘
                                      │  reads persisted contracts
                        ┌───────────────────────────────┐
                        │ evaluation (offline)          │
                        └───────────────────────────────┘

  infrastructure/  (vendor HTTP clients, S3/GCS photo store, Postgres,
                    Anthropic client) — depends on domain ports only,
                    and nothing depends on it except orchestration wiring.
```

Edges, stated as rules:

1. `domain` imports nothing from this project and no third-party runtime library. It is
   the only module allowed to define the vocabulary.
2. `fit_engine` imports `domain` only. No I/O, no clock, no randomness, no network.
3. `explanation` imports `domain.contracts` **only** — explicitly *not* `fit_engine`. This
   is the edge that enforces C1: the renderer physically cannot reach the arithmetic.
4. `acquisition`, `measurement`, `catalog` import `domain` and each other never.
5. `orchestration` is the only module that names concrete adapter classes.
6. `evaluation` reads persisted `FitAssessment` documents and ground-truth tables; it does
   not import adapters.

**The graph is acyclic by construction**: every arrow points at `domain` or at
`orchestration`, and `orchestration` is imported by nothing. This is not an assertion to
be trusted — it is checked in CI by `import-linter` with a layered contract, and a
violating import fails the build. That is the acceptance criterion.

### 2.2 The uncertainty type — how C6 is made structural

```python
Cm = float  # documented alias; the invariant lives in Measure, not the alias

class MeasureSource(StrEnum):
    ESTIMATED = "estimated"        # from the measurement backend
    USER_DECLARED = "user_declared"# self-reported height/weight
    TAPE = "tape"                  # ground truth, evaluation only
    SPEC_SHEET = "spec_sheet"      # garment tech pack
    DERIVED = "derived"            # computed from other Measures

@dataclass(frozen=True, slots=True)
class Measure:
    value_cm: float
    sigma_cm: float                # 1-sigma. No default. Must be > 0.
    source: MeasureSource
```

Three rules make silent false precision impossible:

- **There is no bare centimetre float anywhere in the domain.** Every length is a
  `Measure`. A function that wants to accept a number instead of a `Measure` cannot type-
  check.
- **`sigma_cm` has no default and is validated `> MIN_SIGMA_CM`.** A backend that returns
  bare numbers cannot construct a `Measure` without someone deciding, in code, what its
  uncertainty is. The decision is forced into the open.
- **Arithmetic propagates.** `Measure.__sub__` returns a `Measure` whose sigma is the
  combined sigma. You cannot subtract your way out of the error bars.

### 2.3 Correlated error — why per-region sigma is not enough

Independent per-region sigmas are wrong here, and wrong in the direction that matters. If
the shopper declares 175 cm and is 172 cm, *every* circumference is inflated by roughly the
same proportion. Correlated error shifts all regions the same way, which is a size-shift;
independent error smears them, which is a fit-quality question. Treating the first as the
second will systematically under-abstain.

The v1 model is two-component:

```
sigma_total(r)^2  =  (k_r * sigma_scale)^2  +  sigma_resid(r)^2
```

- `sigma_scale` — relative scale uncertainty from the calibration source. For declared
  height it is dominated by self-report error and rounding; **its value must be measured on
  our validation panel, not assumed** (see §7.4 and Phase 7).
- `k_r` — the sensitivity of region `r` to scale, ~1.0 for circumferences and lengths.
- `sigma_resid(r)` — independent residual, from the backend's measured residual
  distribution.

```python
@dataclass(frozen=True)
class BodyMeasurements:
    values: Mapping[BodyRegion, Measure]
    scale_sigma_rel: float          # the shared component
    provenance: MeasurementProvenance
```

Full covariance is deferred and recorded as such in ADR-004.

---

## 3. The fit-engine → explanation contract

This is the most important artifact in the system, so it is specified before the phases
that produce and consume it.

### 3.1 Shape

`FitAssessment`, JSON, schema id `fit-assessment/1.0.0`.

```jsonc
{
  "schema_version": "fit-assessment/1.0.0",
  "assessment_id": "01J...",                  // ULID
  "garment": {
    "garment_id": "brand:sku",
    "category": "TROUSERS",                    // enum
    "size_system": "EU",
    "fit_intent": "REGULAR"                    // designer's ease intent, enum
  },
  "fabric": {
    "stretch_class": "LOW",                    // enum: NONE|LOW|MEDIUM|HIGH
    "recovery": "GOOD",                        // enum: GOOD|POOR|UNKNOWN
    "usable_extension_pct": 6.0                // what the StretchModel actually allowed
  },
  "recommendation": {
    "verdict": "SINGLE",                       // SINGLE | TWO_SIZES | ABSTAIN
    "primary":   { "size_label": "48", "confidence": 0.71 },
    "alternate": { "size_label": "50", "confidence": 0.24 },   // or null
    "abstain": null                            // or { "code": ..., "detail_codes": [...] }
  },
  "sizes": [
    {
      "size_label": "48",
      "confidence": 0.71,
      "regions": [
        {
          "region": "WAIST",
          "critical": true,
          "delta_cm": -2.0,                    // garment minus body, sign is explicit
          "delta_sigma_cm": 1.4,
          "stretch_absorbed_cm": 1.2,
          "required_ease_cm": { "min": 1.0, "preferred": 2.0, "max": 5.0 },
          "classification": "TIGHT",           // enum, closed vocabulary
          "uncertain": false
        }
      ],
      "coverage": "COMPLETE",                  // COMPLETE | PARTIAL
      "missing_regions": []
    }
  ],
  "inputs_digest": {
    "measurement_backend": "vendor-x@2026.07",
    "measurement_provenance_id": "...",
    "garment_spec_version": "brand:sku@7",
    "engine_version": "fit-engine/1.4.2",
    "policy_version": "policy/merchant-a/3",
    "computed_at": "2026-09-03T11:04:22Z"
  },
  "render_hints": { "locale": "it-IT", "tone": "NEUTRAL" }
}
```

### 3.2 Rules that make C1 and C2 enforceable rather than aspirational

**R1 — No free text from the engine.** Every qualitative statement in the document is a
value from a closed enum. The engine never emits a sentence.

**R2 — Completeness.** Every number a renderer may legitimately mention is present in the
document. There is nothing to compute.

**R3 — Isolation.** `explanation` imports `domain.contracts` and nothing else from the
project. Enforced by the import-linter contract, so a well-meaning future change that
"just needs the garment spec" fails CI rather than review.

**R4 — The numeric guard.** After an LLM renders, every numeric literal in its output is
extracted and checked against an allowlist derived from the document (values, and their
roundings to the tolerances we publish). A number that is not in the allowlist fails the
guard and the response falls back to the template renderer. *This is the teeth behind C1.*
Without it, "the LLM never decides the size" is a hope about a prompt. With it, an LLM that
invents "about 3 cm of room" cannot reach a shopper.

I want to be blunt about R4's limit: it catches invented *numbers*. It does not catch
invented *reassurance* — "should still be comfortable" on a −2 cm waist in rigid denim
contains no numeral. That is why the document carries `classification` as a closed enum and
why the guard also runs a banned-claim check against a controlled vocabulary keyed by
`classification`. C1 as you wrote it protects the arithmetic; the rhetoric needs its own
constraint, and I have added one.

**R5 — Determinism is not available from the LLM, so do not design for it.** Sampling
parameters (`temperature`, `top_p`, `top_k`) are removed on the current Claude models and
return a 400. There is no `temperature=0` to hide behind. Reproducibility therefore comes
from the template renderer (which is exactly reproducible) plus the guard (which bounds
what the LLM can say), never from asking the model to be deterministic.

### 3.3 Versioning

- Semantic versioning on `schema_version`, mandatory on read.
- **Minor = additive optional fields only.** Renderers must ignore unknown optional fields.
- **Major = anything else.** Removing a field, narrowing an enum, changing a unit, or
  changing the sign convention of `delta_cm` are all major.
- Enums are extensible only in a major version; a renderer receiving an unknown enum value
  fails closed to the template renderer rather than guessing.
- Every renderer declares the range it supports. CI runs every renderer against a corpus of
  golden documents for every supported version — that matrix is the compatibility test.
- Persisted documents are never migrated in place. The evaluation harness reads historical
  versions, which is why old readers must keep working.

---

## 4. Error taxonomy and failure propagation

```
FitKitError
├── InputError                      → 4xx, user or caller can fix
│   ├── CaptureRejected             → carries per-gate failures + remediation codes
│   ├── InvalidDeclaredHeight
│   ├── GarmentNotFound
│   └── SizeSpecIncomplete          → the spec lacks a region we need
├── DegradedResult                  → not an exception; a flag on a successful result
│   ├── MEASUREMENT_UNCERTAIN       → engine still runs, may abstain
│   ├── COVERAGE_PARTIAL            → some regions missing, said out loud
│   └── EXPLANATION_TEMPLATED       → LLM unavailable or guard-failed
└── SystemError                     → 5xx
    ├── BackendUnavailable / BackendTimeout
    ├── StorageError
    └── ContractViolation           → the engine emitted a document failing its own schema
```

Propagation rules:

1. **No foreign exception crosses a boundary.** Each port defines its own error type;
   adapters translate. An `httpx` timeout becomes `BackendTimeout` at the adapter, never
   reaches the domain.
2. **A bad fit is a result, not an error.** The engine raises only on `ContractViolation`.
   "Nothing fits" is a perfectly good `FitAssessment` with `verdict: ABSTAIN`.
3. **Fail closed on numbers, fail open on prose.** A measurement we cannot trust produces
   an abstention. An explanation we cannot generate produces a template. The request never
   fails because the LLM is down — that is C2 restated as an operational rule.
4. **Partial data is declared, never silently dropped.** A garment spec missing thigh
   circumference yields `coverage: PARTIAL` and `missing_regions: ["THIGH"]`, which the
   renderer is required to surface.

### 4.1 Degradation ladder

| Failure | Behaviour | User-visible outcome |
|---|---|---|
| LLM down / guard failure | Template renderer | Slightly blander prose, identical numbers |
| Measurement sigma above threshold | Engine abstains or widens to two sizes | "We're not confident enough to call it" |
| Measurement backend down | Fail the request | Retry prompt; no fabricated measurements |
| Garment spec missing regions | Partial assessment, declared | "We can't check the thigh on this one" |
| Capture gate failure | Reject before measuring | Specific instruction: step back, straighten up |

---

## 5. Observability

**The persisted `FitAssessment` is the primary telemetry artifact.** It is the audit record
(why did we tell this person to buy a 48?), the evaluation substrate, and the replay input
for a future engine version. Everything else is secondary.

- **Correlation:** `assessment_id` propagates from capture through to the order record.
  Without that join key, the end-to-end metric in Phase 7 is unmeasurable, so it is a Phase 1
  requirement, not a Phase 6 nicety.
- **Metrics:** per-gate rejection rate and retry count; backend latency, error rate, cost;
  per-region sigma distribution; abstain rate; verdict distribution; renderer fallback rate.
- **The guard-violation rate is a safety metric, not a quality metric.** A rising count
  means the LLM is attempting to state numbers that are not in the contract. Alert on it.
- **Sigma calibration is monitored, not assumed.** Phase 7 measures whether our 1σ really
  contains ~68% of ground truth. A miscalibrated sigma silently breaks abstention, which
  silently breaks C6.
- **PII:** images are the sensitive asset. Logs carry references, never pixels. The
  `PhotoStore` port has a deletion API and a retention policy; measurement records may
  outlive images and usually should.

---

## 6. Phased implementation plan

Eight phases. Phase 0 is one I added: the shared vocabulary. Everything else depends on it,
and building it inside Phase 1 would make the `Measure` invariant an acquisition concern
rather than a system-wide one.

Ordering rationale: 0 → 4 → 3 → 1 → 2 → 5 → 6 → 7 is the *dependency* order, but I propose
building in the order **0, 4, 3, 5, 1, 2, 6, 7** — fit engine and catalog first, against
synthetic bodies. The engine is the part with real intellectual content, it is testable with
zero infrastructure, and building it early forces the contract to be right before anything
depends on it. Measurement, the expensive and uncertain part, comes after we know exactly
what shape of input it must produce.

---

### Phase 0 — Domain kernel (`fitkit.domain`)

**Responsibility.** Define the vocabulary and the invariants: quantities with uncertainty,
body and garment regions, fabric, policies, errors, ports, and the `FitAssessment` contract.

**Non-goals.** No algorithms. No I/O. No persistence models. Not a place for "utilities".

**Public interface.**

```python
Cm = float

class MeasureSource(StrEnum): ...
class BodyRegion(StrEnum):
    BUST; UNDERBUST; WAIST; HIP; THIGH; NECK; SHOULDER_WIDTH; ARM_LENGTH; INSEAM; HEIGHT
class GarmentRegion(StrEnum):
    CHEST_FLAT; WAIST_FLAT; HIP_FLAT; THIGH_FLAT; SHOULDER; SLEEVE_LENGTH; INSEAM; TOTAL_LENGTH
class GarmentCategory(StrEnum): TROUSERS; DRESS; TOP; JACKET; SKIRT
class FitIntent(StrEnum): SLIM; REGULAR; OVERSIZED
class FitPreference(StrEnum): TIGHTER; AS_DESIGNED; LOOSER

@dataclass(frozen=True, slots=True)
class Measure:
    value_cm: float
    sigma_cm: float
    source: MeasureSource
    def __sub__(self, other: "Measure") -> "Measure": ...
    def __add__(self, other: "Measure") -> "Measure": ...
    def scaled(self, factor: float) -> "Measure": ...

@dataclass(frozen=True)
class BodyMeasurements:
    values: Mapping[BodyRegion, Measure]
    scale_sigma_rel: float
    provenance: MeasurementProvenance

@dataclass(frozen=True)
class FabricSpec:
    stretch_class: StretchClass
    elongation_pct: float | None       # measured extension at a defined load
    recovery: RecoveryClass
    composition: str | None

@dataclass(frozen=True)
class GarmentSizeSpec:
    size_label: str
    measurements: Mapping[GarmentRegion, Measure]   # note: Measure, so specs carry sigma

@dataclass(frozen=True)
class GarmentSpec:
    garment_id: GarmentId
    version: int
    category: GarmentCategory
    size_system: str
    fit_intent: FitIntent
    fabric: FabricSpec
    sizes: Sequence[GarmentSizeSpec]

# Ports (Protocols) — defined here, implemented outward
class MeasurementBackend(Protocol): ...
class ScaleCalibrationSource(Protocol): ...
class GarmentRepository(Protocol): ...
class ExplanationRenderer(Protocol): ...
class LlmClient(Protocol): ...
class PhotoStore(Protocol): ...
class AssessmentStore(Protocol): ...
class Clock(Protocol): ...
```

**Dependencies.** None. Standard library only.

**Patterns applied.**
- **Value Object** (not GoF-catalogued but the load-bearing one): `Measure` and all specs
  are frozen and compared by value. *What varies:* nothing — that is the point. *What
  breaks without it:* mutable measurements shared across a request make the audit record a
  lie, and `Measure`'s invariant becomes unenforceable.
- **Nothing else.** There is no factory, no builder, no registry here. Dataclasses and
  enums are sufficient and anything more would be the gratuitous indirection you warned
  about.

**Test strategy.** Pure unit tests. Property-based tests (Hypothesis) on `Measure`
arithmetic: sigma never decreases under subtraction; construction with `sigma <= 0` always
raises; `a - b` and `b - a` have equal sigma and opposite value.

**Acceptance criteria.**
- `import fitkit.domain` pulls in no third-party package (checked by a test that inspects
  `sys.modules` before/after).
- 100% branch coverage on `Measure` validation.
- A grep-based test asserts no public dataclass field in `domain` is annotated as a bare
  `float` with a name ending in `_cm` other than inside `Measure`.
- `import-linter` contract passes: `domain` is the innermost layer.

**Integration contract.** Every other phase imports these types. Breaking changes here are
breaking changes everywhere; the module is versioned with the package.

---

### Phase 4 (built 2nd) — Deterministic fit engine (`fitkit.fit_engine`)

**Responsibility.** Given body measurements, a garment spec, a fit preference and a policy,
produce a `FitAssessment`. All arithmetic, thresholding, ranking, size selection and
abstention live here and nowhere else.

**Non-goals.** No persistence, no clock, no randomness, no network, no natural language, no
knowledge that an LLM exists.

**Public interface.**

```python
class FitEngine(Protocol):
    def assess(
        self,
        body: BodyMeasurements,
        garment: GarmentSpec,
        preference: FitPreference,
        policy: FitPolicy,
    ) -> FitAssessment: ...

class EaseRulePolicy(Protocol):
    def required_ease(
        self, region: BodyRegion, category: GarmentCategory,
        intent: FitIntent, preference: FitPreference,
    ) -> EaseInterval: ...            # min / preferred / max, in cm

class StretchModel(Protocol):
    def usable_extension(
        self, fabric: FabricSpec, region: BodyRegion, category: GarmentCategory,
    ) -> float: ...                   # fraction of circumference, >= 0

class AbstainPolicy(Protocol):
    def decide(self, ranked: Sequence[SizeScore], body: BodyMeasurements) -> Verdict: ...
```

**Algorithm.**

1. **Region mapping.** Garment flat measures are converted to comparable body quantities
   (a flat waist of 40 cm becomes an 80 cm circumference) as a `Measure`, doubling the
   sigma with the value.
2. **Effective garment circumference.** `eff(r,s) = garment(r,s) * (1 + usable_extension)`.
   Usable extension is zero when `recovery == POOR` beyond a small allowance — a fabric that
   stretches and stays stretched does not give you a size, it gives you a bagged-out garment
   after three wears. **This is where C4 lives.**
3. **Ease.** `delta(r,s) = eff(r,s) − body(r)`, a `Measure`, sigma combined.
4. **Per-region penalty.** A piecewise function of `delta` against `EaseInterval`,
   asymmetric: too tight is penalised harder than too loose, and the asymmetry is a policy
   parameter, not a constant.
5. **Aggregation.** Weighted by region, weights supplied by category (trousers weight
   waist/hip/thigh/inseam; a dress weights bust/waist/hip). Critical regions can veto.
6. **Uncertainty propagation (C6).** The score is evaluated over the *distribution*, not
   the point estimate, using **deterministic Gauss–Hermite quadrature** — a fixed 5-node
   grid over the shared scale factor, with the independent residual handled analytically
   per region. Fixed nodes and fixed weights mean the result is bit-reproducible for
   identical inputs. Monte Carlo is explicitly rejected: it would make the engine
   non-deterministic or force a seed, and a seeded RNG in a "deterministic engine" is a
   trap for the next maintainer.
7. **Verdict.** Quadrature yields `P(size s is best)` for each size. Then:
   `P(top) ≥ τ_single` → SINGLE; `P(top) + P(second) ≥ τ_pair` and both plausible →
   TWO_SIZES; otherwise, or if a critical region's sigma exceeds `σ_max` → ABSTAIN with a
   reason code. **Widening to two sizes is derived from the arithmetic, not hand-tuned per
   case.**

**Dependencies.** `fitkit.domain` only.

**Patterns applied.**

| Pattern | What varies | What breaks without it |
|---|---|---|
| **Strategy** — `EaseRulePolicy` | Garment category, brand fit philosophy, merchant overrides | A growing `if category == ...` inside the scorer; per-merchant tuning becomes a fork |
| **Strategy** — `StretchModel` | How usable extension is derived: linear class-based (v1) → measured tension curve (v2) | Fabric maths welded into scoring; C4 becomes untestable in isolation |
| **Strategy** — `AbstainPolicy` | Merchant risk appetite. Free returns tolerate a confident guess; made-to-order does not | Abstention thresholds become global constants and every merchant gets the wrong one |

**Patterns deliberately NOT applied**, with reasons, because you asked:
- **Template Method** for the scoring pipeline — rejected. It would put the pipeline in a
  base class and the variation in subclasses. A plain function composing three injected
  strategies is clearer and does not create an inheritance hierarchy. Composition over
  inheritance.
- **Visitor** over region types — rejected. Regions are an enum, not a type hierarchy; a
  dict lookup is the correct tool.
- **Factory / Abstract Factory** — rejected. The composition root constructs strategies
  with plain calls. A factory would add a layer whose only job is to call a constructor.
- **Chain of Responsibility** for the verdict rules — rejected. The rules are a short,
  ordered, total decision function; `if/elif/else` in one place is more readable and more
  testable than four classes.

**Test strategy.**
- *Unit, no doubles needed* — the engine is pure, so every test is a table test. Golden
  cases per category, including the ones that matter: −2 cm waist on rigid denim (should
  be TIGHT and probably a size up) vs −2 cm on 4-way elastane (should be AS_INTENDED).
- *Property-based* — monotonicity: increasing a body measurement never improves the score
  of a too-tight region; increasing sigma never increases `P(top)`; the sum of size
  probabilities is 1 ± ε.
- *Determinism* — the same inputs produce a byte-identical serialized document, asserted
  across 1000 repeats and across a fresh interpreter (guards against dict-ordering and
  hash-seed dependence).
- *Contract* — every produced document validates against the JSON Schema. This is the
  `ContractViolation` tripwire.
- **Fakes needed:** none. That is the strongest argument that the boundary is right.

**Acceptance criteria.**
- A hand-computed reference table of ≥ 40 (body, garment, fabric) cases is reproduced
  exactly.
- Mutation testing on the scoring module ≥ 85% killed.
- Zero imports outside `fitkit.domain` and the standard library, enforced in CI.
- Given a body whose waist sigma is set to 3.0 cm against a 4 cm size step, the engine
  **must** return TWO_SIZES or ABSTAIN. This single test is the executable form of C6.

**Integration contract.** Emits `FitAssessment v1` (§3). Consumes `BodyMeasurements`,
`GarmentSpec`, `FitPolicy`. Versioned per §3.3.

---

### Phase 3 (built 3rd) — Garment catalog (`fitkit.catalog`)

**Responsibility.** Ingest per-size *physical* garment measurements and fabric properties
from supplier formats, normalise them (flat vs circumference, units, landmark conventions),
attach uncertainty, version them immutably, and serve them.

**Non-goals.** No inference of missing sizes by interpolation in v1 (it hides grading
reality). No scraping of marketing size charts. No fit logic.

**Public interface.**

```python
class GarmentRepository(Protocol):
    def get(self, garment_id: GarmentId, version: int | None = None) -> GarmentSpec: ...
    def latest_version(self, garment_id: GarmentId) -> int: ...

class SpecImporter(Protocol):
    source_format: SourceFormat
    def parse(self, raw: bytes) -> ImportResult: ...   # specs + per-row diagnostics

class GarmentSpecBuilder:
    def with_sizes(...) -> Self
    def with_fabric(...) -> Self
    def with_tolerance(...) -> Self
    def build(self) -> GarmentSpec:  # validates; raises SizeSpecIncomplete
        ...
```

**Dependencies.** `fitkit.domain`. Storage adapter depends on the repository port, not the
other way round.

**Patterns applied.**

| Pattern | What varies | What breaks without it |
|---|---|---|
| **Adapter** — one `SpecImporter` per supplier format | Every brand sends a different spreadsheet, tech pack or API payload | Format-specific parsing bleeds into the domain; adding a brand becomes a core change |
| **Builder** — `GarmentSpecBuilder` | A spec is assembled from *several documents* (measurement sheet, fabric sheet, grading rules) arriving at different times | Half-built specs leak into the system. `build()` is the single validation gate that makes an invalid `GarmentSpec` unconstructible |

**Repository** is used and named honestly as a DDD pattern, not a GoF one.

**Test strategy.** Importer unit tests against real (anonymised) supplier files, including
malformed ones. Repository contract test suite run against both the in-memory fake and the
real Postgres adapter — the same tests, two implementations, which is the whole point of
the port. Round-trip property: parse → build → serialize → parse is identity.

**Acceptance criteria.**
- A spec missing a region required by its category fails `build()` with a named error.
- Every stored spec is immutable; an update creates version *n+1* and version *n* remains
  retrievable forever (asserted, because Phase 7 replay depends on it).
- Unit conversion is exercised by a test in inches that must equal the cm fixture within
  0.01 cm.
- Every `Measure` in a spec has a non-zero sigma, defaulting to a documented per-brand
  grading tolerance rather than to zero.

**Integration contract.** `GarmentSpec` with `version`. The version string appears in
`inputs_digest`, which is what lets Phase 7 attribute a return to the exact spec in force.

---

### Phase 5 (built 4th) — Explanation layer (`fitkit.explanation`)

**Responsibility.** Turn one `FitAssessment` into localized prose.

**Non-goals.** No arithmetic. No data access. No knowledge of how the assessment was
produced. No ability to change a recommendation.

**Public interface.**

```python
@dataclass(frozen=True)
class Explanation:
    text: str
    renderer_id: str
    degraded: bool
    guard_report: GuardReport

class ExplanationRenderer(Protocol):
    supported_schema_versions: SemverRange
    def render(self, assessment: FitAssessment, ctx: RenderContext) -> Explanation: ...

class LlmClient(Protocol):
    def complete(self, prompt: Prompt, *, max_tokens: int) -> Completion: ...
```

Implementations:
- `TemplateRenderer` — **the reference implementation.** Complete, exactly reproducible,
  localizable, free. Not a fallback bolted on afterwards; it is written first and it
  defines what "correct output" means. C2 is satisfied by construction because the template
  renderer is the thing the LLM is compared against, not the thing that rescues it.
- `LlmRenderer` — Adapter over `LlmClient`. Prompt carries the assessment JSON and the
  controlled vocabulary; the model's job is phrasing and ordering, nothing else.
- `GuardedRenderer(inner, fallback, guards)` — **Decorator.** Runs numeric-allowlist,
  banned-claim, length and locale guards; on any violation returns the fallback's output
  with `degraded=True` and a populated `guard_report`.

**Model choice.** Default `claude-opus-5` (1M context; $5/$25 per MTok input/output). This
is a short structured-to-prose transform, so per-call cost is dominated by a small fixed
prompt; prompt caching on the stable system prompt and vocabulary applies directly. Whether
high volume justifies a cheaper model is a cost decision that is yours, not mine — it is
Open Question 7, and the port makes it a one-line change either way.

**Dependencies.** `fitkit.domain.contracts` only. Enforced in CI.

**Patterns applied.**

| Pattern | What varies | What breaks without it |
|---|---|---|
| **Strategy** — `ExplanationRenderer` | Template vs LLM vs A/B experiment arm vs a future vendor | C2 becomes untestable; swapping renderers means editing the orchestrator |
| **Decorator** — `GuardedRenderer` | Guards compose and are added over time; each must be independently testable and reusable across renderers | Guard logic lives inside `LlmRenderer`, cannot be unit-tested alone, and is silently absent from any renderer added later — which is exactly how C1 erodes |

**Not applied:** Chain of Responsibility for the fallback sequence — a two-element decorator
is sufficient and a chain would be ceremony. Observer for guard events — a metrics port
call is a function call.

**Test strategy.**
- `TemplateRenderer`: golden-file tests per locale over the full corpus of assessment
  fixtures. Exactly reproducible, so byte comparison is legitimate.
- `GuardedRenderer`: unit-tested with a `StubLlmClient` returning adversarial outputs —
  invented numbers, a recommendation contradicting the verdict, a different size label,
  wrong locale. Each must be caught. **These are the tests that prove C1.**
- `LlmRenderer`: contract test against a recorded-response fake for CI; a small nightly
  live-API suite, quarantined from the main pipeline so the build never depends on a
  network.
- Compatibility matrix: every renderer × every supported schema version × golden documents.

**Acceptance criteria.**
- Deleting the `LlmClient` implementation from the build leaves every test green except the
  quarantined live suite. **This is the executable definition of C2.**
- A mutation test that alters `delta_cm` in the fixture must change the template output —
  proving the renderer actually reads the contract rather than reciting boilerplate.
- Guard catches 100% of a curated adversarial corpus of ≥ 30 hand-written bad completions.
- No renderer imports `fitkit.fit_engine` (CI-enforced).

**Integration contract.** In: `FitAssessment v1`. Out: `Explanation`.

---

### Phase 1 (built 5th) — Acquisition and capture-quality gating (`fitkit.acquisition`)

**Responsibility.** Turn raw camera frames plus declared attributes into a validated
`CaptureBundle`, or a structured rejection carrying *actionable* remediation.

**Non-goals.** No measuring. No body model. No image storage policy (that is the
`PhotoStore` port). No UI.

**Public interface.**

```python
class ViewKind(StrEnum): FRONTAL; LATERAL

@dataclass(frozen=True)
class GateVerdict:
    gate_id: str
    passed: bool
    score: float
    remediation: RemediationCode | None

class CaptureQualityGate(Protocol):
    gate_id: str
    def evaluate(self, frame: Frame, view: ViewKind) -> GateVerdict: ...

@dataclass(frozen=True)
class CaptureBundle:
    capture_id: CaptureId
    frontal: PhotoRef
    lateral: PhotoRef
    declared_height: Measure          # sigma from measured self-report error, not zero
    declared_weight: Measure | None
    device: DeviceMetadata
    depth_samples: DepthSample | None # the C3 hook, unused in v1
    gate_report: Sequence[GateVerdict]
```

Gates in v1: full-body framing, pose conformance (limb separation, arm angle), motion blur,
exposure, background separability, subject-camera distance, device verticality from the IMU,
and a clothing-tightness heuristic. The last one matters: BodyM's Test-B degradation is
partly a clothing effect, and loose clothing is the failure mode we can actually prevent at
capture time.

**Dependencies.** `fitkit.domain`. The pose detector sits behind a port.

**Patterns applied.**

| Pattern | What varies | What breaks without it |
|---|---|---|
| **Composite** — `CompositeGate` implementing `CaptureQualityGate` over children | The gate set grows, differs per view, and differs per device class; a merchant may relax one | The orchestrator hard-codes a list and gate composition cannot be tested or configured as a unit |
| **Strategy** — `PoseValidator` | On-device (fast, private, weaker) vs server-side (stronger, costs a round trip) | The choice of where pose validation runs leaks into the capture flow |

**Not applied:** Chain of Responsibility. Fail-fast ordering with per-gate remediation is a
`for` loop over an ordered list. Adding four classes to express `break` is the defect you
described.

**Test strategy.** Each gate unit-tested against a labelled fixture corpus of good and bad
frames (blurred, cropped, backlit, cluttered background, arms down). Composite tested with
stub gates — no images needed — for ordering and short-circuit semantics. Integration test
over a small real-photo corpus asserting the false-reject rate stays under budget, because
a gate that rejects good captures is a conversion killer and needs a number attached.

**Acceptance criteria.**
- Every rejection carries at least one `RemediationCode` that maps to a user-facing
  instruction. A rejection with no remediation fails the test suite.
- False-reject rate on the curated "good capture" corpus < 5%; false-accept rate on the
  curated "bad capture" corpus < 10%. Both numbers are provisional and must be re-set once
  Phase 7 relates gate scores to measurement error.
- `declared_height.sigma_cm > 0` always. A test asserts no code path can construct a
  declared height with zero uncertainty.

**Integration contract.** `CaptureBundle v1`, serializable, with `capture_id` as the join
key that eventually reaches the order record.

---

### Phase 2 (built 6th) — Measurement estimation (`fitkit.measurement`)

**Responsibility.** Turn a `CaptureBundle` into `BodyMeasurements` **with honest
uncertainty**, from a swappable backend.

**Non-goals.** No fit logic. No capture validation. No opinion about garments.

**Public interface.**

```python
class ScaleCalibrationSource(Protocol):
    source_id: str
    def calibrate(self, bundle: CaptureBundle) -> ScaleCalibration: ...
    # ScaleCalibration = (scale_factor_or_reference, sigma_rel, source_id)

class MeasurementBackend(Protocol):
    backend_id: str
    supported_regions: frozenset[BodyRegion]
    def estimate(
        self, bundle: CaptureBundle, calibration: ScaleCalibration
    ) -> BodyMeasurements: ...

class UncertaintyCalibrator(MeasurementBackend):   # Decorator
    def __init__(self, inner: MeasurementBackend, table: ResidualTable) -> None: ...
```

`ScaleCalibrationSource` implementations: `DeclaredHeightCalibration` (v1),
`DepthSensorCalibration` (ARKit/ARCore, v2), `ReferenceObjectCalibration` (a credit card in
frame — a cheap hedge). **This separate port is C3.** Height enters the system as a
calibration *source*, so adding a depth source is a new implementation of one Protocol and
touches no downstream code.

**Patterns applied.**

| Pattern | What varies | What breaks without it |
|---|---|---|
| **Strategy** — `ScaleCalibrationSource` | Where metric scale comes from: declared height now, device depth later, reference object as a hedge | Height gets baked into every backend adapter and C3 requires editing every one of them |
| **Adapter** — one per vendor/model | Wire formats, auth, error shapes, region naming, units | Vendor types reach the domain; C5's "it will be replaced" becomes a rewrite |
| **Decorator** — `UncertaintyCalibrator` | *Where sigma comes from* is independent of *where the measurement comes from* | A vendor returning bare numbers gets sigma = 0 by default and C6 is silently defeated. This decorator is the only reason a black-box API can participate honestly |

`UncertaintyCalibrator` deserves emphasis. Vendors return point estimates. Left alone, the
adapter author picks a sigma and the number is folklore. Instead, sigma is looked up from a
`ResidualTable` produced by Phase 7 from our own tape-measured panel, keyed by backend
version, region, and coarse body-shape bucket, and the table is a versioned artifact
recorded in `inputs_digest`. **Our uncertainty estimate is a measurement, not an opinion.**

**Test strategy.**
- Adapters: unit-tested against recorded vendor responses, including error and partial
  responses. No live calls in CI.
- `UncertaintyCalibrator`: unit-tested with a stub backend and a synthetic residual table.
- A **backend contract test suite** every implementation must pass: units are cm, regions
  are complete or explicitly absent, sigma is positive, provenance is populated, the same
  bundle twice yields the same result (or the backend declares itself non-deterministic).
  This suite is what makes C5 real — a replacement backend is "done" when it passes it.
- Integration: one nightly live call per vendor, quarantined.
- **Fakes the interfaces enable:** `FixedMeasurementBackend` (returns a canned body),
  `PerturbingBackend` (adds known error, for testing abstention end-to-end), and
  `FailingBackend` (for the degradation ladder).

**Acceptance criteria.**
- Swapping backends in the composition root requires zero changes outside that file —
  demonstrated by running the full suite against two implementations.
- No `BodyMeasurements` can be constructed with a zero or absent sigma.
- On the validation panel, empirical 1σ coverage is within [60%, 76%] (nominal 68%). A
  backend whose sigma is not calibrated is not shippable, regardless of its MAE.
- The `ResidualTable` version appears in every emitted `FitAssessment`.

**Integration contract.** In: `CaptureBundle v1`. Out: `BodyMeasurements` + provenance.

---

### Phase 6 (built 7th) — Orchestration and public API (`fitkit.orchestration`)

**Responsibility.** Wire the concrete implementations (composition root), expose the public
operation, own timeouts, retries, idempotency and the degradation ladder.

**Non-goals.** No business rules. If a decision is being made here, it belongs in the fit
engine or a policy.

**Public interface.**

```python
@dataclass(frozen=True)
class AdviceRequest:
    capture: CaptureBundle
    garment_id: GarmentId
    preference: FitPreference
    locale: Locale
    merchant_id: MerchantId

@dataclass(frozen=True)
class AdviceResult:
    assessment: FitAssessment
    explanation: Explanation
    degradations: Sequence[DegradationCode]

class SizeAdvisor:
    def __init__(self, backend: MeasurementBackend, calibration: ScaleCalibrationSource,
                 garments: GarmentRepository, engine: FitEngine,
                 renderer: ExplanationRenderer, store: AssessmentStore,
                 clock: Clock, metrics: MetricsPort) -> None: ...
    def advise(self, request: AdviceRequest) -> AdviceResult: ...
```

Every dependency is a constructor parameter. No singletons, no service locator, no module-
level globals, no framework magic. The composition root is one function that reads
configuration and constructs the graph.

**Patterns applied.**
- **Facade** — `SizeAdvisor`. *What varies:* nothing about it; it is a single stable entry
  point over five subsystems. *What breaks without it:* callers assemble the pipeline
  themselves and the ordering, degradation and persistence rules get reimplemented per
  caller. Honest note: this is a facade because the system genuinely has five subsystems,
  not because a pattern was wanted.
- **Not** applied: Abstract Factory for environment-specific object graphs. A composition
  root with `if settings.env == "test"` is one readable function; an abstract factory
  family would add three classes to express it.

**Test strategy.** End-to-end tests with all ports faked — no network, no database, runs in
milliseconds. Every rung of the degradation ladder gets a test: LLM fails → templated and
`degraded=True`; measurement uncertain → ABSTAIN reaches the user; backend down → typed
error, no fabricated numbers. Idempotency: the same `capture_id` + garment + policy returns
the stored assessment rather than recomputing.

**Acceptance criteria.**
- Full pipeline with fakes runs in < 100 ms and requires no external service.
- Killing the LLM adapter does not fail any non-quarantined test (C2, again, end-to-end).
- Every `advise()` call persists exactly one `FitAssessment`, and its `assessment_id` is
  returned to the caller for the order join.
- No module outside `orchestration` names a concrete adapter class (CI-enforced).

---

### Phase 7 (built 8th) — Offline evaluation harness (`fitkit.evaluation`)

**Responsibility.** Answer two questions with numbers: *are our measurements and our sigmas
honest?* and *does the recommendation reduce size-related returns?*

**Non-goals.** Not an analytics dashboard. Not online experimentation infrastructure
(though it consumes its output).

#### 7a. Measurement accuracy and sigma calibration

- A validation panel of subjects measured with tape by a trained measurer to ISO 8559-1
  landmarks, each also captured through the real protocol.
- Reports per-region MAE, bias, and the error distribution by body-shape bucket — bucketing
  matters because BMnet's own results show error concentrates at high BMI, and a global MAE
  hides a fairness problem.
- **Sigma calibration is the distinctive check:** what fraction of ground-truth values fall
  within ±1σ of our estimate? If that is 40% rather than 68%, our abstention logic is
  decorative and C6 is not being met. This check produces the `ResidualTable` that Phase 2's
  `UncertaintyCalibrator` consumes — so the harness is not merely a report, it is a
  component of the runtime system.

#### 7b. End-to-end size accuracy against purchase and return outcomes

This is the metric that matters and it is the hardest to get right.

**Primary metric:** size-related return rate among orders where a recommendation was shown,
against a randomized control. Requires return-reason codes distinguishing "too small" and
"too big" from taste returns. **Without reason codes this metric does not exist** — that is
Open Question 2 and it is a hard dependency, not a nice-to-have.

**Secondary metrics:** kept-rate; exchange-to-adjacent-size rate; agreement between the
recommended size and the size finally kept on bracketed multi-size orders (a strong signal
that needs no reason codes); abstention rate; coverage.

**The confound you must not skip.** The recommendation changes what people buy, so
recommended-size orders are not comparable to non-recommended ones. Observational
before/after comparison will flatter us. I recommend a **user-level randomized holdout
(~5%)** who see the merchant's own size chart. It is the only clean estimate, and it needs
to be agreed commercially before launch, not retrofitted — Open Question 3.

**Offline replay.** Every `FitAssessment` is persisted with its `inputs_digest`. A candidate
engine or policy version is replayed over the historical corpus, producing counterfactual
recommendations compared against the size actually kept. This is why garment specs are
immutable and versioned and why the contract is versioned: replay is only valid if we can
reconstruct the exact inputs. Replay gives us a cheap offline signal; the holdout gives us
the truth.

**Risk–coverage curve.** Because abstention is a first-class outcome, accuracy alone is
meaningless — a system that abstains on everything is perfectly accurate and worthless. The
harness plots accuracy against abstention rate so merchants can choose an operating point,
and so we can detect the failure mode where we buy accuracy by refusing to answer.

**A ceiling to be honest about up front.** With ~2 cm measurement error against a ~4 cm size
step, a meaningful share of shoppers genuinely sit between sizes and no engine resolves
them. Targets should be framed as *relative reduction in size-related returns versus
control*, never as absolute size-prediction accuracy.

**Patterns applied.** None beyond ports already defined. This phase is scripts, SQL and
statistics; imposing patterns here would be exactly the gratuitous indirection you warned
against.

**Acceptance criteria.**
- Replaying a fixed historical corpus twice produces identical output (determinism, again).
- The harness recomputes published measurement-accuracy figures from raw data with a single
  command — no manual spreadsheet steps.
- Sigma-calibration coverage is reported per region and per body-shape bucket, and CI fails
  the release if any critical region is outside [60%, 76%].

---

## 7. Where I think you are wrong, or at least incomplete

You asked for judgement rather than compliance. Five points, strongest first.

### 7.1 The primary failure mode is probably garment data, not body measurement

C6 names silent false precision in *body* measurement as the primary failure mode. I think
that is the second failure mode.

A garment's real measurement is not a number — it is a distribution. Cutting, sewing and
grading tolerances mean two garments of the same SKU and size differ, and published
tolerances in the 1–2 cm range on circumference measures are routine. That is the same
order as our body measurement error, it is invisible to us, it is not reduced by any model
improvement, and it is *systematically* wrong per factory run rather than randomly wrong per
shopper. If the tech pack says 82 cm and the actual run measures 84 cm, every shopper
recommended that size is wrong in the same direction, and returns will look like a
measurement problem.

**I have designed for this** — `GarmentSizeSpec` uses `Measure`, so specs carry sigma, and
the fit engine propagates it. But it needs a data commitment from you: per-brand grading
tolerance, and ideally spot-measurement of physical samples. Note the uncomfortable
implication: if garment sigma is 1.5 cm and body sigma is 1.5 cm, combined sigma is
~2.1 cm against a 4 cm step, and the honest abstention rate may be higher than the business
will tolerate. Better to discover that in Phase 4 with synthetic data than after launch.

*I am not asking you to relitigate C6 — I am asking to extend it to the garment side.*

### 7.2 C1 protects the arithmetic but not the rhetoric

"The LLM never decides the size" is necessary and not sufficient. An LLM that faithfully
reports "waist −2 cm" and then writes "should still feel comfortable" has changed the
purchase decision without touching a number. My additions: `classification` as a closed
enum in the contract, a controlled vocabulary keyed to it, and a banned-claim guard
alongside the numeric guard. Without those, C1 is satisfied on paper and violated in
effect.

### 7.3 The LLM may not earn its place at launch

C2 makes the LLM optional. I would go further: **ship template-only and A/B the LLM.** The
template renderer is free, exactly reproducible, trivially localizable, and passes legal
review without a conversation. The LLM's value here — nicer prose on a short, highly
structured message — is plausible but unproven, and you now have a harness that can measure
it. Building the LLM path in Phase 5 is right; enabling it by default on day one is a
choice that should be made with data. This is not an argument against C1 or C2; it is an
argument that C2's escape hatch should be the default until the LLM proves itself.

### 7.4 Declared height is a first-class input *and* a first-class error source

C3 makes height first-class. It should also be *uncertain*: people round to the nearest
5 cm, use decade-old figures, and inflate. A declared-height sigma of zero would propagate a
false certainty through every circumference via the shared scale factor — the exact failure
C6 exists to prevent, entering through the calibration door instead of the measurement
door. `declared_height.sigma_cm > 0` is therefore a Phase 1 acceptance criterion, and its
actual value must be measured on the validation panel (declared vs measured height), not
assumed.

### 7.5 Weight is worth more than height, and you have not asked for it

This is evidence-backed, from the numbers in §1.2. On BodyM Test-A, adding height to
multi-view drops waist MAE from 27.32 to 18.71 mm; adding **weight instead** drops it to
13.69 mm. Weight is the stronger single input, and height + weight together give the best
chest and hip figures.

Asking for weight has a real UX and sensitivity cost, and it may be the wrong call for a
fashion audience. But it is currently absent from the flow and, on the evidence, it is the
cheapest available accuracy improvement — cheaper than any backend change. I have left
`declared_weight: Measure | None` in `CaptureBundle` so the option stays open at zero cost.
Open Question 6.

---

## 8. Architecture Decision Records

**ADR-001 — The fit decision is deterministic code; the LLM only renders.**
*Context:* Numeric reasoning by an LLM is unauditable, non-reproducible, and cannot carry an
uncertainty budget. *Options:* (a) LLM decides; (b) LLM decides with tool-calls into a
calculator; (c) deterministic engine, LLM renders. *Decision:* (c), plus a post-generation
numeric-allowlist guard and a controlled vocabulary. *Consequences:* Every recommendation is
reproducible and auditable. Prose is less flexible. The guard is extra machinery that must
itself be tested. Enforced structurally by the import graph, not by convention.

**ADR-002 — Launch on a commercial measurement vendor behind a port.**
*Context:* No candidate has published accuracy on our protocol and population; time to the
return-outcome feedback loop dominates. *Options:* (a) in-house from day one; (b) research
stack; (c) vendor. *Decision:* (c), with the permissive open stack developed in parallel.
*Consequences:* Fast to the loop; per-scan cost; a DPA and likely an Article 9 assessment;
vendor uptime dependency. Reversible via `MeasurementBackend`.

**ADR-003 — Reject SMPL-family models and the BodyM dataset for production.**
*Context:* SMPL/SMPL-X/STAR/SUPR are research-licensed with commercial sub-licensing only
via Meshcapade, and the licence forbids training networks for commercial use; BodyM is
CC BY-NC 4.0. *Decision:* Neither enters the production build; both may inform architecture
and serve as research baselines. Every candidate checkpoint must have documented
training-data provenance before adoption. *Consequences:* We forgo the best-published
research stack; we avoid an existential licensing exposure. A Meshcapade commercial licence
remains a live option if the economics work.

**ADR-004 — Every length is a `Measure` with mandatory, non-zero uncertainty.**
*Context:* C6. *Options:* (a) sigma as an optional field; (b) a parallel confidence object;
(c) a value type that cannot exist without sigma. *Decision:* (c), plus a two-component
correlated-error model (shared scale + independent residual). *Consequences:* Uncertainty
cannot be dropped by accident; more verbose call sites; full covariance is deferred and the
shared-scale approximation is documented as such.

**ADR-005 — Scale calibration is a separate port from measurement.**
*Context:* C3 requires a future depth-based calibration source without downstream change.
*Decision:* `ScaleCalibrationSource` Protocol, with `DeclaredHeightCalibration` in v1.
*Consequences:* One extra indirection now; adding ARKit/ARCore later is a new class and a
wiring change. Justified because the axis of variation is named and near-term.

**ADR-006 — Uncertainty is propagated by fixed-node quadrature, not Monte Carlo.**
*Context:* The engine must be bit-reproducible. *Options:* (a) point estimates plus a
threshold; (b) Monte Carlo; (c) deterministic quadrature. *Decision:* (c) — 5-node
Gauss–Hermite over the shared scale factor, analytic residuals per region. *Consequences:*
Exactly reproducible and cheap; approximates non-Gaussian tails less well than MC; the node
count is a tunable that must be justified against an MC reference in testing (offline only).

**ADR-007 — `FitAssessment` is a versioned, serializable JSON contract with no free text.**
*Context:* This boundary is what enforces C1 and C2. *Decision:* Semver, mandatory
`schema_version`, additive-only minors, closed enums, unknown enum → fail closed to
template. *Consequences:* Renderers evolve independently; the compatibility test matrix
grows; historical documents remain readable, which Phase 7 replay requires.

**ADR-008 — The template renderer is the reference implementation, not a fallback.**
*Context:* C2. If the template path is an afterthought it rots and C2 becomes false.
*Decision:* Write it first; it defines correct output; the LLM is measured against it.
*Consequences:* Some duplicated phrasing effort; a genuinely testable system; the option to
ship without the LLM at all (§7.3).

**ADR-009 — Garment specs are immutable and versioned; measurements carry tolerance.**
*Context:* Replay validity and §7.1. *Decision:* Updates create new versions; old versions
are retained indefinitely; every spec `Measure` has a non-zero sigma from documented grading
tolerance. *Consequences:* Storage growth; correct attribution of returns; honest
uncertainty; possibly a higher abstention rate than expected.

**ADR-010 — Uncertainty estimates are measured, not asserted.**
*Context:* A vendor returning bare numbers would otherwise get sigma by folklore.
*Decision:* `UncertaintyCalibrator` decorator reads a versioned `ResidualTable` produced by
the evaluation harness; the table version is recorded in every assessment. *Consequences:*
The evaluation harness becomes a runtime dependency, not just a report; a new backend cannot
ship until it has been characterised on the validation panel — deliberately.

**ADR-011 — End-to-end accuracy is measured against a randomized holdout.**
*Context:* Recommendations change purchase behaviour, so observational comparison is
biased. *Decision:* ~5% user-level holdout on the merchant's own size chart, plus a hard
requirement on return-reason codes. *Consequences:* A small revenue cost and a commercial
conversation before launch; the only unbiased estimate of the metric that matters.

**ADR-012 — Python 3.12+, standard library first.** *Context:* Minimal dependencies.
*Decision:* Runtime deps limited to `pydantic` **or** stdlib dataclasses + `jsonschema` for
contract validation (decide in Phase 0), `anthropic` for the LLM adapter, `httpx` for vendor
adapters, and the storage driver. Dev-only: `pytest`, `hypothesis`, `import-linter`,
`mutmut`. Each earns its place: schema validation is the `ContractViolation` tripwire;
`import-linter` is what makes the dependency-graph claim checkable rather than aspirational;
`hypothesis` is how the `Measure` invariants get tested properly. *Consequences:* Small
surface, easy to audit. **Pending your confirmation that Python is the target.**

---

## 9. Open questions

Ordered by how much they would change the design.

1. **Garment data.** Do you have real per-size *physical* measurements and fabric stretch
   data from brands, or only marketing size charts? Who supplies grading tolerance? This is
   the biggest single risk to the whole system (§7.1).
2. **Return-reason codes.** Do merchant integrations expose "too small" / "too big" on
   returns? Without them the primary end-to-end metric cannot be computed and Phase 7
   degrades to proxy metrics.
3. **Randomized holdout.** Is a ~5% control group commercially acceptable? If not, we
   should agree now what weaker evidence we will accept.
4. **Launch categories.** Which garment categories at launch? Each needs its own ease
   policy, region weights and validation set. Trousers and dresses are the highest-value
   and hardest; tops are the easiest.
5. **Validation panel.** Can we fund a tape-measured panel (subjects, demographics, budget)?
   Without one, no backend can be characterised and ADR-010 cannot be honoured. This is a
   procurement question with an engineering deadline.
6. **Weight.** Are you willing to ask for it, given it is the strongest single input on the
   verified evidence (§7.5)?
7. **LLM economics.** What is the expected assessment volume, and do you want the default
   model to be `claude-opus-5` or a cheaper tier? I have defaulted to Opus 5; the cost
   trade-off is yours.
8. **Platform.** Native mobile app, or web? A rigid capture protocol and any future
   ARKit/ARCore calibration effectively require native. This changes Phase 1 substantially.
9. **Privacy posture.** Are body photos treated as Article 9 biometric data in your target
   markets? On-device vs server processing, and retention, follow from that answer — and it
   materially affects whether a vendor API is viable at all.
10. **Fit preference.** Do we ask the shopper (tighter / as designed / looser), infer it
    from history, or omit it in v1?
11. **Stretch measurement convention.** Who defines it? "Stretch %" without a stated load
    is not a measurable quantity, and C4 needs it to be one. This likely needs a materials
    person, not an engineer.
12. **Language and runtime.** The repository currently contains an empty `LLM_1.py`.
    Confirming Python 3.12+ (ADR-012) before Phase 0 begins.

---

## 10. What Phase 2 would begin with

On approval, Phase 2 implements **Phase 0 (domain kernel)** in the order you specified:
restate the contract and acceptance criteria, write the tests first, implement, self-
critique adversarially, apply warranted fixes, and run the full suite. It is deliberately
small, has no dependencies, and every subsequent phase is shaped by it.
