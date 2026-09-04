# fitkit

Garment size prediction for fashion e-commerce. Given two photographs of a shopper, their
declared height, and the *real physical measurements* of a garment, it answers: **which
size should I order, and how will it fit?**

The distinctive property is not the measurement — it is the honesty about it. Body
measurement from smartphone images carries an error of roughly 1–2 cm on waist and hip,
against a typical inter-size step of about 4 cm. A system that hides that error will
confidently recommend the wrong size. Here, every measurement carries an uncertainty, the
fit engine propagates it, and when it exceeds what the size step can absorb the engine
**abstains or widens to two sizes** rather than guessing.

The full design, including the measurement-backend evaluation, the licensing findings and
twelve ADRs, is in [`docs/architecture/phase-1-design.md`](docs/architecture/phase-1-design.md).

## The five stages

| Stage | Package | What it does |
|---|---|---|
| Acquisition | `fitkit.acquisition` | Gates the capture on framing, distance, sharpness, tilt and pose. A rejection always carries an instruction the shopper can act on. |
| Measurement | `fitkit.measurement` | Estimates body measurements through a swappable backend, then replaces the vendor's claimed precision with a *measured* residual. |
| Catalogue | `fitkit.catalog` | Immutable, versioned garment specs: physical measurements and fabric stretch, not marketing size charts. |
| Fit engine | `fitkit.fit_engine` | Deterministic arithmetic: ease, stretch absorption, ranking, uncertainty propagation, abstention. **No LLM.** |
| Explanation | `fitkit.explanation` | Turns the computed result into prose. Template by default, LLM optional. |

`fitkit.orchestration` wires them; `fitkit.evaluation` measures whether any of it works.

## Install and run

```bash
python -m pip install -e ".[dev]"     # add ".[dev,llm]" for the Anthropic renderer
pytest                                 # 434 tests, no network, under a second
coverage run -m pytest && coverage report
lint-imports                           # the architectural contracts
python examples/quickstart.py          # one run of the whole pipeline
python examples/explore.py --help      # turn the knobs and watch the verdict move
```

## A run, end to end

[`examples/quickstart.py`](examples/quickstart.py) executes the whole pipeline with no
network, no database and no LLM. The four classes at its top are the only things a
deployment must supply: a CV stack, object storage, an assessment store, and a measurement
backend.

```python
capture = CaptureAssembler(Analyzer(), Photos()).assemble(
    "cap_001",
    RawCapture(frontal=b"<jpeg>", lateral=b"<jpeg>", declared_height_cm=175.0,
               declared_weight_kg=None, device=DeviceMetadata("ios", "iPhone15,2", "1.0.0")))

advisor = build_advisor(garments=InMemoryGarmentRepository(spec), store=Store(),
                        residuals=residuals, default_policy=policy, backend=Backend())

result = advisor.advise(AdviceRequest(capture=capture, garment_id="brand:sku-1",
                                      merchant_id="demo", locale="en"))
```

Its actual output:

```
SINGLE: size 50 (confidence 75%)
  hip: +6.0 cm (sigma 1.9) RELAXED
  waist: +4.0 cm (sigma 1.8) RELAXED
Order the 50. Confidence: 75%. Hip: a little relaxed, by about 6 cm. Waist: a little
relaxed, by about 4 cm. Treat the hip and waist figures as approximate.
degradations: []
```

[`examples/explore.py`](examples/explore.py) puts a flag on every input that changes the
answer. Raising the backend's measured error walks the whole degradation ladder on one
body — `--residual 1.8` gives SINGLE, `2.0` gives TWO_SIZES, `2.4` gives ABSTAIN — which is
C6 doing its job rather than being asserted. Switching `--stretch none` to `--stretch high`
on the same body turns a waist that was 2 cm MUCH_TOO_TIGHT into 2 cm absorbed by the
fabric and merely TIGHT, which is C4.

## The constraints, and where they are enforced

The design was written against six non-negotiable constraints. Each one is checked by
something that fails loudly, not by a convention:

- **C1 — the LLM never decides the size.** `fitkit.explanation` is forbidden by an
  import-linter contract from importing `fitkit.fit_engine`. It receives only the
  serialised `FitAssessment`, and `NumericGuard` rejects any generated sentence containing
  a number that is not in the assessment's own allowlist.
- **C2 — the LLM is removable.** `TemplateRenderer` is the default, not a fallback:
  `build_advisor` with no LLM client returns a fully working system. A CI job runs the
  whole suite with `anthropic` deliberately absent.
- **C3 — scale calibration is explicit.** `ScaleCalibrationSource` is a port.
  `DeclaredHeightCalibration` implements it today; an ARKit/ARCore depth source implements
  it later and nothing downstream changes, because backends consume a `ScaleCalibration`,
  never the thing that produced it.
- **C4 — fabric stretch is a first-class input.** `fit_engine/stretch.py` converts stretch
  class and recovery into an absorbable capacity in centimetres, so the same 2 cm delta on
  rigid denim and on elastane jersey produce different verdicts.
- **C5 — the measurement backend is swappable.** `MeasurementBackend` is a Protocol, and
  `tests/measurement/contract.py` is a reusable suite every implementation must pass.
- **C6 — no silent false precision.** A centimetre is never a bare `float`: it is a
  `Measure`, which refuses a non-positive sigma. An AST test in `tests/test_architecture.py`
  fails the build if a `..._cm: float` field appears in the domain outside the wire format
  and a short list of files that must state, in the file, why they are exempt. A backend
  with no measured residual table fails closed rather than inventing a sigma.

## The contract

`fitkit/domain/contracts/fit_assessment.py` defines `FitAssessment`: a versioned,
serialisable document carrying the verdict, ranked sizes with confidence, a per-region
delta in centimetres with its sigma, and an `inputs_digest` recording the engine, policy,
garment-spec, backend and residual-table versions in force. It is the boundary between
arithmetic and prose, and it is what makes the recommendation reproducible years later —
which is the precondition for the evaluation harness attributing a return to a
recommendation.

## Repository layout

```
src/fitkit/
  domain/          the kernel: units, regions, fabric, policy, errors, ports, the contract
    contracts/     FitAssessment and Explanation — the versioned wire format
  acquisition/     capture gating and CaptureBundle assembly
  measurement/     calibration, vendor adapter, residual calibration
  catalog/         garment spec import, build and versioned storage
  fit_engine/      ease, stretch, scoring, quadrature, abstention
  explanation/     template renderer, LLM renderer, output guards
  orchestration/   SizeAdvisor facade and the composition root
  evaluation/      accuracy, sigma calibration, replay, outcome metrics
docs/architecture/ the Phase 1 design document
examples/          a runnable end-to-end demonstration
```

The domain depends on nothing — not on the rest of the project, not on any third-party
package. The package has **zero runtime dependencies**.

## Status

Every phase of the design is implemented and tested. Three things are deliberately absent
and are decisions rather than omissions:

- **No measurement vendor is selected.** `VendorMeasurementBackend` is a generic adapter
  over an injected `HttpTransport`; the choice is a licensing and procurement question,
  discussed in §1 of the design.
- **No residual table ships.** It has to be measured on a tape-measured validation panel.
  Until it exists, `UncertaintyCalibrator` refuses to produce measurements — which is the
  intended behaviour.
- **No HTTP service.** The public API is the `SizeAdvisor` facade; exposing it over HTTP is
  a deployment concern with no business logic in it.

## Licence

See [`LICENSE`](LICENSE).
