# Supervised forecast validation — October 4, 2026

The direct supervised comparison and cached-Jev ablation were executed. Their
results do not meet the 80% selective target or establish a supported neural/Jev
improvement. The research policy abstains and no model is promoted.

| Executed check | Result |
| --- | --- |
| Full local regression suite, including fixture HTTP servers | 870 passed in 135.78 seconds |
| Whole-repository Ruff | Passed |
| Strict mypy, source plus six preparation/training/study/rendering scripts | Passed on 86 source files |
| Independent source timestamp audit | All 46 raw pages and 216,450 regular bars verified; complete 111-session grids |
| Independent input audit | All 34,650 windows/input IDs; exact Float32 arrays; 1,650 fixed static/source samples across all stock/date/role groups |
| Independent result replay | All 81,900 saved TEST probability rows matched exactly; maximum metric difference 2.22e-16 |
| Independent cached-Jev replay | All 400 cases and four arms; all probabilities and metrics matched exactly |
| Scientific charts | Four actual PNGs rendered and visually checked |
| Publication boundary | Aggregate results, settings and hashes only; source rows, targets, prediction vectors, checkpoints and credentials remain private |
| New Jev calls / spend / broker orders | 0 / $0 / 0 |

The first restricted local suite passed 799 tests while 42 HTTP tests were unable
to bind localhost. The affected files passed all 60 tests with localhost access;
the final complete run above passed all 870 tests under that same permission.
Existing warnings concern the legacy WebSockets namespace and the simulator's
unbounded observation space; no warning was converted into a model-quality claim.

Three independent semantic reviewers covered data, models/dependencies and
study/fusion contracts. Two defects were independently reproduced and fixed
before any real fit: the full registration's uncertainty unit and the nested
synthetic-data marker. The repair and result renderer received an independent
review; no findings remain unresolved. The input and result audits are separate
from these source reviews.

Training source was frozen at
`5dc2bf1cd7b3d4b0dde5f7f9b12491fbbe5ac932`. All six neural fits finished with the
registered patience rule; none hit a cap. Source/dependency/input hashes remained
unchanged throughout fitting and scoring. Earlier-stage summaries and retained
checkpoints were replayed; non-improving epoch weights were not retained, so their
history was checked for consistency rather than replayed from weights.

These are local execution proofs. Hosted validation is reported by the
[GitHub Actions workflow](https://github.com/Vajraaaang/tradecopilot/actions/workflows/ci.yml)
on the published commit; this document does not infer hosted status from local
success. Future confirmation requires new dates, because the scored source
period is recorded in the [retirement artifact](results/2026-10-04-supervised-forecast/corpus-retirement.json).

The [study](supervised-forecasting.md),
[aggregate results](results/2026-10-04-supervised-forecast/summary.json),
[Jev ablation](results/2026-10-04-supervised-forecast/jev-fusion-summary.json),
[class diagnostics](results/2026-10-04-supervised-forecast/jev-fusion-class-diagnostics.json)
and [independent audit](results/2026-10-04-supervised-forecast/independent-audit.json)
provide the measured evidence and limitations.
