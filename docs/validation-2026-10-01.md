# Validation snapshot — October 1, 2026

This records executed checks for the initial forecast platform. It separates
software validation, synthetic experiments, and a small live integration pilot.

| Evidence | Observed result | What it establishes |
| --- | --- | --- |
| Local automated suite | 424 tests passed | Exercised contracts, failures, recovery, and existing application behavior |
| Static checks | Ruff passed; strict mypy passed on 55 source files | Lint and checked Python type consistency |
| Packaging | Wheel and source distribution built; dashboard HTML found in wheel | The installed package includes its report UI |
| Container | Healthy as UID 10001, with `--network none` | The default demonstration and report server run without external access |
| Browser | Comparison, diagnostic selectors, and symbol/model filters worked; no observed console errors | Local dashboard rendering and interactions |
| Offline demo | 9,640 synthetic observations; 2,240 examples; 1,640 labeled outcomes | End-to-end dataset, training, evaluation and artifact workflow |
| Live Jev pilot | 3 requests; 3 later outcomes; 2 abstentions; 0 inference errors | Prospective request/response storage and outcome joining |

The synthetic demo uses eight sessions: four training (820 labeled examples),
one validation (205), and three test (615). All five displayed model variants
use the same test cohort. One is explicitly a deterministic local Jev contract
fixture, not the Jev model. Synthetic scores cannot establish market accuracy.

The live pilot used the pinned `jev-1.13.0` model. Inputs were recorded before
requests and the corresponding outcomes were observed 15 minutes later, within
the configured 60-second receipt tolerance. Provider usage totaled 2,219 input
tokens, with an estimated cost of **$0.000093198** at the configured rate. Two
distributions failed the fixed confidence threshold and were marked abstained;
one passed. No order was placed.

All three realized labels were FLAT. This cohort is too small and class-limited
to compare models, calibrate Jev, estimate generalization, or claim profitability.
No held-out real-market benchmark has been completed. The original raw quotes,
prediction snapshots and pilot report remain local and are not distributed in
this repository. The local pilot report's content ID is
`d6abe64572081105ff2cf63918cb657c03c651d8051c62dbd0f4720bec32706c`.

The final resumed collector invocation completed 20 cycles with 100 successful
quotes, one paced retry, and no final quote failures. Twelve responses were older
than the configured 30-second input limit. They remained recorded for auditing
and were not silently treated as fresh inputs.

Hosted CI results are attached to the pull request. The workflow repeats the
offline checks on Linux/Python 3.12 and verifies the image with networking disabled;
it does not use provider keys or execute paid inference.
