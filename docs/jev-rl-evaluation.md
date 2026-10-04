# Jev-assisted recurrent policy evaluation

Status: implementation reviewed and real paid cache acquisition underway under an explicit failure-preservation amendment. No neural or forecast-quality result is claimed yet. The user explicitly approved API spending on October4,2026.

## Frozen comparison

The preceding MARKET_ONLY study selected one-layer256-unit RecurrentPPO but did not earn promotion. This follow-up fixes that architecture and compares two151-dimensional policies across seeds42/43/44:

- CONTEXT_ONLY: market and ledger observations plus the same forecast-age/target/anchor/availability metadata, with valid forecast probabilities neutralized to uniform and confidence zero.
- JEV_ASSISTED: the identical network and metadata, adding validated raw Jev probabilities/confidence.

Primary TEST results average all three seeds; the best TUNE checkpoint per profile is a separately reported diagnostic. Both checkpoint choices freeze before TEST access. No hosted Jev weights are trained and no broker orders are created.

Source:81 AlpacaSIP/raw sessions from June30–October22,2025; June30 is reference warmup. TRAINJuly1–September24 has60dates, TUNESeptember25–October8 has10, TESTOctober9–22 has10. Five stocks are AAPL,AMZN,MSFT,NVDA,TSLA. A timestamp-only preflight found NFLXmissingrequiredminutegriddata in61sessions; its initialsource/registration are preserved privately. NFLXwas replaced before any paid forecasts or quality scoring. Both arms use the same revised universe. Results cannot be directly compared with the prior March cohort.

## Paid forecasting and timing

One request per session contains five stocks at the SAME as-of61minutes after open. Fifteen typed Choice questions cover15minutes,60minutes andsessionclose. The exact target is minute-bar-close proxy return relative to the as-of completed close; DOWNbelow−10bps, FLATinclusive±10bps, UPabove+10bps. Arithmetic, ratios andtargetminutes are computed locally; compact normalized recentbars/context mask actualtickers,calendardates andabsoluteprices.

API limits remain100global attempts,30-secondcooldown,10attempts/$0.05pergroup andzero providerretries. The registered experiment additionally enforces80attempts/$0.25conservativeestimatedcost atomically, withpermanentbatchuniqueness. Unknownusage reserves64,000inputtokens; validreportedusage is settled even oninvalidanswers. Outputtokens are free at the [published pricing](https://typesafe.ai/blog/introducing-system-one-models-and-jev). Actual generation/completiontimes remain UTCtoday. Historical replay availability is separately hypothesized at as-of+60seconds; no historical real-time delivery is claimed.

Forecast fields are zero/masked before hypothesized arrival andaftertargetexpiry. Lowconfidence validprobabilities remain available as features; availability is not selected by modelconfidence. Rawproviderconfidence is not empirical marketaccuracy. Forecastscorecards separately report horizon-specific argmaxaccuracy,logloss,Brier andselectedaccuracy/coverage at min(maxprobability,confidence)>=0.6 against TRAIN-fitted classpriors. Priors are never fed to policytraining, andqualitygrades occur after policyselection seals.

## Evaluation and evidence

TRAIN-only CPUthroughput sets the same bounded step budget for allsixlearners, capped at92,160steps and600learningseconds each. Full steps/updates, elapsedtime, identicalparametercounts, checkpoint/code/lock/data/cache identities must pass beforeTUNE. All scored episodes must resolve. TESTuses mean-seed paired date-bootstrap comparisons against neutral andcash, controls through the same ledger, drawdownlimits and doubled/quadrupled cost stress. Negative results and allfailedruns remain visible; settings are not retuned after TEST.

Independent source audit verified33rawpagehashes/pagination links andevery retained record:320,008downloadedrows,162,958regular-sessionexclusions,157,050retainedbars. All81dates havecomplete minutegrids. All400maskedcontexts/1,200forecastmappings andall400prepared episodes matchedsource. Independently rebuilt98,400TRAINrows, fittedmeans/scales andmissingmasks exactly. The implementation passed701full-suite tests, Ruff andstrictmypy across76sourcefiles; independentreview corrected validationtolerance anddeadlineguard defects before paid acquisition.

Vendorpretrainingcutoff is unknown. Masking dates/tickers/prices reduces recognizable-history exposure but cannot establish that Jev has never seen related historical outcomes. Ten TESTdates, revisedhistoricalbars, synthetic delayed fills/capacity/costs, thefixed surviving-stock universe andindependentdailycapital resets limit conclusions. This is a hypothetical retrospective policy experiment, notprospective accuracy oroperationalprofit evidence.

Rawbars,normalizedvectors,individualforecasts,ledgers andcheckpoints remainprivate. Publishedresults will be aggregateonly; there is no productionpromotion or80%accuracy claim without measured support.

## Acquisition amendment

The initial collector stopped after six requests when one response failed strict validation. All six attempts and their actual usage were preserved. Before any neural training or forecast-quality grading, an explicit amendment authorized only the remaining never-requested batches. Successful and failed requests are never retried; global, per-group and experiment caps are unchanged, and all provider payloads retain their original hashes.

A failed batch produces unavailable forecast records with zero sentinels, never fabricated normalized probabilities. Every forecast field is masked in both policy arms; all registered market episodes remain included. Forecast reports expose provider errors across the full cohort, conditional scores on available forecasts, and accuracy counting unavailable forecasts as incorrect. Matched available-case priors and full-cohort priors are shown separately.

The original registration's abort policy remains archived unchanged. The amended cache explicitly binds the override, permanent reservations, exclusive collector lease and exact saved-response reconciliation against the durable ledger. This is an acquisition-availability adaptation; it follows no observed accuracy, trading return or model-selection result.
