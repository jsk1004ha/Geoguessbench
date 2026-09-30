# Evaluation protocol ggb-1

This document describes **this implementation**, not an assertion of exact compatibility with a commercial game or another paper's full experiment. All tracks are closed-book: no web search, map lookup, geocoder, hidden filenames, ground-truth metadata, or cross-episode memory is an allowed model input.

## Unit of evaluation

A round has one fixed origin node. The model must return the **origin latitude and longitude**, even after moving to other nodes. Runs include every round in the selected split. Round IDs are sorted and shuffled once using `random.Random(seed)`. All models must use the same immutable manifest, split, protocol, implementation and inference settings for a controlled comparison. A subset is explicitly a subset, not a full official test score.

The evaluator hashes the normalized manifest, whose image entries include SHA-256 hashes. It verifies files before serving and records a separate protocol hash incorporating protocol version, configuration and implementation hash. Dependency and Python versions are recorded too; dependency changes can still affect numerical/pixel reproducibility and must be controlled by the experimenter. Read-only mounts are recommended: hashes at startup are not a defense against an administrator changing images during a run.

## Observation boundary

Only the rendered, metadata-free JPEG view, its content hash, camera pose when known, remaining budget/time, local exit ordinals and bearings are exposed. An opaque episode nonce and monotonically increasing step support replay detection; the model prompt omits these identifiers and image hashes. Ground truth, node/sequence/round IDs, source URLs, filenames, countries and candidate destinations are withheld. Coordinates are not encoded in opaque tokens. A model's visual recognition of a published image remains a possible contamination route.

Panoramas use actual inverse-perspective projection of 2:1 equirectangular imagery. The source's horizontal center is its recorded heading; source seam is center + 180 degrees. Positive camera yaw turns right/eastward; positive pitch looks upward. FOV is horizontal. Default view is 960×640, horizontal FOV 90°, pitch 0°. A missing heading for a flat image is reported as unknown, not invented as north. Flat images are fitted/letterboxed without fake camera motion. Panorama provenance must be manually checked against the provider's heading convention when curating a dataset.

## Tracks and budgets

| Config | Exploration | Max accepted/invalid attempts | Cost budget | Wall time per round |
|---|---|---:|---:|---:|
| nmpz.json | none | 0 | 0 | 120 s |
| nm.json | turn, look, zoom | 20 | 20 | 300 s |
| moving.json | turn, look, zoom, move | 20 | 40 | 300 s |
| moving-masked.json | same as moving, source lower band 18% masked | 20 | 40 | 300 s |

The image mask is an **ablation**, not a guarantee of being meta-free: blur, camera generation, stitching and geographic coverage can still leak shortcuts. It modifies source pixels before projection so rotation cannot uncover the masked band.

Example action bodies (one JSON object per model turn):

```json
{"type":"turn","degrees":30}
{"type":"look","degrees":20}
{"type":"zoom","fov":40}
{"type":"move","exit":0}
{"type":"submit","lat":12.34,"lon":56.78,"confidence_25km":0.6}
{"type":"abstain"}
```

`turn` is a relative yaw delta in [-180,180]; `look` is an **absolute** pitch in [-60,60]; `zoom` sets absolute FOV in [20,110]; `move` selects a currently advertised exit. Exit labels are local and do not expose node identities. `submit` requires finite valid latitude/longitude. Optional evidence is a brief observable-clue note, at most 800 characters, not a request for private chain-of-thought. Confidence is optional and means probability of being within **strictly less than 25km**.

Turn/look/zoom cost 1 each, move costs 2. Each consumes one exploration attempt. Invalid/forbidden exploration actions consume one attempt and cost 1; they cannot buy unlimited retries. Once exploration is exhausted, one final submit/abstain remains. An invalid final answer becomes a failed round. Re-sending a stale episode/step is rejected without applying the action twice. No-op legal camera actions still consume their budget.

Wall time begins when a round's first state is created and includes rendering, model calls and transport. Persisted deadlines do not reset after process restart. A response arriving after the deadline is a timeout, even if it contains a correct coordinate. Provider errors are submitted as abstentions with separately recorded error telemetry. Missing answers, invalid final answers, abstentions and timeouts remain in all-round denominators.

## Metrics

Haversine great-circle distance uses R=6371km. The location score is `5000*exp(-distance_km/1492.7)`, with failed rounds scoring 0. This matches the formula in the OSV-5M reference implementation; it is not a claim about every GeoGuessr map's rounding, thresholds or scaling. No arbitrary action-cost subtraction is added to location score.

Accuracy reports strict `< 1,25,200,750,2500km`, matching the reference code's inequality. Score and accuracy divide by **all** scheduled rounds. Valid-only mean/median errors are explicitly labelled; separate failure-penalized errors use `pi*6371km` for failures. This penalty is a reporting convention, not a measured coordinate error.

Bootstrap resamples independent groups, not images assumed independent. Graph components, same capture sequences, duplicate image bytes and supplied round groups are merged into clusters. Percentile 95% intervals use 2000 resamples by default. One cluster produces a degenerate interval and cannot substantiate population generalization. Paired comparison resamples cluster-grouped score differences for exactly matching round IDs and dataset/protocol hashes.

Per-country reporting is **accuracy within 25km grouped by ground-truth country** plus macro averaging over those groups. It is NOT administrative country-classification accuracy. Missing country labels have explicit coverage; country groups are omitted when unavailable. Confidence uses Brier score on valid predictions with supplied probabilities and separately reports confidence coverage. Models could selectively omit confidence, so Brier scores with different coverage are not directly comparable.

Token usage is self-reported from the model adapter/provider response, not authoritative billing. Missing usage is null, not zero; known totals and coverage are separate. Environment action costs are measured independently by the evaluator. The report includes provider-returned model IDs in private traces when available; a requested alias does not prove an immutable model snapshot.

## API and storage

- `POST /api/runs`: `{model, settings}` -> run ID and fingerprints.
- `POST /api/runs/{id}/next`: returns the current observation (or starts the next unfinished round); never resets an active round.
- `POST /api/runs/{id}/step`: `{episode, step, action, telemetry?}` -> next observation/completion.
- `GET /api/runs/{id}/report`: only after completion; aggregate metrics, settings, runtime and usage, no per-round coordinates.

The server is a single Uvicorn worker with SQLite WAL and an in-process lock. Do not run multiple workers against this snapshot-based store. Snapshot persistence ensures restarts don't grant additional attempts. The agent saves exact visible context and pending action before submission; resume replays that pending action only if its episode/step is still current, otherwise it recognizes an already-applied step. It refuses resume with different settings or without its checkpoint. A resume after a real deadline is still a timeout.

Public reports are intentionally non-interactive with respect to ground truth. Local evaluator-only `export --private` contains targets, guesses and action traces. Keep it off the agent filesystem. Reports from different datasets, budgets, masks or code hashes are grouped separately by the local leaderboard. Inference settings are also grouped, except model name and endpoint. Pairwise comparison fixes dataset/protocol but leaves inference settings to the experimenter: explicitly disclose any intended treatment differences.

## Security and scientific limits

This is local research infrastructure, not a hostile multi-tenant competition service. Optional bearer authentication is shared evaluator access, not per-user authorization. Anyone with the shared token/run ID can control that run; use separate evaluator instances or add ownership/rate limiting for independent users. Source parsing assumes evaluator-curated manifests and provider endpoints. Hashes and JSON schema validation do not replace process isolation, TLS, network egress controls, secret management or a signed submission system.

Run untrusted agents in a separate container/host with only evaluator access and any explicitly permitted model endpoint. The repository does not execute submitted arbitrary code, does not expose its data directories as static files, and does not permit web search tools in its adapter. These limits do not make an agent with independent shell/network access unable to cheat. Splits and frozen data improve reproducibility but cannot prove absence from proprietary pretraining data.
