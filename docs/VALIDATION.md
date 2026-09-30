# Local validation record

Date: 2026-09-30. This is an implementation verification record, not model benchmark results.

| Check | Observed result |
|---|---|
| Unit, API, parser, source-adapter and resume tests | 76 passed |
| Python statement coverage | 90.19% (947/1050) |
| Editable installation | Passed, with existing dependencies |
| Wheel build | Passed |
| JavaScript syntax check | Passed (`node --check`) |
| Browser UI / actual evaluator API | Two fixture rounds completed; turn, move, submit, abstain, reports verified |
| Browser JavaScript errors | None observed |
| 390px-wide mobile layout | No horizontal overflow observed |
| Local Ruff check | Not run: Ruff absent and network DNS unavailable for installation |
| GitHub Actions | Configuration included; not executed for this delivery because code upload was blocked |
| Real OSV-5M / Mapillary acquisition | Not live-tested; mocked HTTP contract tests passed |
| Real model performance run | Not run; no API credentials or acquired real dataset available |

Browser requests were bridged in-process to FastAPI TestClient because local browser network navigation was unavailable. This verifies UI/API behavior, not TLS, reverse proxies, public hosting or external network transport. Synthetic images were visibly marked as test fixtures and are never reported as real benchmark performance.

Environment: Python 3.13.5, Linux; NumPy 2.3.5, Pillow 12.3.0, Pydantic 2.13.4, FastAPI 0.128.2, HTTPX 0.28.1.

The GitHub connection created the work branch `feat/benchmark-v0.1`, but code-write requests were blocked before a code commit or PR could be made. The repository's `main` remained at `e7b7dd1715f673317e6580d211ecc95af1585cd9`. The accompanying source ZIP and patch contain the complete locally tested implementation. Applying the patch and pushing it is still required for repository installation.
