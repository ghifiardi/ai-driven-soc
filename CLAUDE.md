# CLAUDE.md

## What this repo is
GATRA AI-driven SOC: a multi-agent security operations platform on GCP. Agents: ADA (anomaly detection),
TAA (triage & analysis), CRA (containment & response), CLA (continuous learning), RVA (reporting & visualization).

## Where the maintained code lives
- `gatra_intelligence/` (memory, routing, learning; own tests in `gatra_intelligence/tests/`)
- `gatra_a2ui/` (A2UI builders, catalog, components, publisher)
- `multi_tenant_manager.py`, `bigquery_client.py`, `siem_bridge.py`, `config/` (JSON configs), `tests/`
- `docs/` and `.agents/` (documentation / Cursor workflow notes)

The repo root holds hundreds of historical scripts, dashboards, `*_fixed.py` variants, reports and
`.docx`/`.pptx` files. They are **not** to be edited unless an issue names them explicitly.

## Setup and test (mirrors `.github/workflows/ci.yml`)
```bash
# Python 3.11 (pyproject: requires-python >=3.11)
pip install -r requirements.txt || true     # CI tolerates partial failures
pip install pytest pytest-asyncio pytest-cov pytest-mock
export ENVIRONMENT=test
export JWT_SECRET=test-secret-for-ci-only-do-not-use-in-production
export MULTITENANT_CONFIG_PATH=config/gatra_multitenant_config.json
export REDIS_URL=redis://localhost:6379     # CI runs a redis:7-alpine service
pytest tests/ -q
```
CI also runs `pytest tests/ --cov=. -v --tb=short || echo "Some tests failed"`, so CI never fails on tests.
Redis: CI provides a service, but nothing in `tests/` connects to it; `tests/conftest.py` mocks
`redis.asyncio.from_url` (`mock_redis` fixture). Tests passed with no Redis running.

## Test baseline (`pytest tests/ -q`, Python 3.11, CI env vars above, no Redis)
Result: **24 passed, 4 skipped, 0 failed, 0 errors** (28 collected). Every test must pass; a failure is yours to fix.
- `test_botsv3_scenarios.py`: 3 skipped. The BOTSv3 scenario files (`datasets/botsv3/scenarios/{login_anomaly,c2_beacon,exfil_s3}.json`) are not in the repo; the skip reason names them.
- `test_botsv3_scenarios.py::test_botsv3_orchestrator_post_and_classification` is integration-only: it also needs `ORCHESTRATOR_BASE_URL` pointing at a live orchestrator (`/api/v2/orchestrate`).
- `test_optional_dependencies.py` 1 skipped: numpy is installed, so the "numpy missing" case is skipped.

To run the BOTSv3 tests, put the scenario files in place and run `ORCHESTRATOR_BASE_URL=http://host:port pytest tests/test_botsv3_scenarios.py -ra`.

## Hard rules
- Never change `deploy/`, `deployment_package/`, `production/`, `Dockerfile*`, `*.sh`, `.github/workflows/`, `.env*`: `cd.yml` ships to GCP production and the shell scripts SSH into live hosts.
- Never call real GCP, BigQuery, Pub/Sub, Vertex AI, Firestore or Secret Manager; use mocks/fakes in tests (see `tests/conftest.py`). Never add credentials, project IDs or service-account keys.
- Don't run deploy, demo, setup or SSH scripts: they touch live infrastructure.
- `.cursorrules` and `.agents/` describe an interactive Research → Plan → Implement workflow for Cursor. Factory agents don't write to `.agents/`; the issue and PR description take that role.
- Keep changes minimal and don't add dependencies unless the issue requires it.

## Conventions (from `pyproject.toml`)
- Black: line length 100, target py311. isort: profile `black`, line length 100.
- Ruff: line length 100, rules E, W, F, I, B, C4, UP, S, T20, SIM; ignores E501, B008, S101, S104, S105, S106.
  CI lint is advisory (`|| true`).
- pytest: `testpaths = ["tests"]`, `asyncio_mode = "auto"`, `--strict-markers`, markers `unit`, `integration`, `slow`, `security`.
- Coverage `fail_under = 50` is configured but CI does not enforce it.
