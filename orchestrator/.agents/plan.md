# Goal & Non-Goals
- Goal: Update packaging metadata to release version 1.0.1 with Semgrep-compatible `rich` pin and ensure entry point remains accurate so GitHub Actions installs working build.
- Non-Goals: Modify CLI functionality, change scanner logic, or edit workflows beyond dependency/version adjustments.
# Perubahan per File
- file: pyproject.toml
  - Lokasi: [L1-L26]
  - Perubahan: bump `[project] version` to `1.0.1`; pin dependency `rich==13.5.2` (or `>=13.5.2,<13.6.0` per instruction); keep other metadata unchanged to avoid regressions.
  - Mengapa: GitHub Actions needs new version to invalidate caches, and Semgrep requires `rich~=13.5.2` (research pyproject.toml [L6-L19]).
  - Dampak: `pip install .` will produce new wheel; workflows should succeed without dependency conflict.
- file: .agents/progress.md
  - Lokasi: append entries per step
  - Perubahan: document execution steps per guardrails.
  - Mengapa: maintain traceability.
# Urutan Eksekusi (Step 1..n + "uji cepat" per step)
1. Edit `pyproject.toml` to update version and dependency pin; uji cepat: `rg "rich" pyproject.toml` to confirm constraint plus `rg "version"` to ensure bump.
2. Reinstall package inside venv (`pip install .`) verifying new version and dependency resolution; uji cepat: `pip show proactive-security-orchestrator` displays 1.0.1 and `Requires: rich (==13.5.2)`.
# Acceptance Criteria (incl. edge-cases)
- `pyproject.toml` reflects `version = "1.0.1"`.
- Dependencies list pins `rich` to `>=13.5.2,<13.6.0` (or exact 13.5.2) preventing Semgrep mismatch.
- Local reinstall completes without errors and `security-scan --help` still works.
- Document progress per guardrails.
# Rollback & Guardrails (feature flag/circuit breaker)
- Rollback by resetting `pyproject.toml` to previous version/dependency entries.
- Guardrail: use venv install test before pushing; verify `pip show` result to ensure correct metadata.
# Risiko Sisa & Mitigasi
- Risk: future dependencies might need richer features from newer `rich`. Mitigation: test Semgrep compatibility before bumping again and coordinate with workflow.
- Risk: forgetting to push new tag means GitHub might still install cached wheel; mitigate by version bump and verifying in logs.
