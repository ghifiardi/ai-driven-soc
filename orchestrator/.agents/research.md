# Scope
- Target change/bug/feature: (1) Ensure packaging aligns with declared module name so `pip install .` works locally/CI; (2) Pin `rich` to Semgrep-compatible version and bump project version for release 1.0.1 so GitHub Actions picks up fixes.
- Components/Services: `pyproject.toml`, packaged CLI modules in `proactive_security_orchestrator/` (cli/scanner/config/utils).
# Peta File & Simbol (path + [Lx-Ly] + 1-line role)
- pyproject.toml [L1-L26] – Declares build metadata; needs updated version and dependency pin (`rich>=13.7.0,<14.0.0` currently conflicts with Semgrep requirement `~13.5.2`).
- proactive_security_orchestrator/cli.py [L1-L34] – CLI entry point exported via `security-scan` script; already under package but entry point version must be reflected in metadata.
- proactive_security_orchestrator/scanner.py [L1-L32] – Uses `rich` for console output; relies on dependency pin for compatibility.
# Alur Eksekusi end-to-end (linked to lines)
1. Install command uses `pyproject.toml` metadata [L5-L25]; when dependency constraint conflicts (`rich>=13.7.0` vs Semgrep `~13.5.2`), workflow fails resolving requirements.
2. Entry point mapping `security-scan = "proactive_security_orchestrator.cli:app"` [L21-L22] already correct but still ships version 1.0.0, so GitHub Actions caches old broken build until version increments.
3. During GitHub Actions run, Semgrep installs and enforces `rich~=13.5.2`; orchestrator reinstall forces higher version, causing Semgrep failure before CLI executes.
# Tes & Observabilitas (tests, log, how-to-run)
- `pip install .` inside venv should succeed with new version/tag.
- `security-scan --help` post-install to confirm entry point still resolves.
- Optional: run `semgrep --version` in workflow to ensure dependency harmony (observed via GH logs referencing `rich 14.2.0`).
# Risiko & Asumsi
- Assume no other dependencies rely on `rich>=13.7`; pinning to `13.5.2` acceptable for console formatting.
- Version bump to 1.0.1 sufficient to invalidate cached wheel; ensure future releases follow semantic versioning to avoid stale installs.
- Need to ensure packaging config remains stable after modifications; avoid editing other metadata inadvertently.
# Bukti (3–5 mini snippets only)
1. ```
pyproject.toml [L6-L19]
version = "1.0.0"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.7.0,<14.0.0",
    ...
]
```
2. ```
pyproject.toml [L21-L25]
[project.scripts]
security-scan = "proactive_security_orchestrator.cli:app"
[tool.setuptools]
packages = ["proactive_security_orchestrator"]
```
3. ```
GitHub Actions log (Semgrep)
semgrep 1.143.1 requires rich~=13.5.2, but you have rich 14.2.0
```
