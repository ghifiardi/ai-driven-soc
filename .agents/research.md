# Scope
- Target change/bug/feature: Security scan workflow in `ghifiardi/ai-driven-soc` fails when invoked from GitHub Actions because the job’s “Install dependencies” step blindly runs `pip install -e .` inside the target repository, which lacks `setup.py`/`pyproject.toml`. Need to delegate scanning to the reusable orchestrator workflow that already handles package installation fallbacks.
- Components/Services: `.github/workflows/security-scan.yml` (caller workflow), orchestrator CLI invoked via `security-scan scan` (installed remotely), `.agents/*` artifacts for spec-first tracking.

# Peta File & Simbol (path + [Lx-Ly] + 1-line role)
- `.github/workflows/security-scan.yml [L1-L87]` – defines local CI jobs; `security-scan` job still contains manual steps that fail without packaging metadata.
- `src/cli.py [L27-L120]` – Typer CLI invoked by the workflow; proves that we only need the orchestrator package itself, not an editable install of the caller repo.
- `README.md [L16-L47]` – documents how to run scans via CLI; relevant when explaining fallback behavior.

# Alur Eksekusi end-to-end (linked to lines)
1. Workflow triggers on PR/push (`security-scan.yml [L3-L9]`) and runs the local `security-scan` job (`[L10-L57]`).
2. Job checks out repo, installs Python/semgrep/gitleaks, then executes `pip install -r requirements.txt && pip install -e .` (`[L21-L40]`). Because `ai-driven-soc` lacks packaging metadata, pip aborts with “does not appear to be a Python project”.
3. As a result, subsequent `python -m src.cli scan .` step (`[L41-L44]`) never runs and no SARIF is produced. Upload steps fail because `findings.sarif` is missing (`[L45-L56]`).
4. We already maintain the scanner as a reusable workflow in `ghifiardi/proactive-security-orchestrator/.github/workflows/security-scan.yml`. Calling it via `uses:` (with `working-directory` input) would install the orchestrator from git when needed, solving the issue without duplicating logic.

# Tes & Observabilitas (tests, log, how-to-run)
- Validate CLI import path unaffected: `pytest tests/test_cli.py -k scan -v` (ensures orchestrator package is functional after workflow changes).
- Optional: dry-run GitHub workflow locally using `act -j security-scan` to ensure `uses:` call resolves.
- Logs: expect called workflow to print “Determine install strategy” followed by either local or git install message; presence of those logs confirms we’re on the new path.

# Risiko & Asumsi
- Assumption: GitHub repository permissions allow `security-events: write` so SARIF uploads succeed when using the reusable workflow.
- Risk: Removing local steps might break developers wanting custom behavior. Mitigation: keep local `test` job intact and only delegate scanning job.
- Risk: `uses:` reference could drift if orchestrator workflow changes. Mitigation: pin to `@main` for now; can pin to a commit if stability required.
- Risk: Need to ensure `working-directory` is passed (default `.`) so the scanner runs on the caller repo. Will set explicitly.

# Bukti (3–5 mini snippets only)
1. `.github/workflows/security-scan.yml [L26-L44]`
   ```
   - name: Install dependencies
     run: |
       pip install -r requirements.txt
       pip install -e .
   - name: Run security scan
     run: |
       python -m src.cli scan . --format sarif --output findings.sarif || true
   ```
   → Workflow assumes repo is a Python package; fails for ai-driven-soc.
2. Failure screenshot (user) confirms error message `does not appear to be a Python project`, aligning with the above step.
3. Existing reusable workflow already exposes `workflow_call` with `working-directory` input, so switching to `uses: ghifiardi/proactive-security-orchestrator/.github/workflows/security-scan.yml@main` removes the dependency on local packaging.
