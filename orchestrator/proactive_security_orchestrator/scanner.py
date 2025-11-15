import json
from rich.console import Console
from proactive_security_orchestrator.utils import safe_run

console = Console()

def run_scan(root: str, config: dict, output_file: str, output_format: str, allow_empty: bool = False):
    console.print(f"[yellow]Scanning: {root}[/]")

    results = []

    if config.get("semgrep", True):
        console.print("[cyan]Running Semgrep...[/]")
        semgrep_output = safe_run([
            "semgrep", "scan", root,
            "--json"
        ])
        if semgrep_output:
            results.append({"semgrep": json.loads(semgrep_output)})

    if config.get("gitleaks", True):
        console.print("[cyan]Running Gitleaks...[/]")
        gitleaks_output = safe_run([
            "gitleaks", "detect", "--no-banner", "--report-format", "json"
        ])
        if gitleaks_output:
            results.append({"gitleaks": json.loads(gitleaks_output)})

    if not results:
        if not allow_empty:
            raise RuntimeError("No findings produced; re-run with --allow-empty to emit an empty SARIF report.")
        minimal_sarif = {
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "proactive-security-orchestrator",
                        }
                    },
                    "results": [],
                }
            ],
        }
        with open(output_file, "w") as f:
            json.dump(minimal_sarif, f, indent=2)
        console.print("[blue]No findings detected; wrote minimal SARIF report.[/]")
        return

    # Write final SARIF payload with findings
    sarif_payload = {
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "proactive-security-orchestrator",
                    }
                },
                "results": results,
            }
        ],
    }
    with open(output_file, "w") as f:
        json.dump(sarif_payload, f, indent=2)
