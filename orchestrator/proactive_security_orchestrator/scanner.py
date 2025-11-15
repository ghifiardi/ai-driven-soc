import json
from rich.console import Console
from proactive_security_orchestrator.utils import safe_run

console = Console()

def run_scan(root: str, config: dict, output_file: str, output_format: str):
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

    # Write final SARIF
    with open(output_file, "w") as f:
        json.dump({"results": results}, f, indent=2)
