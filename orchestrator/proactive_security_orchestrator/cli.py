from typing import Optional

import typer
from rich.console import Console

from proactive_security_orchestrator.config import load_config
from proactive_security_orchestrator.scanner import run_scan

app = typer.Typer(help="Proactive Security Orchestrator CLI")
console = Console()

@app.command()
def scan(
    path: str = typer.Argument(".", help="Target directory to scan"),
    format: str = typer.Option("sarif", "--format", help="Output format"),
    output: str = typer.Option("findings.sarif", "--output", help="Output file"),
    config: str = typer.Option("SECURITY_SCAN_CONFIG.yml", "--config", help="Config file"),
    allow_empty: Optional[bool] = typer.Option(
        False,
        "--allow-empty",
        help="Allow emitting an empty SARIF file even when no findings exist.",
    ),
):
    """
    Top-level orchestrated security scan:
    - Semgrep
    - Gitleaks
    - Custom rules
    """
    console.print("[bold green]Starting security orchestration...[/]")
    cfg = load_config(config)

    run_scan(
        root=path,
        config=cfg,
        output_file=output,
        output_format=format,
        allow_empty=allow_empty,
    )

    console.print(f"[bold blue]Scan complete. Results → {output}[/]")
