import subprocess
from rich.console import Console

console = Console()

def safe_run(cmd: list) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Command failed:[/] {' '.join(cmd)}")
        console.print(e.output.decode())
        return ""
