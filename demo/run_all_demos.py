#!/usr/bin/env python3
"""
Run all demo scenarios and save outputs for social media content.
"""

import subprocess
import os
from datetime import datetime

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(DEMO_DIR, "recordings")

def run_demo(mode, duration=None, output_file=None):
    """Run a demo and save output."""
    cmd = ["python3", os.path.join(DEMO_DIR, "demo_simulation.py"), "--mode", mode]
    if duration:
        cmd.extend(["--duration", str(duration)])

    if output_file is None:
        output_file = os.path.join(RECORDINGS_DIR, f"demo_{mode}_{datetime.now().strftime('%H%M%S')}.txt")

    print(f"\n{'='*60}")
    print(f"Running: {mode} demo")
    print(f"Output: {output_file}")
    print('='*60)

    with open(output_file, 'w') as f:
        result = subprocess.run(cmd, capture_output=True, text=True)
        output = result.stdout + result.stderr
        # Filter out deprecation warnings
        lines = [l for l in output.split('\n') if 'DeprecationWarning' not in l and 'datetime.datetime.utcnow' not in l]
        clean_output = '\n'.join(lines)
        f.write(clean_output)
        print(clean_output)

    return output_file

def main():
    os.makedirs(RECORDINGS_DIR, exist_ok=True)

    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║        GENERATING ALL DEMO RECORDINGS FOR SOCIAL MEDIA      ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    recordings = []

    # Run multiple detection scenarios
    for i in range(3):
        output = run_demo("detection", output_file=os.path.join(RECORDINGS_DIR, f"demo_detection_{i+1}.txt"))
        recordings.append(output)

    # Run event stream
    output = run_demo("stream", duration=10, output_file=os.path.join(RECORDINGS_DIR, "demo_stream.txt"))
    recordings.append(output)

    print(f"\n{'='*60}")
    print("ALL RECORDINGS COMPLETE!")
    print('='*60)
    print("\nFiles created:")
    for r in recordings:
        print(f"  - {r}")
    print(f"\nLocation: {RECORDINGS_DIR}")

if __name__ == "__main__":
    main()
