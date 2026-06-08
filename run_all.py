"""
Run the full XAI-TMS pipeline end to end and collect all outputs.

Usage:
    python run_all.py
    python run_all.py --only validation_configA shap_analysis
"""

import sys
import subprocess
import argparse
from pathlib import Path

STAGES = [
    "validation_configA.py",
    "autoencoder_configB.py",
    "shap_analysis.py",
    "counterfactual_generation.py",
    "benchmark_xai_tms.py",
    "statistical_test.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these stage scripts (filenames)")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="keep going if a stage fails")
    args = ap.parse_args()

    here = Path(__file__).parent
    stages = args.only if args.only else STAGES
    print(f"Pipeline: {stages}\n")

    for s in stages:
        path = here / s
        if not path.exists():
            print(f"[skip] {s} not found")
            continue
        print(f"\n{'='*60}\n[run] {s}\n{'='*60}")
        rc = subprocess.run([sys.executable, str(path)]).returncode
        if rc != 0:
            print(f"[error] {s} exited with code {rc}")
            if not args.continue_on_error:
                sys.exit(rc)

    print("\nDone. Collected outputs in ./outputs_* directories:")
    for d in sorted(here.glob("outputs_*")):
        files = sorted(p.name for p in d.glob("*"))
        print(f"  {d.name}/: {files}")


if __name__ == "__main__":
    main()
