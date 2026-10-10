"""Convenience runner for the Data Quality Guardian pipeline."""

import subprocess
import sys
from pathlib import Path


def run_pipeline():
    project_root = Path(__file__).resolve().parent

    cmd = [
        sys.executable,
        str(project_root / "src" / "guardian.py"),
        "--input",
        str(project_root / "data" / "retail_corrupted.csv"),
        "--out",
        str(project_root / "reports"),
        "--evaluate",
        str(project_root / "data" / "injected_errors.json"),
    ]

    print("🛡️ Running Data Quality Guardian Pipeline...\n")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Pipeline completed successfully.")
        print(f"📁 Reports saved to: {project_root / 'reports'}")
    else:
        print(
            f"\n❌ Pipeline failed with exit status: {result.returncode}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    run_pipeline()
