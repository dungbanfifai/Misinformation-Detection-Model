"""
run_pipeline.py — run the pipeline notebooks locally, in stage order.

Usage (from the project folder, with the conda environment active):
    python run_pipeline.py              # run every stage that exists
    python run_pipeline.py 01           # run only stage 1 (01a-01d)
    python run_pipeline.py 01 02 04     # run several stages

Each notebook is executed in place, so its outputs (tables, plots) are saved inside it.
Notebooks are found by their number prefix, so stages added later are picked up automatically.
On Kaggle this script is not used: each notebook is run with "Save & Run All".
"""
import sys
import time
import subprocess
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
NOTEBOOKS = PROJECT / "notebooks"
STAGES = ["01", "02", "03", "04", "05", "06", "07"]
TIMEOUT_SECONDS = 4 * 3600   # per notebook (DistilBERT on CPU is slow)


def notebooks_for(stage):
    return sorted(NOTEBOOKS.glob(f"{stage}*.ipynb"))


def run(nb):
    print(f"\n=== {nb.name} ===", flush=True)
    start = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         f"--ExecutePreprocessor.timeout={TIMEOUT_SECONDS}", str(nb)],
        cwd=NOTEBOOKS)
    if result.returncode != 0:
        sys.exit(f"FAILED: {nb.name}. Open it in Jupyter to see the error; later stages were not run.")
    print(f"done in {time.time() - start:.0f}s")


def main():
    requested = sys.argv[1:] or STAGES
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        sys.exit(f"Unknown stage(s) {unknown}; choose from {STAGES}")
    for stage in requested:
        nbs = notebooks_for(stage)
        if not nbs:
            print(f"(stage {stage}: no notebooks yet, skipped)")
            continue
        for nb in nbs:
            run(nb)
    print("\nAll requested stages finished. Outputs are in", PROJECT / "outputs")


if __name__ == "__main__":
    main()
