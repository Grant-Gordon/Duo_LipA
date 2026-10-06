"""Run a smoke matrix: python scripts/smoke.py configs/smoke/scratch_matrix.yaml [--out runs/smoke/NAME]"""

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

from duo_lipa.harness.smoke import expand_runs, run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="substring filter on run names")
    ap.add_argument("--no-decode", action="store_true")
    a = ap.parse_args()
    spec_path = Path(a.spec)
    spec = yaml.safe_load(spec_path.read_text())
    runs = expand_runs(spec, spec_path.parent)
    if a.only:
        runs = [r for r in runs if a.only in r[0]]
    out = Path(a.out) if a.out else ROOT / "runs" / "smoke" / f"{spec_path.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    res = run_all(runs, ROOT, out, decode=not a.no_decode)
    n_ok = sum(1 for r in res if r.get("ok"))
    print(f"\n{n_ok}/{len(res)} runs ok -> {out}/results.md")
    sys.exit(0 if n_ok == len(res) else 1)


if __name__ == "__main__":
    main()
