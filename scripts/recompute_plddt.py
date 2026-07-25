#!/usr/bin/env python3
"""
recompute_plddt.py
--------------------
One-off fix script: recomputes mean_plddt / confidence_tier for structures
that were already folded (PDB files already on disk under
results/structures/), using the corrected scale-aware
structure_prediction.mean_plddt_from_pdb. Does NOT call the ESM Atlas API
again -- just re-parses the PDB files you already have and rewrites
results/tables/structure_predictions.csv with corrected values.

Usage:
    python scripts/recompute_plddt.py --results results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import structure_prediction


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/", help="Path to the pipeline's --outdir")
    args = p.parse_args()

    results_dir = Path(args.results)
    csv_path = results_dir / "tables" / "structure_predictions.csv"
    if not csv_path.exists():
        print(f"No structure_predictions.csv found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    n_fixed = 0
    for i, row in df.iterrows():
        if not row.get("fold_success", False) or pd.isna(row.get("pdb_path")):
            continue
        pdb_path = Path(row["pdb_path"])
        if not pdb_path.exists():
            print(f"  WARNING: PDB file not found: {pdb_path}")
            continue
        pdb_text = pdb_path.read_text()
        new_plddt = structure_prediction.mean_plddt_from_pdb(pdb_text)
        df.at[i, "mean_plddt"] = new_plddt
        df.at[i, "confidence_tier"] = structure_prediction._confidence_tier(new_plddt)
        n_fixed += 1

    df.to_csv(csv_path, index=False)
    print(f"Recomputed pLDDT for {n_fixed} structures. Updated {csv_path}")
    print(df[["representative_id", "mean_plddt", "confidence_tier"]].to_string(index=False))


if __name__ == "__main__":
    main()
