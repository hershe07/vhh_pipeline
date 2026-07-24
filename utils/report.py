"""
report.py
----------
Section 17: aggregate every stage's metrics into one comprehensive summary
report (funnel counts, diversity metrics, cluster stats), and export all
tables/FASTA/figures already produced by other modules.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def build_summary_report(
    n_initial_reads: int,
    n_after_qc: int,
    n_after_length_filter: int,
    n_valid_vhh: int,
    n_unique_nt: int,
    n_unique_aa: int,
    n_unique_cdr3: int,
    n_unique_cdr_combinations: int,
    diversity_metrics: dict,
    cluster_stats: dict,
    extra: dict | None = None,
) -> dict:
    """Assemble the funnel + headline stats dict described in Section 17."""
    report = {
        "funnel": {
            "initial_reads": n_initial_reads,
            "after_qc": n_after_qc,
            "after_length_filter": n_after_length_filter,
            "valid_vhh_sequences": n_valid_vhh,
            "pct_retained_overall": round(100 * n_valid_vhh / n_initial_reads, 2) if n_initial_reads else 0.0,
        },
        "uniqueness": {
            "unique_nucleotide_sequences": n_unique_nt,
            "unique_amino_acid_sequences": n_unique_aa,
            "unique_cdr3_sequences": n_unique_cdr3,
            "unique_cdr_length_combinations": n_unique_cdr_combinations,
        },
        "diversity_metrics": diversity_metrics,
        "cluster_stats": cluster_stats,
    }
    if extra:
        report["extra"] = extra
    return report


def write_report(report: dict, outdir: str | Path, filename: str = "summary_report") -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{filename}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    md_path = outdir / f"{filename}.md"
    with open(md_path, "w") as f:
        f.write("# VHH Repertoire Analysis -- Summary Report\n\n")
        for section, content in report.items():
            f.write(f"## {section.replace('_', ' ').title()}\n\n")
            if isinstance(content, dict):
                for k, v in content.items():
                    f.write(f"- **{k}**: {v}\n")
            else:
                f.write(f"{content}\n")
            f.write("\n")
    logger.info("Wrote summary report to %s and %s", json_path, md_path)
