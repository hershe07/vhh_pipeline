"""
qc.py
-----
Quality-control filtering and reporting for raw Nanopore reads.

Design notes:
- Mean Phred quality is computed in probability space (see io_utils.phred_to_mean_qscore),
  which is the correct way to average Nanopore Q-scores.
- QC is deliberately kept separate from length filtering (utils/length_filter.py) and
  from ORF/translation validity (utils/translate.py) so each filter's yield can be
  reported independently -- this is important for diagnosing *where* a library is
  losing reads (e.g. mostly quality vs mostly wrong-length -> chimeras/adapters).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def summarize_reads(df: pd.DataFrame, label: str = "reads") -> dict:
    """Basic summary statistics for a read set (used before/after each filter step)."""
    summary = {
        "label": label,
        "n_reads": len(df),
        "length_mean": float(df["length"].mean()) if len(df) else np.nan,
        "length_median": float(df["length"].median()) if len(df) else np.nan,
        "length_std": float(df["length"].std()) if len(df) else np.nan,
        "length_min": int(df["length"].min()) if len(df) else np.nan,
        "length_max": int(df["length"].max()) if len(df) else np.nan,
    }
    if "mean_qscore" in df.columns and df["mean_qscore"].notna().any():
        summary.update(
            {
                "qscore_mean": float(df["mean_qscore"].mean()),
                "qscore_median": float(df["mean_qscore"].median()),
                "qscore_std": float(df["mean_qscore"].std()),
            }
        )
    logger.info("[%s] n=%d, mean_len=%.1f, mean_Q=%.2f",
                label, summary["n_reads"], summary.get("length_mean", float("nan")),
                summary.get("qscore_mean", float("nan")))
    return summary


def filter_by_quality(
    df: pd.DataFrame,
    min_mean_qscore: float = 9.0,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter reads by mean Phred quality (and optionally a coarse length sanity window,
    finer-grained amplicon-length filtering happens in length_filter.py).

    Records lacking a quality score (e.g. loaded from FASTA/consensus) are passed
    through unfiltered on the quality axis -- there is nothing to filter on.

    Returns (passed_df, failed_df).
    """
    has_q = df["mean_qscore"].notna() if "mean_qscore" in df.columns else pd.Series(False, index=df.index)
    quality_ok = (~has_q) | (df["mean_qscore"] >= min_mean_qscore)

    length_ok = pd.Series(True, index=df.index)
    if min_length is not None:
        length_ok &= df["length"] >= min_length
    if max_length is not None:
        length_ok &= df["length"] <= max_length

    keep_mask = quality_ok & length_ok
    passed, failed = df[keep_mask].copy(), df[~keep_mask].copy()
    logger.info("Quality filter: %d/%d reads passed (min_mean_qscore=%.1f)",
                len(passed), len(df), min_mean_qscore)
    return passed, failed


def plot_quality_distribution(df: pd.DataFrame, outpath: str | Path, title: str = "Read quality distribution"):
    """Histogram of per-read mean Phred quality."""
    fig, ax = plt.subplots(figsize=(6, 4))
    q = df["mean_qscore"].dropna()
    if len(q) == 0:
        ax.text(0.5, 0.5, "No quality scores available\n(FASTA/consensus input)",
                ha="center", va="center")
    else:
        ax.hist(q, bins=60, color="#3b6ea5", edgecolor="none")
        ax.axvline(q.median(), color="black", linestyle="--", linewidth=1,
                    label=f"median={q.median():.1f}")
        ax.legend()
    ax.set_xlabel("Mean Phred quality (Q)")
    ax.set_ylabel("Read count")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outpath)


def plot_length_distribution(df: pd.DataFrame, outpath: str | Path, title: str = "Read length distribution",
                              expected_length: Optional[int] = None, min_len: Optional[int] = None,
                              max_len: Optional[int] = None):
    """Histogram of read lengths, optionally annotated with the expected amplicon window."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df["length"], bins=80, color="#5a9e6f", edgecolor="none")
    if expected_length is not None:
        ax.axvline(expected_length, color="black", linestyle="-", linewidth=1, label=f"expected={expected_length}")
    if min_len is not None and max_len is not None:
        ax.axvspan(min_len, max_len, color="gray", alpha=0.15, label=f"kept window [{min_len},{max_len}]")
    ax.set_xlabel("Read length (bp)")
    ax.set_ylabel("Read count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, outpath)


def plot_before_after(before: pd.DataFrame, after: pd.DataFrame, column: str, outpath: str | Path,
                       xlabel: str, title: str):
    """Overlaid before/after histogram, e.g. for quality or length pre/post filtering."""
    fig, ax = plt.subplots(figsize=(6, 4))
    b = before[column].dropna()
    a = after[column].dropna()
    bins = np.linspace(min(b.min(), a.min()) if len(a) else b.min(),
                        max(b.max(), a.max()) if len(a) else b.max(), 60)
    ax.hist(b, bins=bins, alpha=0.5, label=f"before (n={len(b)})", color="#c0504d")
    ax.hist(a, bins=bins, alpha=0.5, label=f"after (n={len(a)})", color="#4f81bd")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Read count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, outpath)


def _save(fig, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    # also emit a PDF twin for publication use
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def qc_report(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Tabular before/after QC summary (one row each), suitable for the final report."""
    rows = [summarize_reads(before, "before_QC"), summarize_reads(after, "after_QC")]
    return pd.DataFrame(rows)
