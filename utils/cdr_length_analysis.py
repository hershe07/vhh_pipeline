"""
cdr_length_analysis.py
------------------------
Sections 7 & 8 of the brief: per-CDR length distributions, and the joint
CDR1-CDR2-CDR3 length "combination" analysis (e.g. "9-8-18"), which is a
standard, cheap way to characterize germline/junctional diversity in a
repertoire without needing any reference.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_cdr_table(df: pd.DataFrame, id_col: str = "read_id") -> pd.DataFrame:
    """Extract the per-sequence CDR/FR table requested in Section 7."""
    cols = [id_col, "CDR1", "CDR2", "CDR3", "CDR1_length", "CDR2_length", "CDR3_length",
            "FR1", "FR2", "FR3", "FR4", "annotation_method"]
    present = [c for c in cols if c in df.columns]
    table = df[present].copy()
    return table


def cdr_length_distributions(cdr_table: pd.DataFrame) -> dict[str, pd.Series]:
    """Value counts of each CDR's length, for plotting/reporting."""
    dists = {}
    for cdr in ("CDR1", "CDR2", "CDR3"):
        col = f"{cdr}_length"
        if col in cdr_table.columns:
            dists[cdr] = cdr_table[col].dropna().astype(int).value_counts().sort_index()
    return dists


def plot_cdr_length_distributions(dists: dict[str, pd.Series], outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for cdr, dist in dists.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(dist.index, dist.values, color="#4f81bd")
        ax.set_xlabel(f"{cdr} length (aa)")
        ax.set_ylabel("Number of sequences")
        ax.set_title(f"{cdr} length distribution")
        fig.tight_layout()
        fig.savefig(outdir / f"{cdr.lower()}_length_dist.png", dpi=300)
        fig.savefig(outdir / f"{cdr.lower()}_length_dist.pdf")
        plt.close(fig)


def cdr_length_combinations(cdr_table: pd.DataFrame) -> pd.DataFrame:
    """
    Section 8: identify all observed (CDR1_len, CDR2_len, CDR3_len) combinations,
    e.g. "8-7-13", and count their frequency. Returns a DataFrame sorted by
    descending frequency with a formatted 'combination' string column.
    """
    needed = ["CDR1_length", "CDR2_length", "CDR3_length"]
    valid = cdr_table.dropna(subset=needed).copy()
    valid[needed] = valid[needed].astype(int)
    combo = (
        valid.groupby(needed)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    combo["combination"] = combo.apply(
        lambda r: f"{r.CDR1_length}-{r.CDR2_length}-{r.CDR3_length}", axis=1
    )
    combo["relative_abundance"] = combo["count"] / combo["count"].sum()
    total_seqs_used = len(valid)
    combo.attrs["n_sequences_used"] = total_seqs_used
    logger.info("Found %d unique CDR length combinations across %d sequences",
                len(combo), total_seqs_used)
    return combo


def flag_rare_combinations(combo_df: pd.DataFrame, rare_threshold_count: int = 2) -> pd.DataFrame:
    """Mark combinations seen <= rare_threshold_count times as 'rare' (candidate novel junctions)."""
    out = combo_df.copy()
    out["is_rare"] = out["count"] <= rare_threshold_count
    return out


def plot_combination_bar(combo_df: pd.DataFrame, outpath: str | Path, top_n: int = 30):
    top = combo_df.head(top_n)
    fig, ax = plt.subplots(figsize=(max(6, top_n * 0.3), 5))
    ax.bar(top["combination"], top["count"], color="#8064a2")
    ax.set_xticklabels(top["combination"], rotation=90)
    ax.set_ylabel("Sequence count")
    ax.set_title(f"Top {top_n} CDR1-CDR2-CDR3 length combinations")
    fig.tight_layout()
    _save(fig, outpath)


def plot_combination_heatmap(cdr_table: pd.DataFrame, outpath: str | Path,
                              fix_cdr1: int | None = None):
    """
    Heatmap of CDR2_length (rows) x CDR3_length (cols) counts. If fix_cdr1 is
    given, restricts to sequences with that CDR1 length (since a full 3D
    combination doesn't render as one 2D heatmap); otherwise marginalizes
    over CDR1.
    """
    needed = ["CDR1_length", "CDR2_length", "CDR3_length"]
    valid = cdr_table.dropna(subset=needed).copy()
    valid[needed] = valid[needed].astype(int)
    if fix_cdr1 is not None:
        valid = valid[valid["CDR1_length"] == fix_cdr1]
        title = f"CDR2 x CDR3 length counts (CDR1={fix_cdr1})"
    else:
        title = "CDR2 x CDR3 length counts (all CDR1 lengths)"

    pivot = valid.pivot_table(index="CDR2_length", columns="CDR3_length",
                               aggfunc="size", fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("CDR3 length (aa)")
    ax.set_ylabel("CDR2 length (aa)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="sequence count")
    fig.tight_layout()
    _save(fig, outpath)


def _save(fig, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)
