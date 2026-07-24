"""
plotting.py
------------
Shared matplotlib/plotly plotting helpers for diversity, entropy, and
rarefaction visualizations (Section 12), kept separate from the metric
computations in diversity.py so figures can be restyled without touching
analysis logic.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.with_suffix(".pdf"))
    plt.close(fig)


def plot_entropy_by_position(pos_variability_df: pd.DataFrame, outpath: str | Path, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(pos_variability_df["position"], pos_variability_df["entropy"], color="#c0504d")
    ax.set_xlabel("Position")
    ax.set_ylabel("Shannon entropy (nats)")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outpath)


def plot_diversity_bar(diversity_df: pd.DataFrame, metric: str, outpath: str | Path, title: str):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(diversity_df["label"], diversity_df[metric], color="#4f81bd")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outpath)


def plot_rarefaction(rarefaction_df: pd.DataFrame, outpath: str | Path, title: str = "Rarefaction curve"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rarefaction_df["depth"], rarefaction_df["mean_unique"], color="#3b6ea5")
    ax.fill_between(
        rarefaction_df["depth"],
        rarefaction_df["mean_unique"] - rarefaction_df["std_unique"],
        rarefaction_df["mean_unique"] + rarefaction_df["std_unique"],
        alpha=0.2, color="#3b6ea5",
    )
    ax.set_xlabel("Reads sampled")
    ax.set_ylabel("Unique sequences observed")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outpath)


def plot_abundance_rank_curve(rank_table: pd.DataFrame, outpath: str | Path,
                               title: str = "Clonal abundance rank curve"):
    """Log-log rank-abundance curve -- a quick visual for clonal expansion / skew."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rank_table["rank"], rank_table["relative_abundance"], marker=".", linestyle="none",
            color="#8064a2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Relative abundance")
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, outpath)


def plot_sequence_logo(sequences: list[str], outpath: str | Path, title: str = "Sequence logo"):
    """
    Requires `logomaker` and equal-length sequences (group by CDR length first,
    see cdr_length_analysis.py). Falls back to a plain stacked-frequency bar
    plot if logomaker isn't installed.
    """
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(f"plot_sequence_logo requires equal-length sequences; got lengths {lengths}")

    try:
        import logomaker
        counts_df = logomaker.alignment_to_matrix(sequences)
        fig, ax = plt.subplots(figsize=(max(6, len(sequences[0]) * 0.4), 3))
        logomaker.Logo(counts_df, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        _save(fig, outpath)
    except ImportError:
        # Fallback: simple per-position stacked bar of top residue frequency
        import numpy as np
        length = lengths.pop()
        freqs = []
        for pos in range(length):
            col = pd.Series([s[pos] for s in sequences]).value_counts(normalize=True)
            freqs.append(col)
        fig, ax = plt.subplots(figsize=(max(6, length * 0.4), 3))
        bottom = np.zeros(length)
        for aa in sorted(set().union(*[f.index for f in freqs])):
            heights = [f.get(aa, 0) for f in freqs]
            ax.bar(range(1, length + 1), heights, bottom=bottom, label=aa)
            bottom += heights
        ax.set_title(title + " (fallback bar plot -- install `logomaker` for a true logo)")
        ax.set_xlabel("Position")
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        _save(fig, outpath)
