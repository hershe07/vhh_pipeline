"""
diversity.py
-------------
Section 12: nucleotide/protein/CDR diversity metrics.

All metrics are computed directly from the observed repertoire (abundance
counts of unique sequences/CDRs) -- no reference required, consistent with
standard AIRR-seq / ecological diversity analysis (these are the same
Hill-number-family metrics used in classic species-diversity ecology,
applied here to clonotypes/CDR variants instead of species).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def shannon_entropy(counts: pd.Series | np.ndarray, base: float = np.e) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p) / np.log(base)))


def simpson_diversity(counts: pd.Series | np.ndarray) -> float:
    """Simpson's Diversity Index D = 1 - sum(p_i^2) (probability two random draws differ)."""
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()
    return float(1.0 - np.sum(p ** 2))


def richness(counts: pd.Series | np.ndarray) -> int:
    counts = np.asarray(counts)
    return int(np.sum(counts > 0))


def pielou_evenness(counts: pd.Series | np.ndarray) -> float:
    """Pielou's evenness J = H / ln(S); 0 (dominated by one clone) to 1 (perfectly even)."""
    s = richness(counts)
    if s <= 1:
        return np.nan
    h = shannon_entropy(counts, base=np.e)
    return float(h / np.log(s))


def diversity_summary(counts: pd.Series | np.ndarray, label: str = "") -> dict:
    return {
        "label": label,
        "richness": richness(counts),
        "shannon_entropy": shannon_entropy(counts),
        "simpson_diversity": simpson_diversity(counts),
        "pielou_evenness": pielou_evenness(counts),
        "n_observations": int(np.sum(counts)),
    }


def cdr_diversity_table(cdr_table: pd.DataFrame) -> pd.DataFrame:
    """Section 12: Shannon/Simpson/evenness/richness for CDR1, CDR2, CDR3 (by unique CDR sequence)."""
    rows = []
    for cdr in ("CDR1", "CDR2", "CDR3"):
        if cdr not in cdr_table.columns:
            continue
        counts = cdr_table[cdr].dropna().value_counts()
        rows.append(diversity_summary(counts.values, label=cdr))
    return pd.DataFrame(rows)


def position_specific_variability(sequences: list[str]) -> pd.DataFrame:
    """
    Per-position amino-acid variability for a set of EQUAL-LENGTH sequences
    (e.g. all CDR3s of a given length, or an aligned region). Returns, per
    position: Shannon entropy and the consensus/majority residue.
    Sequences of unequal length are not comparable position-wise and should
    be grouped by length first (see cdr_length_analysis.py).
    """
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(
            f"position_specific_variability requires equal-length sequences; got lengths {lengths}. "
            "Group by CDR length first (see cdr_length_analysis.cdr_length_combinations)."
        )
    length = lengths.pop()
    rows = []
    for pos in range(length):
        col = [s[pos] for s in sequences]
        counts = pd.Series(col).value_counts()
        rows.append({
            "position": pos + 1,
            "entropy": shannon_entropy(counts.values),
            "consensus_residue": counts.idxmax(),
            "consensus_freq": counts.max() / counts.sum(),
        })
    return pd.DataFrame(rows)


def rarefaction_curve(counts: pd.Series | np.ndarray, n_steps: int = 20, n_resample: int = 20,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Rarefaction curve: expected number of unique sequences/CDRs observed as a
    function of subsampled read depth, via random subsampling without
    replacement (repeated n_resample times per depth for a mean +/- std).
    Standard way to assess whether sequencing depth has saturated diversity.
    """
    rng = np.random.default_rng(random_state)
    counts = np.asarray(counts, dtype=int)
    # Expand to an array of "individual" clonotype labels (one entry per read)
    labels = np.repeat(np.arange(len(counts)), counts)
    total_n = len(labels)
    depths = np.unique(np.linspace(1, total_n, n_steps, dtype=int))

    rows = []
    for depth in depths:
        observed = []
        for _ in range(n_resample):
            sample = rng.choice(labels, size=depth, replace=False)
            observed.append(len(np.unique(sample)))
        rows.append({"depth": depth, "mean_unique": np.mean(observed), "std_unique": np.std(observed)})
    return pd.DataFrame(rows)
