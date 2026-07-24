"""
novel_discovery.py
--------------------
Section 15: flag rare nanobodies, unique CDR combinations, and outlier
sequences as candidate novel variants, while trying to distinguish true
biological diversity from Nanopore sequencing artefacts.

Key idea: no single signal (low abundance, cluster-outlier status, rare CDR
length combo) is trustworthy alone, because Nanopore's ~single-digit-percent
raw error rate means genuine low-abundance clones and error-derived
"pseudo-clones" both show up as singletons/rare combinations. This module
combines multiple orthogonal signals and reports a composite score plus the
individual flags, so a human reviewer can triage rather than blindly trust
an automated "novel" call. See Critical Review for a fuller discussion.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def flag_candidates(
    full_table: pd.DataFrame,
    copy_number_col: str = "copy_number",
    cluster_col: str | None = "cluster",
    rare_combo_col: str | None = "is_rare",
    mean_qscore_col: str | None = "mean_qscore",
    singleton_threshold: int = 1,
    low_quality_qscore: float = 12.0,
) -> pd.DataFrame:
    """
    Adds boolean flags and a composite `novelty_score` (0-1) to a per-unique-
    sequence table. Higher novelty_score = more evidence this is a genuine
    rare/novel variant rather than a sequencing artefact; LOW mean_qscore on
    a low-copy-number sequence pulls the score down (more likely artefact),
    while corroborating evidence from independent reads (copy_number > 1)
    or a nearest-neighbor cluster of similar rare variants pulls it up.
    """
    out = full_table.copy()

    out["is_singleton"] = out[copy_number_col] <= singleton_threshold

    if cluster_col and cluster_col in out.columns:
        out["is_cluster_outlier"] = out[cluster_col] == -1
    else:
        out["is_cluster_outlier"] = np.nan

    if rare_combo_col and rare_combo_col in out.columns:
        out["has_rare_cdr_combo"] = out[rare_combo_col]
    else:
        out["has_rare_cdr_combo"] = np.nan

    if mean_qscore_col and mean_qscore_col in out.columns:
        out["is_low_quality_support"] = out[mean_qscore_col] < low_quality_qscore
    else:
        out["is_low_quality_support"] = False

    # Composite score: start from evidence-of-rarity, subtract quality concern
    score = np.zeros(len(out))
    score += out["is_singleton"].fillna(False).astype(int) * 0.3
    score += out["is_cluster_outlier"].fillna(False).astype(int) * 0.3
    score += out["has_rare_cdr_combo"].fillna(False).astype(int) * 0.2
    score += (out[copy_number_col] > 1).astype(int) * 0.2  # independent-read corroboration bonus
    score -= out["is_low_quality_support"].fillna(False).astype(int) * 0.3
    out["novelty_score"] = score.clip(0, 1)

    out["candidate_novel"] = (
        (out["novelty_score"] >= 0.5)
        & (~out["is_low_quality_support"].fillna(False))
    )

    logger.info("Flagged %d/%d sequences as candidate_novel", out["candidate_novel"].sum(), len(out))
    return out


def summarize_novel_candidates(flagged_table: pd.DataFrame) -> dict:
    return {
        "n_total": len(flagged_table),
        "n_singletons": int(flagged_table["is_singleton"].sum()),
        "n_cluster_outliers": int(flagged_table["is_cluster_outlier"].fillna(False).sum()),
        "n_rare_cdr_combo": int(flagged_table["has_rare_cdr_combo"].fillna(False).sum()),
        "n_candidate_novel": int(flagged_table["candidate_novel"].sum()),
        "n_flagged_low_quality_support": int(flagged_table["is_low_quality_support"].sum()),
    }
