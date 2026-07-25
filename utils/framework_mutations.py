"""
framework_mutations.py
-------------------------
Reference-free "framework mutation" analysis: since this pipeline has no
external reference sequence, framework deviations are identified by
comparing every sequence against a CONSENSUS built from the repertoire
itself (the same internal-comparison philosophy as similarity.py and
diversity.py), rather than against an external genome/reference.

Rationale: FR1-FR4 are, by construction, the most conserved parts of a VHH
domain -- that's what makes them "framework" rather than "CDR". A position
that's ~99% identical across thousands of independent reads but differs in
one sequence is either a genuine rare framework variant or, far more often
given Nanopore's per-read error rate, a sequencing error. This module
doesn't try to adjudicate that automatically -- it surfaces the deviation,
its rarity across the repertoire, and the supporting read's quality score,
so a reviewer can judge it directly (a rare deviation on a low-quality
read is a strong error signal; the same deviation recurring across many
independent reads is a strong real-variant signal).

Position-wise comparison requires equal-length sequences for a given
region (same reasoning as diversity.position_specific_variability and
cdr_length_analysis.py's length-grouping): each FR region is compared
against the consensus built from sequences sharing that region's single
most common length. Sequences whose region length differs from that mode
are flagged separately as "length_variant" rather than compared
position-by-position, since insertions/deletions aren't meaningfully
alignable without real structural alignment.
"""
from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

logger = logging.getLogger(__name__)

FRAMEWORK_REGIONS = ["FR1", "FR2", "FR3", "FR4"]


def _modal_length(sequences: pd.Series) -> int:
    lengths = sequences.dropna().str.len()
    return int(lengths.mode().iloc[0]) if len(lengths) else 0


def build_region_consensus(sequences: list[str]) -> dict:
    """
    Per-position consensus residue + entropy for a set of EQUAL-LENGTH
    sequences (caller must pre-filter to one length -- see
    build_framework_mutation_table). Returns {position (1-indexed):
    {'consensus': aa, 'consensus_freq': frac, 'entropy': float}}.
    """
    import numpy as np
    if not sequences:
        return {}
    length = len(sequences[0])
    result = {}
    for pos in range(length):
        col = pd.Series([s[pos] for s in sequences]).value_counts()
        p = col / col.sum()
        entropy = float(-np.sum(p * np.log(p)))
        result[pos + 1] = {
            "consensus": col.idxmax(),
            "consensus_freq": float(col.max() / col.sum()),
            "entropy": entropy,
        }
    return result


def build_framework_mutation_table(
    cdr_table: pd.DataFrame,
    quality_lookup: pd.DataFrame | None = None,
    id_col: str = "read_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    cdr_table: output of cdr_annotation (must have read_id + FR1..FR4 columns).
    quality_lookup: optional DataFrame with [id_col, 'mean_qscore'] to attach
        read quality to each flagged deviation.

    Returns:
      mutations_df   -- one row per (sequence, region, position) deviation
                         from that region's consensus, with repertoire-wide
                         deviation_frequency and (if available) mean_qscore.
      position_variability_df -- per-position entropy across all four
                         regions, concatenated with a 'region' column, for
                         a repertoire-wide "which positions vary" plot.
      sequence_summary_df -- one row per sequence: how many framework
                         deviations it carries, and whether any region had
                         a non-modal length (length_variant).
    """
    mutation_rows = []
    variability_rows = []
    length_variant_ids = {region: set() for region in FRAMEWORK_REGIONS}
    per_seq_mutation_count = Counter()

    for region in FRAMEWORK_REGIONS:
        if region not in cdr_table.columns:
            continue
        region_series = cdr_table[[id_col, region]].dropna(subset=[region])
        if len(region_series) == 0:
            continue
        modal_len = _modal_length(region_series[region])
        at_modal = region_series[region_series[region].str.len() == modal_len]
        off_modal = region_series[region_series[region].str.len() != modal_len]
        length_variant_ids[region] = set(off_modal[id_col])

        sequences = at_modal[region].tolist()
        consensus = build_region_consensus(sequences)

        for pos, info in consensus.items():
            variability_rows.append({
                "region": region, "position": pos,
                "consensus_residue": info["consensus"],
                "consensus_freq": info["consensus_freq"],
                "entropy": info["entropy"],
            })

        # First pass: count how many sequences carry each (position, observed) deviation,
        # so we can report a repertoire-wide deviation_frequency in the same pass as flagging.
        deviation_counts = Counter()
        seq_deviations = []  # (read_id, pos, consensus, observed)
        for _, row in at_modal.iterrows():
            seq = row[region]
            for pos, info in consensus.items():
                observed = seq[pos - 1]
                if observed != info["consensus"]:
                    seq_deviations.append((row[id_col], pos, info["consensus"], observed))
                    deviation_counts[(pos, observed)] += 1

        for read_id, pos, consensus_aa, observed_aa in seq_deviations:
            mutation_rows.append({
                "read_id": read_id, "region": region, "position": pos,
                "consensus_residue": consensus_aa, "observed_residue": observed_aa,
                "deviation_frequency": deviation_counts[(pos, observed_aa)],
            })
            per_seq_mutation_count[read_id] += 1

    mutations_df = pd.DataFrame(mutation_rows)
    if len(mutations_df) and quality_lookup is not None:
        mutations_df = mutations_df.merge(quality_lookup, left_on="read_id", right_on=id_col, how="left")

    position_variability_df = pd.DataFrame(variability_rows)

    all_ids = cdr_table[id_col].unique()
    summary_rows = []
    for rid in all_ids:
        n_mut = per_seq_mutation_count.get(rid, 0)
        is_length_variant = any(rid in length_variant_ids[r] for r in FRAMEWORK_REGIONS)
        summary_rows.append({
            "read_id": rid, "n_framework_deviations": n_mut,
            "has_length_variant_region": is_length_variant,
        })
    sequence_summary_df = pd.DataFrame(summary_rows)

    logger.info(
        "Framework mutation analysis: %d deviations across %d sequences (%d with >=1 deviation)",
        len(mutations_df), len(all_ids), int((sequence_summary_df["n_framework_deviations"] > 0).sum()),
    )
    return mutations_df, position_variability_df, sequence_summary_df
