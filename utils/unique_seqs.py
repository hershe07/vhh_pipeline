"""
unique_seqs.py
---------------
Section 9: collapse duplicate sequences (nucleotide and amino-acid level),
compute copy number / relative abundance, and produce ranked abundance
tables. This is the repertoire's own "clonal abundance" view and is used
downstream both for diversity metrics and for novel/rare-variant flagging.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def collapse_unique(df: pd.DataFrame, seq_col: str, id_col: str = "read_id") -> pd.DataFrame:
    """
    Collapse to unique sequences in `seq_col`, keeping:
    - representative_id: the first read_id observed for that sequence
    - copy_number: how many reads collapse to this sequence
    - relative_abundance: copy_number / total reads
    - member_ids: list of all read_ids collapsing here (for traceability)
    """
    grouped = df.groupby(seq_col)[id_col].agg(list).reset_index()
    grouped["copy_number"] = grouped[id_col].apply(len)
    grouped["representative_id"] = grouped[id_col].apply(lambda ids: ids[0])
    grouped = grouped.rename(columns={id_col: "member_ids"})
    total = grouped["copy_number"].sum()
    grouped["relative_abundance"] = grouped["copy_number"] / total
    grouped = grouped.sort_values("copy_number", ascending=False).reset_index(drop=True)
    grouped["rank"] = grouped.index + 1
    grouped["cumulative_abundance"] = grouped["relative_abundance"].cumsum()
    logger.info("Collapsed %d reads into %d unique sequences (column=%s)", len(df), len(grouped), seq_col)
    return grouped


def unique_nt_and_aa(df: pd.DataFrame, nt_col: str = "sequence", aa_col: str = "protein",
                      id_col: str = "read_id") -> dict[str, pd.DataFrame]:
    """Convenience wrapper producing both nucleotide-level and amino-acid-level tables."""
    return {
        "unique_nt": collapse_unique(df, nt_col, id_col),
        "unique_aa": collapse_unique(df, aa_col, id_col),
    }


def diversity_from_abundance(rank_table: pd.DataFrame, copy_col: str = "copy_number") -> dict:
    """
    Quick top-line numbers derived from an abundance-ranked table -- richness
    and the fraction of reads that are singletons (copy_number==1), which is
    a useful, cheap proxy for how much of the apparent diversity might be
    Nanopore error-inflated (see novel_discovery.py / critical review).
    """
    n_unique = len(rank_table)
    n_reads = rank_table[copy_col].sum()
    n_singletons = int((rank_table[copy_col] == 1).sum())
    return {
        "n_unique_sequences": n_unique,
        "n_total_reads": int(n_reads),
        "n_singletons": n_singletons,
        "pct_singletons_of_unique": round(100 * n_singletons / n_unique, 2) if n_unique else 0.0,
        "pct_reads_in_top10": round(
            100 * rank_table.sort_values(copy_col, ascending=False)[copy_col].head(10).sum() / n_reads, 2
        ) if n_reads else 0.0,
    }
