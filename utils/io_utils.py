"""
io_utils.py
-----------
Loading and writing sequence data. Centralizes all file I/O so the rest of the
pipeline works on a single internal representation: a pandas DataFrame with
(at minimum) columns ['read_id', 'sequence', 'length'] and, for FASTQ,
['mean_qscore', 'qualities'].

Supports: FASTQ (raw Nanopore reads), FASTA (raw or consensus sequences).
"""
from __future__ import annotations

import gzip
import logging
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

logger = logging.getLogger(__name__)


def _smart_open(path: str | Path):
    """Open plain or gzip-compressed files transparently."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def phred_to_mean_qscore(qualities: list[int]) -> float:
    """
    Mean Phred quality, computed the standard Nanopore way: average the
    per-base ERROR PROBABILITIES, then convert back to Phred scale (not a
    naive arithmetic mean of Phred scores, which overestimates quality).
    """
    if not qualities:
        return 0.0
    probs = np.power(10, -np.asarray(qualities) / 10.0)
    mean_prob = probs.mean()
    if mean_prob <= 0:
        return 99.0
    return float(-10 * np.log10(mean_prob))


def load_fastq(path: str | Path) -> pd.DataFrame:
    """Load a FASTQ file into a DataFrame with per-read quality info."""
    records = []
    with _smart_open(path) as handle:
        for rec in SeqIO.parse(handle, "fastq"):
            quals = rec.letter_annotations.get("phred_quality", [])
            records.append(
                {
                    "read_id": rec.id,
                    "sequence": str(rec.seq).upper(),
                    "length": len(rec.seq),
                    "mean_qscore": phred_to_mean_qscore(quals),
                    "min_qscore": float(np.min(quals)) if quals else np.nan,
                }
            )
    df = pd.DataFrame.from_records(records)
    logger.info("Loaded %d reads from FASTQ %s", len(df), path)
    return df


def load_fasta(path: str | Path, is_consensus: bool = False) -> pd.DataFrame:
    """
    Load a FASTA (raw or consensus) file. No quality scores are available,
    so mean_qscore is set to NaN and QC filtering on quality is skipped
    downstream for these records (length/translation filters still apply).
    """
    records = []
    with _smart_open(path) as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            records.append(
                {
                    "read_id": rec.id,
                    "sequence": str(rec.seq).upper(),
                    "length": len(rec.seq),
                    "mean_qscore": np.nan,
                    "min_qscore": np.nan,
                    "is_consensus": is_consensus,
                }
            )
    df = pd.DataFrame.from_records(records)
    logger.info("Loaded %d sequences from FASTA %s (consensus=%s)", len(df), path, is_consensus)
    return df


def load_input(path: str | Path, is_consensus: bool = False) -> pd.DataFrame:
    """Dispatch on file extension: .fastq/.fq(.gz) -> FASTQ, .fasta/.fa(.gz) -> FASTA."""
    path = Path(path)
    suffixes = "".join(path.suffixes).lower()
    if any(s in suffixes for s in [".fastq", ".fq"]):
        return load_fastq(path)
    elif any(s in suffixes for s in [".fasta", ".fa", ".fna"]):
        return load_fasta(path, is_consensus=is_consensus)
    else:
        raise ValueError(f"Unrecognized input file type for {path}. Expected FASTQ or FASTA.")


def write_fasta(df: pd.DataFrame, path: str | Path, id_col: str = "read_id", seq_col: str = "sequence",
                 description_cols: Optional[list[str]] = None) -> None:
    """Write a DataFrame of sequences out to FASTA, optionally with metadata in the header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records: list[SeqRecord] = []
    for _, row in df.iterrows():
        desc = ""
        if description_cols:
            desc = " ".join(f"{c}={row[c]}" for c in description_cols if c in row)
        records.append(SeqRecord(seq=_to_bio_seq(row[seq_col]), id=str(row[id_col]), description=desc))
    SeqIO.write(records, path, "fasta")
    logger.info("Wrote %d sequences to %s", len(records), path)


def _to_bio_seq(seq_str: str):
    from Bio.Seq import Seq
    return Seq(seq_str)


def write_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Write CSV or Excel depending on extension."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(path, index=index)
    else:
        df.to_csv(path, index=index)
    logger.info("Wrote table (%d rows) to %s", len(df), path)
