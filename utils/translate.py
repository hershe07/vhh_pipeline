"""
translate.py
------------
Translate nucleotide reads into amino-acid sequences and flag/remove
problematic ORFs: premature stop codons, frameshifts (relative to the
best-scoring frame), and sequences with no clean open reading frame at all.

Because there is no reference, "best frame" is chosen heuristically per-read:
try all 6 frames (3 forward + 3 reverse complement), translate each, and
score by (a) absence of internal stop codons and (b) similarity to expected
VHH length/composition. This is intentionally reference-free.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

STOP = "*"


@dataclass
class TranslationResult:
    protein: str
    frame: int          # 1,2,3 = forward frames; -1,-2,-3 = reverse-complement frames
    strand: str          # '+' or '-'
    n_internal_stops: int
    ends_clean: bool     # True if the only stop (if any) is the terminal one
    is_valid_orf: bool


def _translate_frame(seq: str, frame: int) -> str:
    """frame in {1,2,3}; operates on the given strand already (fwd or revcomp)."""
    s = Seq(seq[frame - 1:])
    # trim to multiple of 3 to avoid Biopython partial-codon warnings
    s = s[: len(s) - (len(s) % 3)]
    return str(s.translate())


def best_frame_translation(nt_seq: str, expected_aa_length: int | None = None) -> TranslationResult:
    """
    Try all 6 frames, pick the one with the fewest internal stop codons,
    breaking ties by closeness to expected_aa_length (if provided).
    """
    nt_seq = nt_seq.upper().replace("U", "T")
    rc = str(Seq(nt_seq).reverse_complement())

    candidates = []
    for frame in (1, 2, 3):
        aa = _translate_frame(nt_seq, frame)
        candidates.append((aa, frame, "+"))
    for frame in (1, 2, 3):
        aa = _translate_frame(rc, frame)
        candidates.append((aa, -frame, "-"))

    scored = []
    for aa, frame, strand in candidates:
        internal = aa[:-1].count(STOP) if aa.endswith(STOP) else aa.count(STOP)
        n_stops = aa.count(STOP)
        ends_clean = aa.endswith(STOP) and n_stops == 1 or n_stops == 0
        len_penalty = abs(len(aa) - expected_aa_length) if expected_aa_length else 0
        # primary key: fewer internal stops; secondary: closer to expected length
        scored.append(((internal, len_penalty), aa, frame, strand, internal, ends_clean))

    scored.sort(key=lambda t: t[0])
    (_, aa, frame, strand, internal, ends_clean) = scored[0]

    is_valid = (internal == 0) and (len(aa.rstrip(STOP)) >= 3)
    return TranslationResult(
        protein=aa.rstrip(STOP) if aa.endswith(STOP) else aa,
        frame=frame,
        strand=strand,
        n_internal_stops=internal,
        ends_clean=ends_clean,
        is_valid_orf=is_valid,
    )


def translate_dataframe(
    df: pd.DataFrame,
    seq_col: str = "sequence",
    expected_aa_length: int | None = 132,  # ~397bp / 3 minus primers/tags, adjust as needed
) -> pd.DataFrame:
    """
    Adds columns: protein, frame, strand, n_internal_stops, ends_clean, is_valid_orf
    to the input DataFrame. Does NOT drop rows -- filtering is a separate,
    explicit step (see split_valid_orfs) so every decision is auditable.
    """
    results = df[seq_col].apply(lambda s: best_frame_translation(s, expected_aa_length))
    out = df.copy()
    out["protein"] = [r.protein for r in results]
    out["frame"] = [r.frame for r in results]
    out["strand"] = [r.strand for r in results]
    out["n_internal_stops"] = [r.n_internal_stops for r in results]
    out["ends_clean"] = [r.ends_clean for r in results]
    out["is_valid_orf"] = [r.is_valid_orf for r in results]
    out["protein_length"] = out["protein"].str.len()
    return out


def split_valid_orfs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Split translated DataFrame into valid-ORF vs flagged (stop codons / frameshift artifacts)."""
    valid = df[df["is_valid_orf"]].copy()
    invalid = df[~df["is_valid_orf"]].copy()
    report = {
        "n_input": len(df),
        "n_valid_orf": len(valid),
        "n_invalid_orf": len(invalid),
        "pct_valid": round(100 * len(valid) / len(df), 2) if len(df) else 0.0,
        "invalid_reasons": {
            "internal_stop_codon": int((invalid["n_internal_stops"] > 0).sum()),
            "too_short": int((invalid["protein_length"] < 3).sum()),
        },
    }
    logger.info("ORF validation: %d/%d valid (%.1f%%)", len(valid), len(df), report["pct_valid"])
    return valid, invalid, report
