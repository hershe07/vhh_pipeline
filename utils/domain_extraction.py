"""
domain_extraction.py
-----------------------
Extracts the VHH domain nucleotide sub-sequence from each raw read, instead
of judging a read's validity by its *total* raw length. This replaces a
blunt whole-read length filter as the first real filtering step: a read
with extra flanking sequence (vector backbone, expression tag, adapter
remnant, or a second concatenated domain) is no longer discarded outright
-- the domain(s) it actually contains are pulled out and treated as the
working sequence(s) going forward.

Approach (reference-free, same conserved-anchor logic used throughout the
pipeline -- see vhh_identify.py / cdr_annotation.py):
  1. Translate the raw read in all 6 frames (3 forward, 3 reverse-complement).
  2. In each frame, find every FR4 "domain end" motif (WGxGxQVTVSS) via
     non-overlapping regex matches -- this is the single most reliable VHH
     anchor and, critically, essentially never occurs by chance.
  3. For each FR4 hit, look backward (within that frame, after any prior
     domain's end) for an FR1 "domain start" anchor (SLRLSCAAS). If found,
     the domain spans from there to the FR4 hit. If not found (e.g. the
     read/adapter trimming cut into FR1), fall back to a fixed ~132 aa
     window ending at the FR4 hit -- close to the expected ~397 bp / 3
     translated VHH domain length.
  4. The frame with the most domain hits is used (ties broken toward the
     domain-length closest to expected) -- this handles single-domain
     reads, reads with extra flanking sequence, AND concatemers/chimeras
     uniformly: each detected domain becomes its own extracted sub-read.
  5. Amino-acid coordinates are mapped back to nucleotide coordinates in
     the ORIGINAL (input-orientation) read, so extracted sub-sequences are
     always nucleotide, ready to re-enter the normal length filter ->
     translate -> VHH-screen -> CDR-annotate pipeline unchanged.

Reads with zero FR4 hits in every frame contain no detectable VHH domain
at all (adapter dimers, other off-target product, or reads too
error-ridden to translate cleanly) and are correctly dropped rather than
guessed at.

Note on quality scores: per-base quality strings aren't retained in the
pipeline's working DataFrame (only summary mean/min Phred), so an
extracted sub-domain's mean_qscore is inherited from its parent read as an
approximation, not recomputed over just that sub-region.
"""
from __future__ import annotations

import logging
import re

import pandas as pd
from Bio.Seq import Seq

logger = logging.getLogger(__name__)

FR1_END_RE = re.compile(r"SLRLSCAAS")
FR4_RE = re.compile(r"WG.G(?:TQ|LQ|MQ)VTVSS")
EXPECTED_DOMAIN_AA_LENGTH = 132  # ~397 bp / 3


def _frame_translation_with_offset(nt_seq: str, frame: int) -> tuple[str, int, bool]:
    """Returns (protein, nt_offset_of_aa0_in_strand, is_reverse) for a signed frame (+-1,2,3)."""
    is_reverse = frame < 0
    strand_seq = str(Seq(nt_seq).reverse_complement()) if is_reverse else nt_seq
    f = abs(frame)
    trimmed = strand_seq[f - 1:]
    trimmed = trimmed[: len(trimmed) - (len(trimmed) % 3)]
    protein = str(Seq(trimmed).translate())
    return protein, f - 1, is_reverse


def _domain_regions_in_protein(protein: str) -> list[tuple[int, int]]:
    """Returns [(start_aa, end_aa), ...] for every detected domain in this frame's protein."""
    regions = []
    prev_end = 0
    for m in FR4_RE.finditer(protein):
        fr4_end = m.end()
        search_zone = protein[prev_end:m.start()]
        fr1_match = FR1_END_RE.search(search_zone)
        if fr1_match:
            start = prev_end + fr1_match.start()
        else:
            start = max(prev_end, fr4_end - EXPECTED_DOMAIN_AA_LENGTH)
        if fr4_end > start:
            regions.append((start, fr4_end))
        prev_end = fr4_end
    return regions


def _aa_region_to_nt_span(start_aa: int, end_aa: int, aa0_nt_offset: int, is_reverse: bool,
                           raw_len: int) -> tuple[int, int]:
    """Maps an (start_aa, end_aa) region back to (nt_start, nt_end) in the ORIGINAL read orientation."""
    strand_nt_start = aa0_nt_offset + start_aa * 3
    strand_nt_end = aa0_nt_offset + end_aa * 3
    if not is_reverse:
        return strand_nt_start, strand_nt_end
    # strand coordinates are on the reverse-complemented sequence; flip back to original orientation
    orig_start = raw_len - strand_nt_end
    orig_end = raw_len - strand_nt_start
    return max(0, orig_start), min(raw_len, orig_end)


def extract_domains(nt_seq: str, read_id: str = "") -> list[dict]:
    """
    Runs all 6 frames, picks the frame with the most detected domains (ties
    broken by closeness of total domain length to expected), and returns a
    list of extracted nucleotide sub-sequences (nucleotide coordinates in
    the ORIGINAL read's orientation, ready to slice directly).
    """
    nt_seq = nt_seq.upper().replace("U", "T")
    raw_len = len(nt_seq)

    best = None  # (n_domains, -length_penalty, frame, regions, aa0_offset, is_reverse)
    for frame in (1, 2, 3, -1, -2, -3):
        protein, aa0_offset, is_reverse = _frame_translation_with_offset(nt_seq, frame)
        regions = _domain_regions_in_protein(protein)
        if not regions:
            continue
        length_penalty = sum(abs((e - s) - EXPECTED_DOMAIN_AA_LENGTH) for s, e in regions)
        key = (len(regions), -length_penalty)
        if best is None or key > best[0]:
            best = (key, frame, regions, aa0_offset, is_reverse)

    if best is None:
        return []

    _, frame, regions, aa0_offset, is_reverse = best
    extracted = []
    for i, (start_aa, end_aa) in enumerate(regions):
        nt_start, nt_end = _aa_region_to_nt_span(start_aa, end_aa, aa0_offset, is_reverse, raw_len)
        sub_seq = nt_seq[nt_start:nt_end]
        if len(sub_seq) < 50:
            continue
        extracted.append({
            "read_id": f"{read_id}_domain{i + 1}" if len(regions) > 1 else read_id,
            "sequence": sub_seq,
            "length": len(sub_seq),
            "extracted_from": read_id,
            "domain_index": i + 1,
            "n_domains_in_read": len(regions),
            "extraction_frame": frame,
        })
    return extracted


def extract_domains_dataframe(df: pd.DataFrame, seq_col: str = "sequence", id_col: str = "read_id") -> tuple[pd.DataFrame, dict]:
    """
    Applies extract_domains to every row. Returns (extracted_df, diagnostics).
    extracted_df has one row per detected domain (0, 1, or many per input
    read) with mean_qscore/min_qscore carried over from the parent read
    when those columns are present on the input.
    """
    has_qscore = "mean_qscore" in df.columns
    has_min_q = "min_qscore" in df.columns

    all_rows = []
    n_no_domain = 0
    n_single = 0
    n_multi = 0
    for _, row in df.iterrows():
        domains = extract_domains(row[seq_col], row[id_col])
        if not domains:
            n_no_domain += 1
            continue
        elif len(domains) == 1:
            n_single += 1
        else:
            n_multi += 1
        for d in domains:
            if has_qscore:
                d["mean_qscore"] = row["mean_qscore"]
            if has_min_q:
                d["min_qscore"] = row["min_qscore"]
            all_rows.append(d)

    extracted_df = pd.DataFrame(all_rows)
    diagnostics = {
        "n_input_reads": len(df),
        "n_reads_no_domain_detected": n_no_domain,
        "n_reads_single_domain": n_single,
        "n_reads_multi_domain": n_multi,
        "n_domains_extracted_total": len(extracted_df),
    }
    logger.info(
        "Domain extraction: %d input reads -> %d domains extracted "
        "(%d reads: no domain, %d: single, %d: multi-domain)",
        len(df), len(extracted_df), n_no_domain, n_single, n_multi,
    )
    return extracted_df, diagnostics
