"""
vhh_identify.py
----------------
Confirms that a translated protein is a genuine camelid VHH (nanobody)
domain, as opposed to a partial read, an off-target amplicon, or noise
that happened to translate cleanly.

Two complementary layers:
1. Fast framework-motif / conserved-residue screen (this module) -- cheap,
   reference-free, and a good first pass on all sequences.
2. HMM/profile-based confirmation (delegate to ANARCI in cdr_annotation.py,
   which internally uses HMMER profiles for V-gene germlines including
   IGHV/VHH-like camelid frameworks) -- authoritative, used to confirm and
   to get FR/CDR boundaries in one step.

Rationale for the hallmark-residue check:
Camelid VHH domains are distinguished from conventional heavy-chain VH
domains by four hallmark substitutions in FR2 (Kabat positions ~37, 44, 45, 47),
classically Phe/Tyr37, Glu44, Arg45, Gly/Leu47 (VGLW motif region), which
replace residues that in conventional VH would pack against a light chain.
FR1 typically begins near a conserved "(E/Q)VQLVESGGGLVQ.GGSLRLSCAAS"-like
motif and FR4 ends in the conserved "WGQGTQVTVSS"-like motif, with an
essentially invariant Trp residue at the start of FR4 (Kabat ~103) and
paired conserved cysteines flanking CDR3 (one at the end of FR3, canonically
~Cys22 in FR1 and ~Cys96 near CDR3, forming the conserved intradomain
disulfide -- occasionally a second, VHH-specific disulfide is also present,
often tethering CDR1/CDR3 or CDR2/CDR3).

These are *heuristics* for a fast pre-screen; true confirmation should come
from ANARCI/HMM numbering (Section 6), which will also reject sequences
that don't fit an immunoglobulin V-domain profile at all.
"""
from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Conserved framework anchors (regex, deliberately somewhat permissive to tolerate
# Nanopore-derived single-base substitution noise that survives ORF filtering).
FR1_START_RE = re.compile(r"[EQ][VI]QLVE?SGG")
FR4_MOTIF_RE = re.compile(r"WG.G(?:TQ|LQ|MQ)VTVSS")
CYS_MIN_COUNT = 2  # canonical FR1/FR3 Cys pair forming the Ig-fold disulfide

# VHH-diagnostic FR2 hallmark positions (approximate substring window, since we
# have no aligned numbering yet at this stage -- this is a coarse screen only).
HALLMARK_RESIDUES = set("FYERGL")  # any of these appearing in the FR2 hallmark window counts as supportive evidence


def screen_vhh_hallmarks(protein: str) -> dict:
    """
    Cheap, reference-free scan for VHH hallmark features on an unaligned
    protein sequence. Returns a dict of boolean/countable evidence; does
    NOT make the final accept/reject call (see classify_vhh).
    """
    has_fr1 = bool(FR1_START_RE.search(protein))
    has_fr4 = bool(FR4_MOTIF_RE.search(protein))
    n_cys = protein.count("C")
    has_cys_pair = n_cys >= CYS_MIN_COUNT

    # crude FR2 hallmark window: conventionally ~40-50 residues in from a
    # detected FR1 start; only evaluated if FR1 was found
    fr2_window = ""
    if has_fr1:
        m = FR1_START_RE.search(protein)
        start = m.end()
        fr2_window = protein[start + 15: start + 45]  # heuristic offset
    hallmark_hits = sum(1 for aa in HALLMARK_RESIDUES if aa in fr2_window)

    return {
        "has_fr1_motif": has_fr1,
        "has_fr4_motif": has_fr4,
        "n_cysteines": n_cys,
        "has_conserved_cys_pair": has_cys_pair,
        "fr2_hallmark_hits": hallmark_hits,
    }


def classify_vhh(protein: str, min_length: int = 105, max_length: int = 140) -> dict:
    """
    Combine hallmark evidence into a coarse call: 'complete_vhh', 'partial_vhh',
    or 'not_vhh'. This is a pre-filter meant to be reconciled/overridden by the
    ANARCI numbering step, which is authoritative.
    """
    evidence = screen_vhh_hallmarks(protein)
    length_ok = min_length <= len(protein) <= max_length

    if evidence["has_fr1_motif"] and evidence["has_fr4_motif"] and evidence["has_conserved_cys_pair"] and length_ok:
        call = "complete_vhh"
    elif (evidence["has_fr1_motif"] or evidence["has_fr4_motif"]) and evidence["has_conserved_cys_pair"]:
        call = "partial_vhh"  # missing one terminal framework -> likely a truncated/partial read
    else:
        call = "not_vhh"

    evidence["length_ok"] = length_ok
    evidence["protein_length"] = len(protein)
    evidence["vhh_call"] = call
    return evidence


def identify_vhh_dataframe(df: pd.DataFrame, protein_col: str = "protein") -> pd.DataFrame:
    """Apply classify_vhh to every row; adds evidence columns + 'vhh_call'."""
    results = df[protein_col].apply(classify_vhh)
    evidence_df = pd.DataFrame(list(results), index=df.index)
    out = pd.concat([df, evidence_df], axis=1)
    counts = out["vhh_call"].value_counts().to_dict()
    logger.info("VHH hallmark screen: %s", counts)
    return out
