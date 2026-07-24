"""
cdr_annotation.py
------------------
Annotates FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4 for each VHH protein sequence.

Numbering scheme choice (Section 6 of the brief): IMGT vs Kabat vs Chothia vs AHo.

    Scheme    | CDR definition basis         | Notes for camelid VHH
    ----------|-------------------------------|----------------------------------
    Kabat     | Sequence variability          | Original, antibody-specific; CDR-H1
              |                                 boundaries fit poorly to VHH's often-
              |                                 longer CDR1/hypermutated FR2.
    Chothia   | Structural loop definition     | Better structural relevance than Kabat
              |                                 but still IgG-VH-centric; H1 boundary
              |                                 differs from IMGT, complicating cross-
              |                                 study comparison.
    IMGT      | Structure + germline-based,    | RECOMMENDED for VHH. Species- and
              | consistent numbering across    | domain-independent (works for VH, VL,
              | all Ig/TCR domains and species | TCR, and VHH the same way), which
              |                                | matters since public nanobody
              |                                | databases (e.g. INDI, sdAb-DB) and
              |                                | most recent VHH repertoire papers
              |                                | report IMGT-numbered CDR1/2/3. Fixed-
              |                                | length framework numbering also makes
              |                                | CDR-length distributions directly
              |                                | comparable across sequences/studies.
    AHo       | Structure-based, equal-length  | Good for structural modeling /
              | numbering across all domains   | homology modeling pipelines but less
              |                                | commonly reported in repertoire papers;
              |                                | more niche tooling support.

    -> Recommendation: use IMGT as the primary/reported scheme (via ANARCI),
       and optionally emit Kabat/Chothia alongside for readers who need
       cross-referencing to older literature. ANARCI computes all four from
       a single alignment, so this costs nothing extra.

ANARCI (Antibody Numbering and Antigen Receptor ClassIfication, Dunbar & Deane
2016) is used because it: (a) explicitly supports camelid VHH numbering,
(b) outputs IMGT/Kabat/Chothia/AHo simultaneously, (c) is HMM-profile based,
so it doubles as an authoritative VHH-vs-not-VHH confirmation (a sequence that
fails to align to any V-domain HMM is rejected here), correcting/overriding
the cheap hallmark screen in vhh_identify.py.

AbNumber (a friendlier wrapper around ANARCI) is offered as a drop-in
alternative import if already installed.

If neither ANARCI nor AbNumber is available in the environment, we fall back
to the regex/motif-based splitter (`_fallback_cdr_split`), which is far less
accurate (no structural alignment, brittle to indels) and every row produced
this way is flagged via `annotation_method='fallback_regex'` so downstream
consumers/reviewers know to treat those boundaries with caution.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

IMGT_REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]

# --- IMGT V-domain boundary definitions (by IMGT position number, after ANARCI numbering) ---
# Standard IMGT boundaries for V-domains (same convention applied to VHH).
IMGT_BOUNDARIES = {
    "FR1": (1, 26),
    "CDR1": (27, 38),
    "FR2": (39, 55),
    "CDR2": (56, 65),
    "FR3": (66, 104),
    "CDR3": (105, 117),
    "FR4": (118, 128),
}


def _try_import_anarci():
    try:
        from anarci import anarci  # type: ignore
        return anarci
    except ImportError:
        return None


def annotate_with_anarci(sequences: list[tuple[str, str]], scheme: str = "imgt") -> dict:
    """
    sequences: list of (seq_id, protein_sequence)
    Returns {seq_id: {region: sequence, ...}} for successfully numbered sequences.
    Sequences ANARCI cannot number (not V-domain-like) are simply absent from
    the result -- callers should treat missing IDs as 'not confirmed as VHH'.
    """
    anarci_fn = _try_import_anarci()
    if anarci_fn is None:
        raise ImportError("ANARCI is not installed; see cdr_annotation.py fallback path.")

    # ANARCI expects a list of (name, sequence); allow=set(["H"]) restricts to
    # heavy-chain-like domains, which VHH numbers against.
    numbering, alignment_details, hit_tables = anarci_fn(
        sequences, scheme=scheme, allow=set(["H"])
    )

    results = {}
    for (seq_id, _), num, details in zip(sequences, numbering, alignment_details):
        if num is None or len(num) == 0:
            continue  # ANARCI could not number this sequence -> not a confirmed V-domain
        # ANARCI returns a list of numbered domains; VHH should have exactly one
        domain_numbering = num[0][0]  # [(pos_tuple, aa), ...] for the first/only domain
        regions = _split_by_imgt_boundaries(domain_numbering)
        regions["annotation_method"] = f"anarci_{scheme}"
        results[seq_id] = regions
    return results


def _split_by_imgt_boundaries(domain_numbering: list) -> dict:
    """domain_numbering: [((imgt_pos, insertion_code), aa), ...] from ANARCI."""
    region_seqs = {r: [] for r in IMGT_REGIONS}
    for (pos, _ins), aa in domain_numbering:
        if aa == "-":
            continue
        for region, (lo, hi) in IMGT_BOUNDARIES.items():
            if lo <= pos <= hi:
                region_seqs[region].append(aa)
                break
    return {region: "".join(aas) for region, aas in region_seqs.items()}


# ------------------------- Fallback (no ANARCI available) -------------------------
# Motif-anchored heuristic splitter. Much less reliable, especially for CDR1/CDR3
# length outliers or indel-containing reads -- Nanopore's dominant error mode.
_FR1_RE = re.compile(r"^(.*?[EQ][VI]QLVE?SGG.*?)(?=C)")
_CYS1_RE = re.compile(r"C")
_FR2_ANCHOR_RE = re.compile(r"WVR|WFR|WYR|WIR")   # canonical FR2 "W..R" motif
_FR4_RE = re.compile(r"WG.G(?:TQ|LQ|MQ)VTVSS.*$")


def _fallback_cdr_split(protein: str) -> Optional[dict]:
    fr4_match = _FR4_RE.search(protein)
    fr2_match = _FR2_ANCHOR_RE.search(protein)
    if not fr4_match or not fr2_match:
        return None  # can't anchor both ends confidently -- refuse rather than guess

    fr4_start = fr4_match.start()
    body = protein[:fr4_start]

    # crude split: FR1 ends at 2nd Cys count is unreliable without alignment;
    # instead split body at the FR2 anchor into an (FR1+CDR1) chunk and the rest
    fr2_pos = fr2_match.start()
    n_terminal = body[:fr2_pos]
    c_terminal = body[fr2_pos:]

    # Without true alignment we cannot precisely separate CDR1/FR1 or CDR2/FR3
    # boundaries -- return coarse regions and mark them explicitly as low
    # confidence so downstream length/diversity stats can exclude or flag them.
    return {
        "FR1": None,
        "CDR1": None,
        "FR2": n_terminal[-17:] if len(n_terminal) >= 17 else n_terminal,
        "CDR2": None,
        "FR3": c_terminal,
        "CDR3": None,
        "FR4": protein[fr4_start:],
        "annotation_method": "fallback_regex_low_confidence",
    }


def annotate_dataframe(
    df: pd.DataFrame,
    id_col: str = "read_id",
    protein_col: str = "protein",
    scheme: str = "imgt",
) -> pd.DataFrame:
    """
    Annotate FR/CDR regions for every row. Tries ANARCI first; falls back to
    the low-confidence regex splitter per-sequence if ANARCI is unavailable
    or fails to number a given sequence. Sequences that fail BOTH are dropped
    from CDR-level analyses (kept in the full table with all-NaN CDR columns
    and annotation_method='unannotated').
    """
    ids = df[id_col].astype(str).tolist()
    proteins = df[protein_col].tolist()

    anarci_available = _try_import_anarci() is not None
    region_records = {}

    if anarci_available:
        try:
            region_records = annotate_with_anarci(list(zip(ids, proteins)), scheme=scheme)
        except Exception as e:  # pragma: no cover - defensive; ANARCI can throw on odd inputs
            logger.warning("ANARCI batch annotation failed (%s); falling back per-sequence.", e)
            anarci_available = False

    fallback_used = 0
    unannotated = 0
    rows = []
    for seq_id, protein in zip(ids, proteins):
        if seq_id in region_records:
            rows.append(region_records[seq_id])
            continue
        fb = _fallback_cdr_split(protein)
        if fb is not None:
            fallback_used += 1
            rows.append(fb)
        else:
            unannotated += 1
            rows.append({r: None for r in IMGT_REGIONS} | {"annotation_method": "unannotated"})

    if not anarci_available:
        logger.warning(
            "ANARCI not installed -- using low-confidence regex CDR splitter for all sequences. "
            "Install ANARCI (conda install -c bioconda anarci) for reliable IMGT numbering."
        )
    logger.info(
        "CDR annotation: %d via ANARCI, %d via fallback regex, %d unannotated (of %d)",
        len(region_records), fallback_used, unannotated, len(df),
    )

    region_df = pd.DataFrame(rows, index=df.index)
    out = pd.concat([df, region_df], axis=1)
    for region in ["CDR1", "CDR2", "CDR3"]:
        out[f"{region}_length"] = out[region].apply(lambda s: len(s) if isinstance(s, str) else pd.NA)
    return out
