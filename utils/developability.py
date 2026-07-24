"""
developability.py
--------------------
Sequence-based developability / liability screening for candidate VHH
sequences -- cheap, reference-free checks worth running on your top
abundance/novelty candidates before committing to expression or
follow-up work (Section 19's "recommended extensions" list).

None of this requires structural prediction; everything here is computed
directly from the amino acid sequence:

- Isoelectric point (pI), molecular weight, GRAVY (hydropathy), aromaticity,
  instability index -- via Biopython's ProtParam (Bio.SeqUtils.ProtParam),
  the standard tool for these calculations.
- Aggregation-prone regions (APRs): a simple sliding-window hydrophobicity
  scan (Kyte-Doolittle scale). This is a lightweight heuristic, NOT a
  substitute for dedicated aggregation predictors (e.g. TANGO, AGGRESCAN,
  CamSol) -- those use learned/biophysical models and should be used for
  anything beyond a first-pass triage.
- N-linked glycosylation sequons (N-X-S/T, X != Proline): relevant mainly
  if you plan to express in a mammalian/yeast host where N-glycosylation
  occurs; largely irrelevant for E. coli expression.
- Unpaired/odd cysteine count: canonical VHH has 2 Cys (the conserved
  intradomain disulfide); some VHH subfamilies have a second, CDR-tethering
  disulfide (4 Cys). An ODD number of Cys is the actionable red flag here
  (an unpaired Cys is a common cause of aggregation/mispairing); an even
  count of 2 or 4 is expected and not flagged.
"""
from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Kyte-Doolittle hydropathy scale
_KD_SCALE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

_NGLYC_RE = re.compile(r"N[^P][ST]")


def _try_import_protparam():
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis  # type: ignore
        return ProteinAnalysis
    except ImportError:
        return None


def basic_physicochemical_properties(protein: str) -> dict:
    """pI, MW, GRAVY, aromaticity, instability index via Biopython ProtParam."""
    ProteinAnalysis = _try_import_protparam()
    clean = protein.replace("*", "").replace("X", "")  # ProtParam can't handle stop/unknown symbols
    if ProteinAnalysis is None or len(clean) < 5:
        return {
            "molecular_weight": np.nan, "isoelectric_point": np.nan,
            "gravy": np.nan, "aromaticity": np.nan, "instability_index": np.nan,
        }
    try:
        pa = ProteinAnalysis(clean)
        return {
            "molecular_weight": pa.molecular_weight(),
            "isoelectric_point": pa.isoelectric_point(),
            "gravy": pa.gravy(),
            "aromaticity": pa.aromaticity(),
            "instability_index": pa.instability_index(),  # >40 conventionally flagged 'unstable'
        }
    except Exception as e:  # pragma: no cover -- ProtParam can choke on unusual residues
        logger.warning("ProtParam failed on a sequence (len=%d): %s", len(clean), e)
        return {
            "molecular_weight": np.nan, "isoelectric_point": np.nan,
            "gravy": np.nan, "aromaticity": np.nan, "instability_index": np.nan,
        }


def find_aggregation_prone_regions(protein: str, window: int = 7, threshold: float = 2.0) -> list[dict]:
    """
    Sliding-window mean Kyte-Doolittle hydropathy scan. Windows with mean
    hydropathy >= threshold are flagged as candidate aggregation-prone
    regions (APRs). Overlapping flagged windows are merged into a single
    span for readability.
    """
    if len(protein) < window:
        return []
    scores = np.array([_KD_SCALE.get(aa, 0.0) for aa in protein])
    window_means = np.convolve(scores, np.ones(window) / window, mode="valid")
    flagged = window_means >= threshold

    spans = []
    start = None
    for i, is_flagged in enumerate(flagged):
        if is_flagged and start is None:
            start = i
        elif not is_flagged and start is not None:
            spans.append((start, i + window - 1))
            start = None
    if start is not None:
        spans.append((start, len(flagged) + window - 1))

    return [
        {"start": s, "end": e, "sequence": protein[s:e], "mean_hydropathy": float(scores[s:e].mean())}
        for s, e in spans
    ]


def find_glycosylation_sites(protein: str) -> list[dict]:
    """N-X-S/T sequons (X != Proline). Only relevant for mammalian/yeast expression hosts."""
    return [{"position": m.start() + 1, "sequon": m.group()} for m in _NGLYC_RE.finditer(protein)]


def cysteine_liability(protein: str) -> dict:
    n_cys = protein.count("C")
    return {
        "n_cysteines": n_cys,
        "cysteine_count_odd": bool(n_cys % 2 == 1),  # actionable red flag: unpaired Cys likely
    }


def screen_sequence(protein: str) -> dict:
    """Run all developability checks on one protein sequence; returns a flat dict."""
    props = basic_physicochemical_properties(protein)
    aprs = find_aggregation_prone_regions(protein)
    glyc = find_glycosylation_sites(protein)
    cys = cysteine_liability(protein)

    return {
        **props,
        "n_aggregation_prone_regions": len(aprs),
        "aggregation_prone_regions": "; ".join(f"{a['sequence']}({a['start']+1}-{a['end']})" for a in aprs),
        "n_glycosylation_sites": len(glyc),
        "glycosylation_sites": "; ".join(f"{g['sequon']}@{g['position']}" for g in glyc),
        **cys,
        "is_unstable": bool(props["instability_index"] > 40) if not np.isnan(props["instability_index"]) else None,
    }


def screen_dataframe(df: pd.DataFrame, protein_col: str = "protein") -> pd.DataFrame:
    """Apply screen_sequence to every row; adds developability columns."""
    results = df[protein_col].apply(screen_sequence)
    result_df = pd.DataFrame(list(results), index=df.index)
    out = pd.concat([df, result_df], axis=1)

    n_flagged = int((out["n_aggregation_prone_regions"] > 0).sum())
    n_odd_cys = int(out["cysteine_count_odd"].sum())
    n_unstable = int(out["is_unstable"].fillna(False).sum())
    logger.info(
        "Developability screen: %d/%d sequences with >=1 APR, %d with odd Cys count, %d flagged unstable (instability index >40)",
        n_flagged, len(out), n_odd_cys, n_unstable,
    )
    return out
