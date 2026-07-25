"""
structure_prediction.py
--------------------------
Predicts 3D structure for a shortlist of top candidate sequences (Section
19's recommended extension: "structural modeling of top novelty-score /
most-abundant clones to support functional claims").

Method: the ESM Metagenomic Atlas fold API (api.esmatlas.com), which runs
Meta AI's ESMFold on a submitted sequence and returns a PDB structure --
no local GPU, no multi-gigabyte model download, no MSA/database required
(unlike AlphaFold2). This is the practical choice for folding a small
shortlist (tens of sequences) of short (~130 aa) VHH domains on a laptop.

Trade-offs vs. running ESMFold locally (torch + fair-esm/transformers) or
AlphaFold2:
- Requires internet access and a public API with no SLA -- fine for an
  occasional shortlist of a few dozen sequences, not appropriate for
  folding your entire repertoire or for a production/high-throughput
  pipeline.
- ESMFold (single-sequence, no MSA) is generally somewhat less accurate
  than AlphaFold2 for regions lacking sequence homologs in ESM's training
  set, but is fast and single-sequence, which suits VHH CDR loops
  (relatively short, evolutionarily under-sampled regions where AF2's MSA
  advantage matters less anyway).
- The API has a practical sequence length ceiling (see MAX_SEQUENCE_LENGTH)
  -- comfortably above a VHH domain's ~110-140 aa, so this isn't a
  practical constraint here.

Confidence: ESMFold reports per-residue pLDDT (0-100, higher = more
confident) in the PDB B-factor column, exactly like AlphaFold2's output
convention. This module extracts a per-structure mean pLDDT as a quick
confidence summary -- treat structures with mean pLDDT below ~70 as low
confidence, consistent with general AlphaFold2/ESMFold usage guidance.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ESMATLAS_FOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
MAX_SEQUENCE_LENGTH = 400  # API practical ceiling; VHH domains (~110-140 aa) are well within this
REQUEST_TIMEOUT_S = 120


def fold_sequence(protein: str, max_retries: int = 2, retry_delay_s: float = 3.0) -> str | None:
    """
    Submits one protein sequence to the ESM Atlas fold API. Returns raw PDB
    text on success, or None on failure (network error, API error, or
    sequence too long) -- callers should treat None as "skip this one",
    not as a fatal error for the batch.
    """
    if len(protein) == 0 or len(protein) > MAX_SEQUENCE_LENGTH:
        logger.warning("Sequence length %d outside foldable range (1-%d); skipping.",
                        len(protein), MAX_SEQUENCE_LENGTH)
        return None

    clean = protein.replace("*", "").replace("X", "")  # API expects standard AA letters only

    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(ESMATLAS_FOLD_URL, data=clean, timeout=REQUEST_TIMEOUT_S)
            if resp.status_code == 200 and resp.text.startswith(("HEADER", "ATOM", "MODEL")):
                return resp.text
            logger.warning("ESM Atlas API returned status %d (attempt %d/%d)",
                            resp.status_code, attempt + 1, max_retries + 1)
        except requests.RequestException as e:
            logger.warning("ESM Atlas API request failed (attempt %d/%d): %s",
                            attempt + 1, max_retries + 1, e)
        if attempt < max_retries:
            time.sleep(retry_delay_s)
    return None


def mean_plddt_from_pdb(pdb_text: str) -> float:
    """
    ESMFold/AlphaFold2 convention: per-residue pLDDT is stored in the
    B-factor column of each ATOM record. Averages over all atoms
    (equivalently, since pLDDT is per-residue, over all residues weighted
    by atom count -- a minor approximation, fine for a summary statistic).

    The ESM Atlas API returns this on a 0-1 fractional scale in practice,
    NOT the 0-100 scale AlphaFold2 conventionally uses -- auto-detects
    which scale is in play (rather than hardcoding an assumption) so this
    keeps working correctly if that ever changes, and always returns a
    0-100 value for consistent downstream interpretation/thresholds.
    """
    values = []
    for line in pdb_text.splitlines():
        if line.startswith("ATOM"):
            try:
                b_factor = float(line[60:66])
                values.append(b_factor)
            except ValueError:
                continue
    if not values:
        return float("nan")
    mean_raw = sum(values) / len(values)
    # If every value is <= ~1.5, this is a 0-1 fractional scale -- rescale to 0-100.
    if max(values) <= 1.5:
        return mean_raw * 100
    return mean_raw


def predict_structures(
    candidates: pd.DataFrame,
    id_col: str = "representative_id",
    protein_col: str = "protein",
    outdir: str | Path = "structures",
    delay_between_requests_s: float = 1.0,
) -> pd.DataFrame:
    """
    Folds every sequence in `candidates` (expected to already be a small
    shortlist -- see select_top_candidates), saves each as a .pdb file
    under `outdir`, and returns a summary table with per-sequence success
    status and mean pLDDT.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (_, row) in enumerate(candidates.iterrows()):
        seq_id = str(row[id_col])
        protein = row[protein_col]
        logger.info("Folding %d/%d: %s (len=%d)", i + 1, len(candidates), seq_id, len(protein))

        pdb_text = fold_sequence(protein)
        if pdb_text is None:
            rows.append({id_col: seq_id, "fold_success": False, "mean_plddt": float("nan"), "pdb_path": None})
            continue

        pdb_path = outdir / f"{seq_id}.pdb"
        pdb_path.write_text(pdb_text)
        mean_plddt = mean_plddt_from_pdb(pdb_text)
        rows.append({
            id_col: seq_id, "fold_success": True, "mean_plddt": mean_plddt,
            "pdb_path": str(pdb_path), "confidence_tier": _confidence_tier(mean_plddt),
        })

        if i < len(candidates) - 1:
            time.sleep(delay_between_requests_s)  # be polite to a free public API

    result_df = pd.DataFrame(rows)
    n_ok = int(result_df["fold_success"].sum())
    logger.info("Structure prediction: %d/%d sequences folded successfully", n_ok, len(candidates))
    return result_df


def _confidence_tier(mean_plddt: float) -> str:
    if pd.isna(mean_plddt):
        return "unknown"
    if mean_plddt >= 90:
        return "very_high"
    if mean_plddt >= 70:
        return "confident"
    if mean_plddt >= 50:
        return "low"
    return "very_low"


def select_top_candidates(
    novel_flagged: pd.DataFrame,
    protein_lookup: pd.DataFrame,
    n: int = 10,
    id_col: str = "representative_id",
    protein_col: str = "protein",
    require_confirmed_cdr: bool = True,
    complete_domain_col: str = "complete_domain",
) -> pd.DataFrame:
    """
    Picks the top-N sequences by novelty_score from novel_discovery.py's
    output. novel_candidates.csv already carries the protein sequence
    (it's derived from the unique_aa table), so this only falls back to
    joining against `protein_lookup` if that column is somehow missing --
    avoids creating duplicate protein_x/protein_y columns from re-merging
    two tables that already share the column.

    require_confirmed_cdr=True (default) restricts candidates to sequences
    with ALL SEVEN regions (FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4) present --
    not merely a non-null CDR3. This matters more than it might look: ANARCI
    can successfully number and return a confident CDR3 for a read whose
    FR1 (or FR2) is missing/garbled from a frameshift artifact upstream of
    the real domain -- such a read is a genuine PARTIAL domain, not a
    complete one, and is exactly the kind of "novel" result driven by being
    structurally broken (hence a cluster outlier and singleton) rather than
    biologically interesting. Folding a partial domain's full translated
    ORF (which may carry substantial frameshifted garbage flanking the
    real V-domain) would waste an API call on a structure with no
    biological meaning. If `complete_domain_col` isn't present in
    `novel_flagged`, this filter is skipped with a warning.
    """
    candidates = novel_flagged
    if require_confirmed_cdr:
        if complete_domain_col in candidates.columns:
            gated = candidates[candidates[complete_domain_col].fillna(False)]
            if len(gated) == 0:
                logger.warning(
                    "No candidates passed the complete-domain gate; "
                    "falling back to ungated ranking (results may include partial/junk domains)."
                )
            else:
                candidates = gated
        else:
            logger.warning(
                "'%s' column not found on novel_flagged; skipping the complete-domain quality gate. "
                "Top candidates may include partial or frameshifted-artifact domains.", complete_domain_col,
            )

    top = candidates.sort_values("novelty_score", ascending=False).head(n)
    if protein_col not in top.columns:
        top = top.merge(protein_lookup[[id_col, protein_col]], on=id_col, how="left")
    return top.dropna(subset=[protein_col])
