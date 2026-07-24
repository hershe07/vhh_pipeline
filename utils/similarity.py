"""
similarity.py
--------------
Section 10: since there is no reference sequence, all similarity is computed
internally -- sequences are compared against each other (all-vs-all or
against a set of medoids/representatives).

Method notes (why each is included, and when to use it):

- Levenshtein (edit) distance: fast, alignment-free-ish string distance.
  Good default for large N (via python-Levenshtein's C implementation) and
  for near-duplicate / Nanopore-error-collapsing use cases where sequences
  are almost the same length.
- Global (Needleman-Wunsch) alignment: appropriate for full-length VHH
  domains that should align end-to-end (they're all the same domain type).
  Use when you want a proper %identity over the whole aligned length.
- Local (Smith-Waterman) alignment: better when sequences may be partial/
  truncated (e.g. before length filtering removes fragments) or when you
  only care about the best-matching sub-region.
- BLOSUM62 substitution scoring: used inside NW/SW for protein alignments,
  since raw identity ignores biochemically conservative substitutions and
  is noisier for the low-identity comparisons typical of CDR3.

Scalability (recommendation, elaborated in Section 19 / README):
  - Pure-Python Bio.pairwise2 does NOT scale past a few thousand sequences
    for all-vs-all (O(N^2) alignments, each O(L^2)).
  - `parasail` (SIMD-vectorized C library) is 10-100x faster and is the
    recommended engine here; this module uses it when available and falls
    back to Biopython's PairwiseAligner (much slower) otherwise.
  - For truly large repertoires (>~50k unique sequences), avoid full N^2
    entirely: pre-cluster with fast k-mer/MinHash sketches (e.g. via
    `datasketch`, or simple k-mer Jaccard) or length+CDR3-length binning,
    THEN do exact alignment only within candidate bins. See clustering.py
    and the Critical Review for details.
"""
from __future__ import annotations

import itertools
import logging
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import parasail  # type: ignore
    _HAS_PARASAIL = True
except ImportError:
    _HAS_PARASAIL = False

try:
    import Levenshtein  # type: ignore
    _HAS_LEVENSHTEIN = True
except ImportError:
    _HAS_LEVENSHTEIN = False


def levenshtein_distance(a: str, b: str) -> int:
    if _HAS_LEVENSHTEIN:
        return Levenshtein.distance(a, b)
    # O(len(a)*len(b)) pure-python fallback -- fine for small N / short CDRs only
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def percent_identity_from_levenshtein(a: str, b: str) -> float:
    """1 - normalized edit distance, as a cheap identity proxy (not alignment-based %id)."""
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return 1.0 - dist / max_len if max_len else 1.0


def global_alignment_identity(a: str, b: str, is_protein: bool = True,
                               gap_open: int = 10, gap_extend: int = 1) -> dict:
    """Needleman-Wunsch global alignment; returns score and %identity over the alignment."""
    if _HAS_PARASAIL:
        matrix = parasail.blosum62 if is_protein else parasail.dnafull
        result = parasail.nw_stats(a, b, gap_open, gap_extend, matrix)
        aln_len = result.length
        identity = result.matches / aln_len if aln_len else 0.0
        return {"score": result.score, "identity": identity, "alignment_length": aln_len}
    else:
        from Bio.Align import PairwiseAligner, substitution_matrices
        aligner = PairwiseAligner()
        aligner.mode = "global"
        if is_protein:
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -gap_open
        aligner.extend_gap_score = -gap_extend
        aln = aligner.align(a, b)[0]
        aligned_a, aligned_b = str(aln[0]), str(aln[1])
        matches = sum(1 for x, y in zip(aligned_a, aligned_b) if x == y and x != "-")
        aln_len = len(aligned_a)
        return {"score": aln.score, "identity": matches / aln_len if aln_len else 0.0, "alignment_length": aln_len}


def local_alignment_identity(a: str, b: str, is_protein: bool = True,
                              gap_open: int = 10, gap_extend: int = 1) -> dict:
    """Smith-Waterman local alignment; useful for partial/truncated sequences."""
    if _HAS_PARASAIL:
        matrix = parasail.blosum62 if is_protein else parasail.dnafull
        result = parasail.sw_stats(a, b, gap_open, gap_extend, matrix)
        aln_len = result.length
        identity = result.matches / aln_len if aln_len else 0.0
        return {"score": result.score, "identity": identity, "alignment_length": aln_len}
    else:
        from Bio.Align import PairwiseAligner, substitution_matrices
        aligner = PairwiseAligner()
        aligner.mode = "local"
        if is_protein:
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        aligner.open_gap_score = -gap_open
        aligner.extend_gap_score = -gap_extend
        aln = aligner.align(a, b)[0]
        aligned_a, aligned_b = str(aln[0]), str(aln[1])
        matches = sum(1 for x, y in zip(aligned_a, aligned_b) if x == y and x != "-")
        aln_len = len(aligned_a)
        return {"score": aln.score, "identity": matches / aln_len if aln_len else 0.0, "alignment_length": aln_len}


def pairwise_distance_matrix(
    sequences: list[str],
    method: str = "levenshtein",
    is_protein: bool = True,
    max_n_full: int = 3000,
) -> np.ndarray:
    """
    All-vs-all distance matrix (1 - identity, so 0=identical).

    Guardrail: for N above max_n_full this raises rather than silently
    running an O(N^2) job that may take hours/OOM -- callers should use
    clustering.py's sketch-based pre-binning for large N instead (see
    module docstring and Critical Review).
    """
    n = len(sequences)
    if n > max_n_full:
        raise ValueError(
            f"N={n} exceeds max_n_full={max_n_full} for full pairwise computation. "
            "Use k-mer/MinHash pre-clustering (clustering.py) to bin sequences first, "
            "or raise max_n_full if you have the compute budget (cost is O(N^2))."
        )

    dist = np.zeros((n, n), dtype=np.float32)
    for i, j in itertools.combinations(range(n), 2):
        if method == "levenshtein":
            d = 1.0 - percent_identity_from_levenshtein(sequences[i], sequences[j])
        elif method == "global":
            d = 1.0 - global_alignment_identity(sequences[i], sequences[j], is_protein)["identity"]
        elif method == "local":
            d = 1.0 - local_alignment_identity(sequences[i], sequences[j], is_protein)["identity"]
        else:
            raise ValueError(f"Unknown method: {method}")
        dist[i, j] = dist[j, i] = d
    return dist


def mean_pairwise_identity(sequences: list[str], method: str = "levenshtein",
                            is_protein: bool = True, max_n_full: int = 3000) -> float:
    """Single summary number: mean pairwise %identity across the (sub)set of sequences."""
    dist = pairwise_distance_matrix(sequences, method=method, is_protein=is_protein, max_n_full=max_n_full)
    iu = np.triu_indices_from(dist, k=1)
    if len(iu[0]) == 0:
        return 1.0
    return float(1.0 - dist[iu].mean())
