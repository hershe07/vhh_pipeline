"""
clustering.py
--------------
Section 11: cluster similar nanobody sequences.

Method comparison (summarized; full discussion in Critical Review / README):

- Hierarchical (agglomerative, e.g. average-linkage on a distance matrix):
  + Deterministic, interpretable dendrogram, no need to pre-pick k precisely.
  - O(N^2) memory/time for the distance matrix; doesn't scale past ~10-20k
    sequences without pre-binning.

- DBSCAN (density-based, on a distance matrix or embedding):
  + No need to specify number of clusters; naturally labels outliers (-1),
    which is directly useful for Section 15 (novel/outlier discovery).
  - Sensitive to eps; struggles with clusters of very different density,
    which is common in repertoires (a few big clonal expansions + a long
    tail of singletons).

- HDBSCAN:
  + Like DBSCAN but handles variable density automatically, generally the
    best default choice for repertoire data with abundance skew, and still
    flags outliers. Recommended default here.
  - Slightly more complex parameters (min_cluster_size, min_samples);
    still needs a distance/embedding input.

- UMAP / t-SNE: NOT clustering algorithms themselves -- dimensionality
  reduction for VISUALIZATION (and, for UMAP, sometimes as a pre-step
  feeding HDBSCAN on the reduced embedding for speed). t-SNE is good for
  visualization but its axes/distances are not globally meaningful and it
  should not be used to justify cluster assignments; UMAP better preserves
  some global structure and is faster, but the same caveat about not
  over-interpreting embedding distances applies.

- Embeddings from protein language models (ESM-2, ProtT5): capture
  higher-order sequence/structure-correlated features beyond edit distance,
  which can help separate sequences that are similar in raw sequence but
  biophysically different (or vice versa). Optional here (compute-heavy);
  see Section 14 / Critical Review for when this is actually worth it for
  a VHH repertoire (short, ~130 aa domains where classical alignment-based
  distance is already quite informative -- PLM embeddings add the most
  value for structure/function-aware clustering, e.g. grouping by putative
  paratope similarity rather than just sequence identity).

This module supports clustering on:
  (a) a precomputed distance matrix (from similarity.py), or
  (b) numeric embeddings (PLM or simple k-mer/one-hot).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

try:
    import hdbscan as hdbscan_lib  # type: ignore
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False


# --------------------------- k-mer embedding (fast, no external deps) ---------------------------
def kmer_embedding(sequences: list[str], k: int = 3) -> np.ndarray:
    """
    Simple, fast, reference-free embedding: normalized k-mer frequency vectors.
    Useful as a default when PLM embeddings aren't available/worth the cost,
    and as a cheap pre-binning feature before exact alignment (see similarity.py).
    """
    from itertools import product
    alphabet = sorted(set("".join(sequences)))
    kmers = ["".join(p) for p in product(alphabet, repeat=k)]
    kmer_idx = {kmer: i for i, kmer in enumerate(kmers)}
    mat = np.zeros((len(sequences), len(kmers)), dtype=np.float32)
    for row, seq in enumerate(sequences):
        for i in range(len(seq) - k + 1):
            km = seq[i:i + k]
            if km in kmer_idx:
                mat[row, kmer_idx[km]] += 1
        total = mat[row].sum()
        if total > 0:
            mat[row] /= total
    return mat


# --------------------------- PLM embedding (optional, heavy) ---------------------------
def esm2_embedding(sequences: list[str], model_name: str = "esm2_t12_35M_UR50D",
                    batch_size: int = 16, device: str = "cpu") -> np.ndarray:
    """
    Optional ESM-2 embedding via `fair-esm`. Requires torch + fair-esm installed
    (not in default requirements.txt -- see Section 14 recommendation on when
    this is worth the extra dependency/compute).
    Returns per-sequence mean-pooled representations from the final layer.
    """
    try:
        import torch
        import esm  # type: ignore
    except ImportError as e:
        raise ImportError(
            "esm2_embedding requires `torch` and `fair-esm`. Install with "
            "`pip install torch fair-esm` (or use kmer_embedding instead)."
        ) from e

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    n_layers = model.num_layers

    embeddings = []
    for start in range(0, len(sequences), batch_size):
        batch = [(str(i), seq) for i, seq in enumerate(sequences[start:start + batch_size])]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[n_layers])
        reps = out["representations"][n_layers]
        for i, (_, seq) in enumerate(batch):
            embeddings.append(reps[i, 1: len(seq) + 1].mean(0).cpu().numpy())
        logger.info("ESM-2 embedded %d/%d sequences", min(start + batch_size, len(sequences)), len(sequences))
    return np.vstack(embeddings)


# --------------------------- clustering algorithms ---------------------------
def cluster_hierarchical(distance_matrix: np.ndarray, n_clusters: Optional[int] = None,
                          distance_threshold: Optional[float] = None) -> np.ndarray:
    """Agglomerative clustering on a precomputed distance matrix (average linkage)."""
    if n_clusters is None and distance_threshold is None:
        raise ValueError("Provide either n_clusters or distance_threshold.")
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(distance_matrix)


def cluster_dbscan(distance_matrix: np.ndarray, eps: float = 0.15, min_samples: int = 3) -> np.ndarray:
    """DBSCAN on a precomputed distance matrix. Label -1 = outlier/noise point."""
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    return model.fit_predict(distance_matrix)


def cluster_hdbscan(embedding_or_distance: np.ndarray, min_cluster_size: int = 5,
                     min_samples: Optional[int] = None, metric: str = "euclidean") -> np.ndarray:
    """
    HDBSCAN -- recommended default. Pass metric='precomputed' if
    embedding_or_distance is a distance matrix, otherwise a raw embedding.
    """
    if not _HAS_HDBSCAN:
        raise ImportError("hdbscan is not installed (`pip install hdbscan`).")
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric
    )
    return clusterer.fit_predict(np.asarray(embedding_or_distance, dtype=np.float64))


def reduce_umap(embedding: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1,
                 n_components: int = 2, metric: str = "euclidean") -> np.ndarray:
    if not _HAS_UMAP:
        raise ImportError("umap-learn is not installed (`pip install umap-learn`).")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         n_components=n_components, metric=metric, random_state=42)
    return reducer.fit_transform(embedding)


def reduce_tsne(embedding: np.ndarray, perplexity: float = 30, n_components: int = 2) -> np.ndarray:
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init="pca")
    return tsne.fit_transform(embedding)


def cluster_summary(labels: np.ndarray, ids: list[str]) -> pd.DataFrame:
    """Cluster sizes + membership, with cluster -1 (if present) labeled as noise/outlier."""
    df = pd.DataFrame({"id": ids, "cluster": labels})
    sizes = df["cluster"].value_counts().rename("size").reset_index().rename(columns={"index": "cluster"})
    sizes["is_outlier_cluster"] = sizes["cluster"] == -1
    sizes = sizes.sort_values("size", ascending=False).reset_index(drop=True)
    return sizes
