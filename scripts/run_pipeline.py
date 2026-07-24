#!/usr/bin/env python3
"""
run_pipeline.py
-----------------
End-to-end orchestration of the reference-free VHH repertoire pipeline.

Usage:
    python scripts/run_pipeline.py --input data/reads.fastq --outdir results/

See README.md for the full module-by-module description. This script wires
the utils/* modules together in order and writes all intermediate + final
artifacts (tables, FASTA, figures, JSON/Markdown report) to --outdir.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import (
    io_utils,
    qc,
    length_filter,
    translate,
    vhh_identify,
    cdr_annotation,
    cdr_length_analysis,
    unique_seqs,
    similarity,
    clustering,
    diversity,
    plotting,
    motif,
    novel_discovery,
    report,
    developability,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


def parse_args():
    p = argparse.ArgumentParser(description="Reference-free Nanopore VHH repertoire pipeline")
    p.add_argument("--input", required=True, help="FASTQ or FASTA input file")
    p.add_argument("--outdir", required=True, help="Output directory root")
    p.add_argument("--is-consensus", action="store_true", help="Input is a consensus FASTA (skips quality filter)")
    p.add_argument("--min-qscore", type=float, default=9.0, help="Minimum mean Phred quality")
    p.add_argument("--expected-length", type=int, default=397, help="Expected amplicon length (bp)")
    p.add_argument("--length-tolerance", type=float, default=0.10, help="Fractional length tolerance")
    p.add_argument("--expected-aa-length", type=int, default=132, help="Expected translated VHH length (aa)")
    p.add_argument("--cdr-scheme", default="imgt", choices=["imgt", "kabat", "chothia", "aho"])
    p.add_argument("--clustering-method", default="hdbscan", choices=["hierarchical", "dbscan", "hdbscan"])
    p.add_argument("--use-plm-embeddings", action="store_true", help="Use ESM-2 embeddings for clustering (heavy)")
    p.add_argument("--max-n-full-pairwise", type=int, default=3000,
                    help="Guardrail: max N for full O(N^2) pairwise similarity/clustering")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    fig_dir = outdir / "figures"
    res_dir = outdir / "tables"
    for d in (outdir, fig_dir, res_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Load
    df = io_utils.load_input(args.input, is_consensus=args.is_consensus)
    n_initial = len(df)
    before_qc_summary = qc.summarize_reads(df, "raw")

    # 2. QC
    passed_qc, failed_qc = qc.filter_by_quality(df, min_mean_qscore=args.min_qscore)
    qc.plot_quality_distribution(df, fig_dir / "quality_before.png", "Quality (before filtering)")
    qc.plot_quality_distribution(passed_qc, fig_dir / "quality_after.png", "Quality (after filtering)")
    qc.plot_before_after(df, passed_qc, "mean_qscore", fig_dir / "quality_before_after.png",
                          "Mean Phred quality", "Quality: before vs after filtering")
    qc_report_df = qc.qc_report(df, passed_qc)
    io_utils.write_table(qc_report_df, res_dir / "qc_report.csv")

    # 3. Length filtering
    min_len, max_len = length_filter.compute_length_window(args.expected_length, args.length_tolerance)
    qc.plot_length_distribution(passed_qc, fig_dir / "length_before_filter.png",
                                 "Length distribution (before length filter)",
                                 args.expected_length, min_len, max_len)
    passed_len, failed_len, len_report = length_filter.filter_by_length(
        passed_qc, args.expected_length, args.length_tolerance
    )
    qc.plot_length_distribution(passed_len, fig_dir / "length_after_filter.png",
                                 "Length distribution (after length filter)",
                                 args.expected_length, min_len, max_len)
    io_utils.write_table(pd.DataFrame([len_report]), res_dir / "length_filter_report.csv")

    # 4. Translation
    translated = translate.translate_dataframe(passed_len, expected_aa_length=args.expected_aa_length)
    valid_orf, invalid_orf, orf_report = translate.split_valid_orfs(translated)

    # 5. VHH identification (hallmark pre-screen)
    identified = vhh_identify.identify_vhh_dataframe(valid_orf)
    confirmed_vhh = identified[identified["vhh_call"].isin(["complete_vhh", "partial_vhh"])].copy()

    # 6. CDR annotation (ANARCI / fallback)
    annotated = cdr_annotation.annotate_dataframe(confirmed_vhh, scheme=args.cdr_scheme)
    io_utils.write_table(annotated, res_dir / "annotated_sequences.csv")

    # 7 & 8. CDR length analysis + combinations
    cdr_table = cdr_length_analysis.build_cdr_table(annotated)
    io_utils.write_table(cdr_table, res_dir / "cdr_table.csv")
    length_dists = cdr_length_analysis.cdr_length_distributions(cdr_table)
    cdr_length_analysis.plot_cdr_length_distributions(length_dists, fig_dir)
    combo_df = cdr_length_analysis.cdr_length_combinations(cdr_table)
    combo_df = cdr_length_analysis.flag_rare_combinations(combo_df)
    io_utils.write_table(combo_df, res_dir / "cdr_length_combinations.csv")
    if len(combo_df):
        cdr_length_analysis.plot_combination_bar(combo_df, fig_dir / "cdr_combination_bar.png")
        cdr_length_analysis.plot_combination_heatmap(cdr_table, fig_dir / "cdr23_heatmap.png")

    # 9. Unique sequence detection
    uniq = unique_seqs.unique_nt_and_aa(annotated)
    io_utils.write_table(uniq["unique_nt"].drop(columns=["member_ids"]), res_dir / "unique_nt_abundance.csv")
    io_utils.write_fasta(uniq["unique_aa"], res_dir / "unique_aa_sequences.fasta",
                          id_col="representative_id", seq_col="protein" if "protein" in uniq["unique_aa"] else "protein")
    plotting.plot_abundance_rank_curve(uniq["unique_aa"], fig_dir / "aa_abundance_rank.png")
    nt_diversity_quick = unique_seqs.diversity_from_abundance(uniq["unique_nt"])
    aa_diversity_quick = unique_seqs.diversity_from_abundance(uniq["unique_aa"])

    # 10 & 11. Similarity + clustering (guarded for size)
    aa_seqs = uniq["unique_aa"]["protein"].tolist() if "protein" in uniq["unique_aa"].columns else annotated["protein"].unique().tolist()
    cluster_labels = None
    if len(aa_seqs) <= args.max_n_full_pairwise:
        dist = similarity.pairwise_distance_matrix(aa_seqs, method="levenshtein",
                                                     max_n_full=args.max_n_full_pairwise)
        if args.clustering_method == "hierarchical":
            cluster_labels = clustering.cluster_hierarchical(dist, distance_threshold=0.15)
        elif args.clustering_method == "dbscan":
            cluster_labels = clustering.cluster_dbscan(dist, eps=0.15, min_samples=3)
        else:
            cluster_labels = clustering.cluster_hdbscan(dist, min_cluster_size=5, metric="precomputed")
    else:
        logger.warning(
            "N=%d unique AA sequences exceeds max_n_full_pairwise=%d; skipping full pairwise "
            "clustering. Use k-mer embeddings + HDBSCAN or pre-binning instead (see clustering.py).",
            len(aa_seqs), args.max_n_full_pairwise,
        )
        emb = clustering.kmer_embedding(aa_seqs, k=3)
        cluster_labels = clustering.cluster_hdbscan(emb, min_cluster_size=5, metric="euclidean")

    if cluster_labels is not None:
        uniq["unique_aa"]["cluster"] = cluster_labels
        cluster_stats_df = clustering.cluster_summary(cluster_labels, uniq["unique_aa"]["representative_id"].tolist())
        io_utils.write_table(cluster_stats_df, res_dir / "cluster_summary.csv")
        cluster_stats = {"n_clusters": int((cluster_stats_df["cluster"] != -1).sum()),
                          "n_outliers": int(cluster_stats_df.loc[cluster_stats_df["cluster"] == -1, "size"].sum())
                          if (cluster_stats_df["cluster"] == -1).any() else 0}
    else:
        cluster_stats = {}

    # Write unique_aa_abundance.csv now, AFTER clustering, so the 'cluster'
    # column (if computed) is actually included -- writing this earlier meant
    # the dashboard's Cluster Explorer tab never saw cluster assignments even
    # when clustering ran successfully.
    io_utils.write_table(uniq["unique_aa"].drop(columns=["member_ids"]), res_dir / "unique_aa_abundance.csv")

    # 14b. Developability screening (Section 19 recommended extension) -- run on every
    # unique amino-acid sequence, not just candidates, since it's cheap; the dashboard/
    # spreadsheet can be sorted or filtered by these columns for top candidates.
    dev_col = "protein" if "protein" in uniq["unique_aa"].columns else None
    if dev_col:
        dev_screen_df = developability.screen_dataframe(uniq["unique_aa"], protein_col=dev_col)
        io_utils.write_table(
            dev_screen_df.drop(columns=["member_ids"]) if "member_ids" in dev_screen_df.columns else dev_screen_df,
            res_dir / "developability_screen.csv",
        )
        dev_summary = {
            "n_screened": len(dev_screen_df),
            "n_with_aggregation_prone_region": int((dev_screen_df["n_aggregation_prone_regions"] > 0).sum()),
            "n_with_odd_cysteine_count": int(dev_screen_df["cysteine_count_odd"].sum()),
            "n_flagged_unstable": int(dev_screen_df["is_unstable"].fillna(False).sum()),
            "n_with_glycosylation_site": int((dev_screen_df["n_glycosylation_sites"] > 0).sum()),
        }
    else:
        dev_summary = {}

    # 12. Diversity
    cdr_div_df = diversity.cdr_diversity_table(cdr_table)
    io_utils.write_table(cdr_div_df, res_dir / "cdr_diversity.csv")
    for cdr in ("CDR1", "CDR2", "CDR3"):
        plotting.plot_diversity_bar(cdr_div_df, "shannon_entropy", fig_dir / "cdr_diversity_shannon.png",
                                     "CDR Shannon entropy") if cdr == "CDR3" else None
    aa_rarefaction = diversity.rarefaction_curve(uniq["unique_aa"]["copy_number"])
    plotting.plot_rarefaction(aa_rarefaction, fig_dir / "rarefaction_aa.png",
                               "Rarefaction: unique AA sequences vs read depth")

    # 13. Motif discovery (on CDR3s grouped by length, largest group)
    cdr3s = cdr_table["CDR3"].dropna().tolist()
    if cdr3s:
        try:
            kmer_df = motif.kmer_enrichment(cdr3s, k=3, n_shuffles=50)
            io_utils.write_table(kmer_df.head(50), res_dir / "cdr3_kmer_enrichment.csv")
        except Exception as e:
            logger.warning("Motif discovery skipped: %s", e)

    # 15. Novel/rare/outlier discovery
    #
    # flag_candidates() needs two per-sequence signals that unique_aa doesn't
    # carry on its own: (a) whether this sequence's CDR length combination was
    # flagged 'rare' in Section 8, and (b) the read quality backing it up.
    # Both are per-READ attributes (from `annotated`/`cdr_table`), so we look
    # them up via each unique sequence's representative_id (== a read_id) and
    # merge them in here. Previously these were silently missing, which meant
    # the novelty score only ever reflected cluster-outlier status.
    read_aux = annotated[["read_id", "mean_qscore"]].copy()
    if len(combo_df):
        cdr_rarity = cdr_table.merge(
            combo_df[["CDR1_length", "CDR2_length", "CDR3_length", "is_rare"]],
            on=["CDR1_length", "CDR2_length", "CDR3_length"],
            how="left",
        )[["read_id", "is_rare"]]
        read_aux = read_aux.merge(cdr_rarity, on="read_id", how="left")
    else:
        read_aux["is_rare"] = False
    read_aux = read_aux.rename(columns={"read_id": "representative_id", "is_rare": "has_rare_cdr_combo"})

    novel_input = uniq["unique_aa"].merge(read_aux, on="representative_id", how="left")
    novel_flagged = novel_discovery.flag_candidates(
        novel_input,
        cluster_col="cluster" if "cluster" in novel_input.columns else None,
        rare_combo_col="has_rare_cdr_combo",
    )
    io_utils.write_table(novel_flagged, res_dir / "novel_candidates.csv")
    novel_summary = novel_discovery.summarize_novel_candidates(novel_flagged)

    # 17. Summary report
    final_report = report.build_summary_report(
        n_initial_reads=n_initial,
        n_after_qc=len(passed_qc),
        n_after_length_filter=len(passed_len),
        n_valid_vhh=len(confirmed_vhh),
        n_unique_nt=len(uniq["unique_nt"]),
        n_unique_aa=len(uniq["unique_aa"]),
        n_unique_cdr3=cdr_table["CDR3"].nunique(),
        n_unique_cdr_combinations=len(combo_df),
        diversity_metrics={
            "nt_level": nt_diversity_quick,
            "aa_level": aa_diversity_quick,
            "cdr_level": cdr_div_df.to_dict(orient="records"),
        },
        cluster_stats=cluster_stats,
        extra={"orf_report": orf_report, "length_filter_report": len_report,
               "novel_candidates": novel_summary,
               "developability_screen": dev_summary},
    )
    report.write_report(final_report, outdir)
    logger.info("Pipeline complete. Outputs in %s", outdir)


if __name__ == "__main__":
    main()
