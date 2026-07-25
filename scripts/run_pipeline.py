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
import matplotlib.pyplot as plt

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
    domain_extraction,
    structure_prediction,
    framework_mutations,
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
    p.add_argument("--n-structures", type=int, default=10,
                    help="Number of top novel candidates to fold via ESMFold (ESM Atlas API)")
    p.add_argument("--skip-structure-prediction", action="store_true",
                    help="Skip structure prediction entirely (e.g. if no internet access is available)")
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

    # 2.5 Domain extraction (default, not optional) -- rather than judging each raw
    # read's validity by its total length, pull out the actual VHH domain
    # sub-sequence(s) it contains (via conserved FR1/FR4 framework anchors), and
    # run length filtering on the EXTRACTED domain length, not the raw read length.
    # This recovers reads with flanking vector/tag/adapter sequence and correctly
    # splits concatemers/chimeras into their individual domains, instead of
    # discarding the whole read for being the "wrong" total length. Reads with no
    # detectable domain anchors at all are dropped here (see domain_extraction.py).
    extracted_df, extraction_diagnostics = domain_extraction.extract_domains_dataframe(passed_qc)
    io_utils.write_table(pd.DataFrame([extraction_diagnostics]), res_dir / "domain_extraction_report.csv")
    if len(extracted_df):
        # Save per-domain lengths (not just the aggregate summary) so the dashboard
        # can offer an interactive length/tolerance filter over the extracted
        # domains, the same way it does for raw sequences in Sequence Browser --
        # useful for previewing different --expected-length/--length-tolerance
        # choices without re-running the pipeline.
        io_utils.write_table(
            extracted_df[["read_id", "length", "extracted_from", "domain_index", "n_domains_in_read"]],
            res_dir / "extracted_domains.csv",
        )
        qc.plot_length_distribution(
            extracted_df, fig_dir / "extracted_domain_length_dist.png",
            "Extracted domain length distribution (pre length-filter)",
            args.expected_length,
        )

    # 3. Length filtering (now operating on extracted domain sequences)
    min_len, max_len = length_filter.compute_length_window(args.expected_length, args.length_tolerance)
    qc.plot_length_distribution(extracted_df, fig_dir / "length_before_filter.png",
                                 "Length distribution (before length filter)",
                                 args.expected_length, min_len, max_len)
    passed_len, failed_len, len_report = length_filter.filter_by_length(
        extracted_df, args.expected_length, args.length_tolerance
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

    # 8b. Framework mutation / deviation analysis (reference-free: compares every
    # sequence against a repertoire-built consensus for each framework region,
    # not an external reference -- see framework_mutations.py docstring).
    quality_lookup = annotated[["read_id", "mean_qscore"]]
    fw_mutations_df, fw_position_var_df, fw_seq_summary_df = framework_mutations.build_framework_mutation_table(
        cdr_table, quality_lookup=quality_lookup
    )
    io_utils.write_table(fw_mutations_df, res_dir / "framework_mutations.csv")
    io_utils.write_table(fw_position_var_df, res_dir / "framework_position_variability.csv")
    io_utils.write_table(fw_seq_summary_df, res_dir / "framework_mutation_summary.csv")
    if len(fw_position_var_df):
        fig, ax = plt.subplots(figsize=(9, 4))
        colors_map = {"FR1": "#4f81bd", "FR2": "#c0504d", "FR3": "#9bbb59", "FR4": "#8064a2"}
        offset = 0
        xticks, xlabels = [], []
        for region in ["FR1", "FR2", "FR3", "FR4"]:
            sub = fw_position_var_df[fw_position_var_df["region"] == region].sort_values("position")
            xs = sub["position"] + offset
            ax.bar(xs, sub["entropy"], color=colors_map[region], label=region)
            xticks.extend(xs.tolist())
            xlabels.extend(sub["position"].tolist())
            offset += sub["position"].max() + 2 if len(sub) else 0
        ax.set_xlabel("Position (within region, concatenated FR1->FR4)")
        ax.set_ylabel("Shannon entropy (nats)")
        ax.set_title("Framework position variability (repertoire-wide)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "framework_position_variability.png", dpi=300)
        fig.savefig(fig_dir / "framework_position_variability.pdf")
        plt.close(fig)
    fw_summary = {
        "n_total_deviations": len(fw_mutations_df),
        "n_sequences_with_deviation": int((fw_seq_summary_df["n_framework_deviations"] > 0).sum()),
        "n_sequences_with_length_variant_region": int(fw_seq_summary_df["has_length_variant_region"].sum()),
    }

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

    # complete_domain + reconstructed_domain: whether ANARCI/fallback found ALL
    # SEVEN regions (not just CDR3), and the clean concatenated domain sequence
    # from just those regions -- used to gate and source sequences for structure
    # prediction below, since a confident CDR3 alone does not guarantee FR1/FR2
    # weren't lost to a frameshift artifact (see structure_prediction.py).
    _regions = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
    domain_lookup = cdr_table[["read_id"] + _regions].copy()
    domain_lookup["complete_domain"] = domain_lookup[_regions].notna().all(axis=1)
    domain_lookup["reconstructed_domain"] = domain_lookup.apply(
        lambda r: "".join(r[c] for c in _regions) if r["complete_domain"] else None, axis=1
    )
    domain_lookup = domain_lookup.rename(columns={"read_id": "representative_id"})
    read_aux = read_aux.merge(
        domain_lookup[["representative_id", "complete_domain", "reconstructed_domain"]],
        on="representative_id", how="left",
    )

    novel_input = uniq["unique_aa"].merge(read_aux, on="representative_id", how="left")
    novel_flagged = novel_discovery.flag_candidates(
        novel_input,
        cluster_col="cluster" if "cluster" in novel_input.columns else None,
        rare_combo_col="has_rare_cdr_combo",
    )
    io_utils.write_table(novel_flagged, res_dir / "novel_candidates.csv")
    novel_summary = novel_discovery.summarize_novel_candidates(novel_flagged)

    # 19b. Structure prediction on top candidates (Section 19 recommended extension).
    # Uses the CLEAN reconstructed domain (FR1..FR4/CDR1..3 concatenated), not the
    # raw translated ORF, and only among complete_domain==True sequences -- see
    # structure_prediction.select_top_candidates docstring for why.
    struct_dir = outdir / "structures"
    if not args.skip_structure_prediction:
        top_candidates = structure_prediction.select_top_candidates(
            novel_flagged, uniq["unique_aa"], n=args.n_structures,
            protein_col="reconstructed_domain",
        )
        if len(top_candidates):
            struct_results = structure_prediction.predict_structures(
                top_candidates, protein_col="reconstructed_domain", outdir=struct_dir,
            )
            io_utils.write_table(struct_results, res_dir / "structure_predictions.csv")
            struct_summary = {
                "n_attempted": len(top_candidates),
                "n_folded_successfully": int(struct_results["fold_success"].sum()),
            }
        else:
            logger.warning("No candidates available for structure prediction (none passed the quality gate).")
            struct_summary = {"n_attempted": 0, "n_folded_successfully": 0}
    else:
        struct_summary = {}

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
               "domain_extraction": extraction_diagnostics,
               "framework_mutations": fw_summary,
               "structure_prediction": struct_summary},
    )
    report.write_report(final_report, outdir)
    logger.info("Pipeline complete. Outputs in %s", outdir)


if __name__ == "__main__":
    main()
