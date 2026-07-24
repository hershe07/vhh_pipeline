#!/usr/bin/env python3
"""
dashboard/app.py
------------------
Section 16: interactive Streamlit dashboard over the tables produced by
scripts/run_pipeline.py.

Run with:
    streamlit run dashboard/app.py -- --results results/

Streamlit was chosen over Dash for this brief because the deliverable is
primarily "browse tables / filter / download", which Streamlit's widget
model handles with far less boilerplate; Dash is preferable if you later
need fully custom multi-page callback graphs or tighter layout control.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


def parse_args():
    # Streamlit passes args after `--`; argparse handles that fine.
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/", help="Path to the pipeline's --outdir")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


@st.cache_data
def load_table(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    st.set_page_config(page_title="VHH Repertoire Dashboard", layout="wide")
    args = parse_args()
    results_dir = Path(args.results)
    tables_dir = results_dir / "tables"

    st.title("Nanobody (VHH) Repertoire Dashboard")
    st.caption(f"Data source: {results_dir.resolve()}")

    annotated = load_table(tables_dir / "annotated_sequences.csv")
    cdr_table = load_table(tables_dir / "cdr_table.csv")
    combo_df = load_table(tables_dir / "cdr_length_combinations.csv")
    unique_aa = load_table(tables_dir / "unique_aa_abundance.csv")
    novel = load_table(tables_dir / "novel_candidates.csv")
    cdr_div = load_table(tables_dir / "cdr_diversity.csv")
    dev = load_table(tables_dir / "developability_screen.csv")
    extraction = load_table(tables_dir / "domain_extraction_report.csv")

    if annotated is None:
        st.error(f"No results found under {tables_dir}. Run scripts/run_pipeline.py first.")
        return

    tabs = st.tabs([
        "Domain Extraction", "Sequence Browser", "CDR Viewer", "CDR Length Combinations",
        "Cluster Explorer", "Diversity Statistics", "Novel Candidates",
        "Developability", "Downloads",
    ])

    # ---------------- Domain Extraction ----------------
    with tabs[0]:
        st.subheader("Domain extraction diagnostics")
        st.caption(
            "Shows how raw reads were handled before length filtering: reads with no "
            "detectable VHH domain (dropped), reads with exactly one domain, and "
            "reads with multiple domains (concatemers/chimeras, each domain extracted "
            "as its own sequence)."
        )
        if extraction is not None and len(extraction):
            row = extraction.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Input reads", int(row["n_input_reads"]))
            c2.metric("No domain detected", int(row["n_reads_no_domain_detected"]))
            c3.metric("Single-domain reads", int(row["n_reads_single_domain"]))
            c4.metric("Multi-domain reads", int(row["n_reads_multi_domain"]))

            st.metric("Total domains extracted", int(row["n_domains_extracted_total"]))

            call_counts = pd.DataFrame({
                "call": ["no_domain", "single_domain", "multi_domain"],
                "count": [
                    int(row["n_reads_no_domain_detected"]),
                    int(row["n_reads_single_domain"]),
                    int(row["n_reads_multi_domain"]),
                ],
            })
            fig = px.bar(call_counts, x="call", y="count", title="Raw read domain-content breakdown")
            st.plotly_chart(fig, use_container_width=True)

            fig_path = results_dir / "figures" / "extracted_domain_length_dist.png"
            if fig_path.exists():
                st.image(str(fig_path), caption="Extracted domain length distribution (pre length-filter)")
        else:
            st.info("domain_extraction_report.csv not found. Rerun scripts/run_pipeline.py to generate it.")

    # ---------------- Sequence Browser ----------------
    with tabs[1]:
        st.subheader("Sequence browser")
        search = st.text_input("Search by sequence ID or (sub)sequence")
        view = annotated

        # -- Length filters --
        if "length" in annotated.columns and annotated["length"].notna().any():
            with st.expander("Length filters", expanded=False):
                mode = st.radio(
                    "Filter by",
                    ["Min/Max range", "Expected length \u00b1 tolerance %"],
                    horizontal=True,
                )
                data_min, data_max = int(annotated["length"].min()), int(annotated["length"].max())

                if mode == "Min/Max range":
                    lo, hi = st.slider(
                        "Read length (bp)", data_min, data_max, (data_min, data_max)
                    )
                else:
                    expected = st.number_input("Expected length (bp)", value=397, step=1)
                    tolerance_pct = st.slider("Tolerance (%)", 0, 50, 10)
                    delta = expected * tolerance_pct / 100
                    lo, hi = int(round(expected - delta)), int(round(expected + delta))
                    st.caption(f"Window: [{lo}, {hi}] bp")

                view = view[(view["length"] >= lo) & (view["length"] <= hi)]

        if search:
            mask = (
                view.get("read_id", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
                | view.get("protein", pd.Series(dtype=str)).astype(str).str.contains(search, case=False, na=False)
            )
            view = view[mask]
        st.write(f"{len(view)} sequences")
        st.dataframe(view, use_container_width=True, height=500)

    # ---------------- CDR Viewer ----------------
    with tabs[2]:
        st.subheader("CDR viewer & length filters")
        if cdr_table is not None:
            c1, c2, c3 = st.columns(3)
            len_ranges = {}
            for col, c in zip(["CDR1_length", "CDR2_length", "CDR3_length"], (c1, c2, c3)):
                if col in cdr_table.columns and cdr_table[col].notna().any():
                    lo, hi = int(cdr_table[col].min()), int(cdr_table[col].max())
                    len_ranges[col] = c.slider(col, lo, hi, (lo, hi))
            filtered = cdr_table.copy()
            for col, (lo, hi) in len_ranges.items():
                filtered = filtered[(filtered[col] >= lo) & (filtered[col] <= hi)]
            st.write(f"{len(filtered)} sequences match filters")
            st.dataframe(filtered, use_container_width=True, height=450)

            for cdr in ("CDR1", "CDR2", "CDR3"):
                col = f"{cdr}_length"
                if col in filtered.columns and filtered[col].notna().any():
                    fig = px.histogram(filtered, x=col, nbins=30, title=f"{cdr} length distribution")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("cdr_table.csv not found.")

    # ---------------- CDR Length Combinations ----------------
    with tabs[3]:
        st.subheader("CDR1-CDR2-CDR3 length combinations")
        if combo_df is not None:
            # Force 'combination' to string dtype -- Plotly's client-side type
            # inference otherwise treats number-separated labels (e.g. "8-7-13")
            # as dates and renders month-abbreviation tick labels (Aug, Sep, ...)
            # instead of the actual CDR length combination.
            combo_df = combo_df.copy()
            combo_df["combination"] = combo_df["combination"].astype(str)

            top_n = st.slider("Show top N combinations", 5, min(100, len(combo_df)), min(30, len(combo_df)))
            fig = px.bar(combo_df.head(top_n), x="combination", y="count",
                         color="is_rare" if "is_rare" in combo_df.columns else None,
                         title="Top CDR length combinations")
            fig.update_xaxes(type="category")  # belt-and-suspenders: never auto-detect as a date axis
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(combo_df, use_container_width=True, height=400)
        else:
            st.info("cdr_length_combinations.csv not found.")

    # ---------------- Cluster Explorer ----------------
    with tabs[4]:
        st.subheader("Cluster explorer")
        if unique_aa is not None and "cluster" in unique_aa.columns:
            cluster_sizes = unique_aa["cluster"].value_counts().reset_index()
            cluster_sizes.columns = ["cluster", "size"]
            fig = px.bar(cluster_sizes.sort_values("size", ascending=False), x="cluster", y="size",
                         title="Cluster sizes (-1 = outlier / unclustered)")
            st.plotly_chart(fig, use_container_width=True)
            chosen = st.selectbox("Inspect cluster", sorted(unique_aa["cluster"].unique()))
            st.dataframe(unique_aa[unique_aa["cluster"] == chosen], use_container_width=True)
        else:
            st.info("No cluster assignments found in unique_aa_abundance.csv.")

    # ---------------- Diversity Statistics ----------------
    with tabs[5]:
        st.subheader("Diversity statistics")
        if cdr_div is not None:
            st.dataframe(cdr_div, use_container_width=True)
            fig = px.bar(cdr_div, x="label", y="shannon_entropy", title="Shannon entropy by CDR")
            st.plotly_chart(fig, use_container_width=True)
        if unique_aa is not None:
            fig2 = px.scatter(unique_aa.reset_index(), x="rank", y="relative_abundance",
                               log_x=True, log_y=True, title="Rank-abundance curve (amino acid level)")
            st.plotly_chart(fig2, use_container_width=True)

    # ---------------- Novel Candidates ----------------
    with tabs[6]:
        st.subheader("Novel / rare / outlier candidates")
        if novel is not None:
            min_score = st.slider("Minimum novelty score", 0.0, 1.0, 0.5, 0.05)
            view = novel[novel["novelty_score"] >= min_score].sort_values("novelty_score", ascending=False)
            st.write(f"{len(view)} candidates at or above threshold")
            st.dataframe(view, use_container_width=True, height=450)
        else:
            st.info("novel_candidates.csv not found.")

    # ---------------- Developability ----------------
    with tabs[7]:
        st.subheader("Developability screening")
        if dev is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Aggregation-prone hits", int((dev["n_aggregation_prone_regions"] > 0).sum()))
            c2.metric("Odd cysteine count", int(dev["cysteine_count_odd"].sum()))
            c3.metric("Flagged unstable", int(dev["is_unstable"].fillna(False).sum()))
            c4.metric("Glycosylation sites", int((dev["n_glycosylation_sites"] > 0).sum()))

            only_flagged = st.checkbox("Show only sequences with at least one liability flag", value=False)
            view = dev
            if only_flagged:
                view = dev[
                    (dev["n_aggregation_prone_regions"] > 0)
                    | (dev["cysteine_count_odd"])
                    | (dev["is_unstable"].fillna(False))
                    | (dev["n_glycosylation_sites"] > 0)
                ]
            st.write(f"{len(view)} sequences")
            st.dataframe(view, use_container_width=True, height=400)

            if dev["isoelectric_point"].notna().any():
                fig = px.scatter(
                    dev, x="isoelectric_point", y="gravy",
                    color="cysteine_count_odd" if "cysteine_count_odd" in dev.columns else None,
                    hover_data=["representative_id", "n_aggregation_prone_regions"],
                    title="pI vs GRAVY (hydropathy), colored by odd-cysteine flag",
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(
                    "Physicochemical columns (pI, GRAVY, etc.) are empty -- Biopython's "
                    "ProtParam may not be available in this environment."
                )

            # Cross-reference with Novel Candidates: your practical shortlist is
            # sequences that are BOTH interesting (novel) and clean (no liability flags).
            if novel is not None and "candidate_novel" in novel.columns and "representative_id" in dev.columns:
                st.markdown("---")
                st.subheader("Shortlist: novel candidates with no developability flags")
                merged = novel.merge(
                    dev[["representative_id", "n_aggregation_prone_regions", "cysteine_count_odd", "is_unstable"]],
                    on="representative_id", how="left",
                )
                shortlist = merged[
                    merged["candidate_novel"]
                    & (merged["n_aggregation_prone_regions"] == 0)
                    & (~merged["cysteine_count_odd"].fillna(False))
                    & (~merged["is_unstable"].fillna(False))
                ]
                st.write(f"{len(shortlist)} candidates are both novel and free of flagged liabilities")
                st.dataframe(shortlist, use_container_width=True, height=350)
        else:
            st.info("developability_screen.csv not found.")

    # ---------------- Downloads ----------------
    with tabs[8]:
        st.subheader("Downloadable tables & figures")
        for csv_path in sorted(tables_dir.glob("*.csv")):
            st.download_button(
                label=f"Download {csv_path.name}",
                data=csv_path.read_bytes(),
                file_name=csv_path.name,
            )
        fig_dir = results_dir / "figures"
        if fig_dir.exists():
            st.write("Figures:")
            for fig_path in sorted(fig_dir.glob("*.png")):
                st.image(str(fig_path), caption=fig_path.name, width=350)


if __name__ == "__main__":
    main()
