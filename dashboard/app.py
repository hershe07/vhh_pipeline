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
import streamlit.components.v1 as components


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
    struct_pred = load_table(tables_dir / "structure_predictions.csv")
    extraction = load_table(tables_dir / "domain_extraction_report.csv")
    extracted_domains = load_table(tables_dir / "extracted_domains.csv")

    if annotated is None:
        st.error(f"No results found under {tables_dir}. Run scripts/run_pipeline.py first.")
        return

    tabs = st.tabs([
        "Domain Extraction", "Sequence Browser", "CDR Viewer", "CDR Length Combinations",
        "Cluster Explorer", "Diversity Statistics", "Novel Candidates",
        "Structure Predictions", "Downloads",
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

            # -- Interactive length / tolerance preview --
            # Lets you try different --expected-length / --length-tolerance choices
            # against the ACTUAL extracted domain lengths, without re-running the
            # pipeline -- useful for picking values before committing to a full rerun.
            if extracted_domains is not None and len(extracted_domains):
                st.markdown("---")
                st.subheader("Preview: length filter window on extracted domains")
                data_min, data_max = int(extracted_domains["length"].min()), int(extracted_domains["length"].max())

                mode = st.radio(
                    "Filter by", ["Expected length \u00b1 tolerance %", "Min/Max range"],
                    horizontal=True, key="domain_extraction_filter_mode",
                )
                if mode == "Expected length \u00b1 tolerance %":
                    expected = st.number_input("Expected length (bp)", value=397, step=1,
                                                key="domain_extraction_expected_len")
                    tolerance_pct = st.slider("Tolerance (%)", 0, 50, 10, key="domain_extraction_tolerance")
                    delta = expected * tolerance_pct / 100
                    lo, hi = int(round(expected - delta)), int(round(expected + delta))
                    st.caption(f"Window: [{lo}, {hi}] bp")
                else:
                    lo, hi = st.slider(
                        "Domain length (bp)", data_min, data_max, (data_min, data_max),
                        key="domain_extraction_range",
                    )

                in_window = extracted_domains[
                    (extracted_domains["length"] >= lo) & (extracted_domains["length"] <= hi)
                ]
                pct = 100 * len(in_window) / len(extracted_domains) if len(extracted_domains) else 0.0
                st.metric(
                    f"Domains within [{lo}, {hi}] bp",
                    f"{len(in_window)} / {len(extracted_domains)}",
                    f"{pct:.1f}%",
                )
                fig_preview = px.histogram(extracted_domains, x="length", nbins=60,
                                            title="Extracted domain lengths (window shown highlighted)")
                fig_preview.add_vrect(x0=lo, x1=hi, fillcolor="green", opacity=0.15, line_width=0)
                st.plotly_chart(fig_preview, use_container_width=True)
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

    # ---------------- Structure Predictions ----------------
    with tabs[7]:
        st.subheader("Structure predictions (ESMFold, top novel candidates)")
        st.caption(
            "Predicted 3D structures for the top-ranked novel candidates, folded via "
            "the ESM Atlas API on the clean reconstructed VHH domain (FR1-FR4/CDR1-3), "
            "not the raw translated read. Only candidates with all seven regions "
            "confidently annotated are eligible -- see structure_prediction.py."
        )
        if struct_pred is not None and len(struct_pred):
            n_ok = int(struct_pred["fold_success"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Attempted", len(struct_pred))
            c2.metric("Folded successfully", n_ok)
            mean_plddt_overall = struct_pred.loc[struct_pred["fold_success"], "mean_plddt"].mean()
            c3.metric("Mean pLDDT (successful)", f"{mean_plddt_overall:.1f}" if n_ok else "n/a")

            ok_df = struct_pred[struct_pred["fold_success"]].sort_values("mean_plddt", ascending=False)
            if len(ok_df):
                fig = px.bar(
                    ok_df, x="representative_id", y="mean_plddt", color="confidence_tier",
                    title="Mean pLDDT per folded candidate (higher = more confident)",
                    color_discrete_map={
                        "very_high": "#2c7a2c", "confident": "#4f81bd",
                        "low": "#e0a530", "very_low": "#c0504d",
                    },
                )
                fig.add_hline(y=70, line_dash="dot", annotation_text="pLDDT=70 (confident threshold)")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                st.subheader("3D structure viewer")
                chosen_id = st.selectbox("Select a candidate to view", ok_df["representative_id"].tolist())
                chosen_row = ok_df[ok_df["representative_id"] == chosen_id].iloc[0]
                pdb_path = Path(chosen_row["pdb_path"]) if chosen_row["pdb_path"] else None
                if pdb_path and pdb_path.exists():
                    pdb_text = pdb_path.read_text()
                    st.caption(f"pLDDT: {chosen_row['mean_plddt']:.1f} ({chosen_row['confidence_tier']})")
                    viewer_html = f"""
                    <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
                    <div id="viewer" style="height: 450px; width: 100%; position: relative;"></div>
                    <script>
                    let viewer = $3Dmol.createViewer("viewer", {{backgroundColor: "white"}});
                    let pdbData = `{pdb_text}`;
                    viewer.addModel(pdbData, "pdb");
                    viewer.setStyle({{}}, {{cartoon: {{colorscheme: "bfactor"}}}});
                    viewer.zoomTo();
                    viewer.render();
                    </script>
                    """
                    components.html(viewer_html, height=470)
                    st.download_button(
                        "Download this PDB file", data=pdb_text,
                        file_name=f"{chosen_id}.pdb",
                    )
                else:
                    st.info("PDB file not found on disk for this candidate.")

            failed = struct_pred[~struct_pred["fold_success"]]
            if len(failed):
                st.markdown("---")
                st.caption(f"{len(failed)} candidate(s) failed to fold (network/API error) -- see table below.")
                st.dataframe(failed, use_container_width=True)
        else:
            st.info(
                "structure_predictions.csv not found. Either the pipeline was run with "
                "--skip-structure-prediction, or no candidates passed the complete-domain "
                "quality gate. Rerun scripts/run_pipeline.py without that flag to generate it."
            )

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
