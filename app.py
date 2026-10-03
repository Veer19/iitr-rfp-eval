import json
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from engine import (
    DEFAULT_CRITERIA,
    evaluate_batch,
    export_csv,
    init_database,
    load_run_history,
    save_criteria,
)

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "data" / "rfp_review.db"

st.set_page_config(
    page_title="RFP Review Studio",
    page_icon="📑",
    layout="wide",
)

init_database(DB_PATH)

st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
.metric-card {
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 12px;
    padding: 16px;
}
.small-muted {color: #777; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

if "criteria" not in st.session_state:
    st.session_state.criteria = DEFAULT_CRITERIA.copy()
if "result" not in st.session_state:
    st.session_state.result = None

st.title("RFP Review Studio")
st.caption("Evidence-first proposal analysis, deterministic scoring, peer benchmarking and risk review.")

with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input(
        "OpenRouter API key",
        type="password",
        value=st.session_state.get("api_key", ""),
        help="Used only for the current Streamlit session unless you save it elsewhere.",
    )
    if api_key:
        st.session_state.api_key = api_key

    model = st.text_input(
        "Model",
        value="openai/gpt-4o-mini",
        help="Any OpenRouter model that supports structured JSON output can be used.",
    )

    st.divider()
    st.subheader("Scoring criteria")
    st.caption("Weights should normally total 100%.")

    edited = []
    for idx, item in enumerate(st.session_state.criteria):
        with st.expander(f"{idx+1}. {item['name']}", expanded=False):
            name = st.text_input("Name", item["name"], key=f"crit_name_{idx}")
            desc = st.text_area("Description", item["description"], key=f"crit_desc_{idx}")
            weight = st.number_input(
                "Weight (%)", min_value=0.0, max_value=100.0,
                value=float(item["weight"]), step=5.0, key=f"crit_weight_{idx}"
            )
            max_score = st.number_input(
                "Maximum score", min_value=1.0, max_value=100.0,
                value=float(item["max_score"]), step=1.0, key=f"crit_max_{idx}"
            )
            edited.append({
                "criterion_id": idx + 1,
                "name": name,
                "description": desc,
                "weight": weight,
                "max_score": max_score,
            })

    if st.button("Save criteria", use_container_width=True):
        total = sum(x["weight"] for x in edited)
        if abs(total - 100.0) > 0.001:
            st.error(f"Weights currently total {total:.1f}%. Set them to 100%.")
        else:
            st.session_state.criteria = edited
            save_criteria(DB_PATH, edited)
            st.success("Criteria saved.")

tab_run, tab_history, tab_criteria = st.tabs(["New evaluation", "Run history", "Criteria"])

with tab_run:
    st.subheader("Evaluate supplier proposals")

    files = st.file_uploader(
        "Upload supplier proposals",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more supplier proposal PDFs.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Suppliers loaded", len(files))
    with c2:
        st.metric("Active criteria", len(st.session_state.criteria))
    with c3:
        st.metric("Weight total", f"{sum(x['weight'] for x in st.session_state.criteria):.1f}%")

    if files:
        meta = []
        for f in files:
            meta.append({
                "Supplier": Path(f.name).stem.replace("_", " "),
                "File": f.name,
                "Submission date": date.today().isoformat(),
            })
        st.dataframe(pd.DataFrame(meta), use_container_width=True, hide_index=True)

        if st.button("Run evaluation", type="primary", use_container_width=True):
            if not api_key:
                st.error("Enter an OpenRouter API key in the sidebar first.")
            elif abs(sum(x["weight"] for x in st.session_state.criteria) - 100.0) > 0.001:
                st.error("Criterion weights must total 100%.")
            else:
                with st.status("Running proposal review...", expanded=True) as status:
                    try:
                        result = evaluate_batch(
                            files=files,
                            criteria=st.session_state.criteria,
                            api_key=api_key,
                            model=model,
                            db_path=DB_PATH,
                            progress_callback=lambda msg: st.write(msg),
                        )
                        st.session_state.result = result
                        status.update(label="Evaluation complete", state="complete")
                    except Exception as exc:
                        status.update(label="Evaluation failed", state="error")
                        st.exception(exc)

    result = st.session_state.result
    if result:
        st.divider()
        st.subheader("Evaluation results")

        leaderboard = pd.DataFrame(result["leaderboard"])
        st.dataframe(
            leaderboard.rename(columns={
                "rank": "Rank",
                "supplier_name": "Supplier",
                "absolute_score": "Absolute score",
                "ppi": "PPI (%)",
                "experience_rating": "Experience",
                "submission_date": "Submission date",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = export_csv(result)
        st.download_button(
            "Download leaderboard CSV",
            data=csv_bytes,
            file_name=f"{result['run_id']}_leaderboard.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download complete JSON",
            data=json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name=f"{result['run_id']}_evaluation.json",
            mime="application/json",
        )

        st.divider()
        st.subheader("Supplier scorecards")

        for card in result["scorecards"]:
            with st.expander(
                f"#{card['rank']} — {card['supplier_name']} | "
                f"Score {card['absolute_score']:.2f} | PPI {card['ppi']:.2f}%"
            ):
                m1, m2, m3 = st.columns(3)
                m1.metric("Absolute score", f"{card['absolute_score']:.2f}/100")
                m2.metric("Peer performance", f"{card['ppi']:.2f}%")
                m3.metric("Experience", f"{card['experience_rating']:.1f}/10")

                st.markdown("**Criterion breakdown**")
                rows = []
                for x in card["criteria"]:
                    rows.append({
                        "Criterion": x["criterion_name"],
                        "Score": f"{x['score']:.2f}/{x['max_score']:.2f}",
                        "Weight": f"{x['weight']:.1f}%",
                        "Contribution": x["weighted_contribution"],
                        "Benchmark": x["benchmark"],
                        "Gap": x["gap"],
                        "Relative": f"{x['relative_performance']:.2f}%",
                        "Evidence": x["evidence_quality"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                st.markdown("**Assessment details**")
                for x in card["criteria"]:
                    st.markdown(f"**{x['criterion_name']} — {x['score']:.2f}/{x['max_score']:.2f}**")
                    st.write(x["justification"])
                    st.caption(f"Evidence: {x['evidence']}")

                st.markdown("**Risks / missing information**")
                if card["risks"]:
                    for risk in card["risks"]:
                        st.warning(risk)
                else:
                    st.success("No proposal risks were returned by the model.")

                st.markdown("**Risk radar**")
                radar = next(
                    (x for x in result["risk_radar"] if x["supplier_name"] == card["supplier_name"]),
                    None,
                )
                if radar and radar["risks"]:
                    radar_df = pd.DataFrame(radar["risks"])
                    st.dataframe(radar_df.rename(columns={"risk": "Risk", "severity": "Severity"}),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("No classified risks.")

                st.markdown("**Overall summary**")
                st.write(card["overall_summary"])

                if card["validation_warnings"]:
                    st.markdown("**Validation warnings**")
                    for warning in card["validation_warnings"]:
                        st.warning(warning)

        with st.expander("Peer benchmarks"):
            benchmark_rows = []
            for criterion in result["criteria"]:
                cid = criterion["criterion_id"]
                benchmark_rows.append({
                    "Criterion": criterion["name"],
                    "Benchmark score": result["peer_benchmarks"].get(str(cid), result["peer_benchmarks"].get(cid, 0)),
                })
            st.dataframe(pd.DataFrame(benchmark_rows), use_container_width=True, hide_index=True)

        if result["warnings"]:
            with st.expander("Run warnings"):
                for warning in result["warnings"]:
                    st.warning(warning)

with tab_history:
    st.subheader("Saved evaluation runs")
    history = load_run_history(DB_PATH)
    if history:
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)
    else:
        st.info("No completed runs yet.")

with tab_criteria:
    st.subheader("Current evaluation framework")
    st.dataframe(
        pd.DataFrame(st.session_state.criteria),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("The scoring engine calculates weighted scores outside the LLM so arithmetic remains deterministic.")
