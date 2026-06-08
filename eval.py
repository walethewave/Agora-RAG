# eval.py
# Agora RAG — Evaluation Dashboard
# Standalone Streamlit app for evaluating RAG answer quality.
# Calls SimplifiedRAG directly in-process — no FastAPI backend required.
# Deploy independently on Streamlit Cloud with eval.py as the entry point.

import streamlit as st
import pandas as pd
import time
import uuid
import io
import yaml
import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.simplified_rag import SimplifiedRAG


#  Load judge prompts from prompts.yaml 
_prompts = yaml.safe_load(
    (pathlib.Path(__file__).parent / "src" / "prompts.yaml").read_text()
)
JUDGE_SYSTEM_PROMPT = _prompts["judge_system_prompt"].strip()
JUDGE_PROMPT_TEMPLATE = _prompts["judge_prompt"].strip()


# -- RAG Initialization (cached -- only runs once per session) ----------------
def get_rag():
    import os
    for key in ["GEMINI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "REDIS_URL"]:
        try:
            os.environ.setdefault(key, st.secrets[key])
        except (KeyError, FileNotFoundError):
            pass
    return SimplifiedRAG()


#  Page Configuration 
st.set_page_config(
    page_title="Agora — Evaluation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom Styling 
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .pass-badge {
        background: #a6e3a1;
        color: #1e1e2e;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .fail-badge {
        background: #f38ba8;
        color: #1e1e2e;
        padding: 2px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #cdd6f4;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #313244;
        padding-bottom: 6px;
    }
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

#  Page Header 
st.markdown("# 📊 Agora Evaluation Dashboard")
st.markdown("Measure your RAG system's answer quality, grounding accuracy, and retrieval precision against a golden dataset.")
st.divider()

#  RAG system 
rag = get_rag()

#  Sidebar 
with st.sidebar:

    # Section 1 — Namespace Config
    st.subheader("⚙️ Configuration")
    entity_id = st.text_input("Namespace (entity_id)", value="policy")

    # Section 2 — Evaluation Settings
    st.subheader("🎯 Evaluation Settings")
    request_delay = st.number_input(
        "Delay Between Questions (s)",
        min_value=0.0, max_value=5.0, value=3.0, step=0.1,
        help="Pause between questions to avoid rate limiting",
    )
    max_workers = st.slider(
        "Parallel Workers",
        min_value=1, max_value=5, value=1, step=1,
        help="How many questions to evaluate simultaneously. Higher = faster but more likely to hit API rate limits. Keep at 1 for Gemini free tier.",
    )

    # Section 3 — System Status
    st.subheader("🔌 System Status")
    if st.button("Check RAG System"):
        try:
            rag.ask_questions("test", str(uuid.uuid4()), entity_id)
            st.success("✅ RAG system is online")
        except Exception as e:
            st.error(f"❌ RAG error: {str(e)[:100]}")

    # Section 4 — About
    with st.expander("About this dashboard", expanded=False):
        st.markdown("""
This dashboard evaluates your Agora RAG system by running a set of test questions 
and comparing answers against expected outputs.

Metrics computed:
- LLM Judge — Gemini compares actual answer against expected answer (YES/NO)
- Source Hit — correct document cited in sources
- Correct Refusal — system refuses out-of-scope questions properly
- Cosine Score — retrieval relevance of returned chunks

Upload a CSV with columns: question, expected_answer, expected_source
        """)


#  MAIN AREA — SECTION 1: CSV Upload 
st.subheader("📁 Upload Golden Dataset")

SAMPLE_CSV = """question,expected_answer,expected_source
What reports are required on AI integration within the intelligence community?,Agencies must submit quarterly reports within 30 days,10.txt
What is the Digital Development Infrastructure Plan?,A national strategy for digital infrastructure development,1.txt
How do different jurisdictions approach AI transparency requirements?,Multiple jurisdictions require disclosure of AI use,
What enforcement mechanisms exist for AI compliance violations?,Regulatory bodies have authority to issue fines and require remediation,3.txt
What does this document say about quantum computing?,SHOULD REFUSE,
Hi what can you help me with?,SHOULD REFUSE,
What working groups exist for AI governance?,working groups committees established oversight,
"""

col_upload, col_format = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Choose your evaluation CSV", type=["csv"])

with col_format:
    with st.expander("📋 CSV Format", expanded=False):
        st.markdown("""
Required columns:

| Column | Description |
|---|---|
| question | The question to ask the RAG system |
| expected_answer | Key facts that should appear in the answer. Write SHOULD REFUSE for out-of-scope questions |
| expected_source | Filename to check in sources (e.g. 10.txt). Leave empty to skip source check |
        """)

    st.download_button(
        label="⬇️ Download Sample CSV",
        data=SAMPLE_CSV.encode(),
        file_name="sample_eval.csv",
        mime="text/csv",
    )


#  MAIN AREA — SECTION 2: Data Preview 
if uploaded_file is not None:
    # Parse and validate the uploaded CSV
    df = pd.read_csv(uploaded_file)
    required_cols = {"question", "expected_answer", "expected_source"}

    if not required_cols.issubset(set(df.columns)):
        st.error(
            "❌ Invalid CSV format. Your file must have exactly these three columns: "
            "`question`, `expected_answer`, `expected_source`. "
            "Download the sample CSV above to use as a template."
        )
        st.stop()

    # Summary metrics
    total_questions = len(df)
    refusal_questions = df["expected_answer"].apply(
        lambda x: str(x).strip().upper() == "SHOULD REFUSE"
    ).sum()
    source_checked = df["expected_source"].apply(
        lambda x: pd.notna(x) and str(x).strip() != ""
    ).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Questions", total_questions)
    c2.metric("Refusal Questions", int(refusal_questions))
    c3.metric("Source-Checked Questions", int(source_checked))

    # Data preview
    with st.expander("Preview uploaded data (first 5 rows)", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

    st.divider()

    #  MAIN AREA — SECTION 3: Run Evaluation 
    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):

        # Step A — Setup
        session_id = str(uuid.uuid4())
        total = len(df)

        #  LLM Judge: Gemini evaluates on 4 dimensions 
        def refusal_judge(actual: str) -> bool:
            """Ask Gemini if the actual answer is a proper refusal -- no hardcoded phrases."""
            prompt = (
                f"Does the following response refuse to answer, decline the request, "
                f"say it lacks information, state the question is out of scope, "
                f"say the documents do not contain sufficient information, "
                f"say it cannot help with the request, or respond without "
                f"providing the requested information? "
                f"Important: a response saying 'The provided documents do not contain "
                f"sufficient information to answer this question' IS a proper refusal. "
                f"Reply with YES or NO only.\n\nRESPONSE: {actual}"
            )
            try:
                response = rag.chat_client.generate_text(
                    system_prompt="You are a strict evaluation judge.",
                    user_message=prompt,
                    max_tokens=50,
                    temperature=0.0,
                )
                return response.strip().upper().startswith("YES")
            except Exception:
                return False

        def llm_judge(expected: str, actual: str) -> tuple:
            """Ask Gemini to evaluate actual answer on 4 YES/NO dimensions.
            Returns (overall_pass: bool, relevance: bool, correctness: bool, completeness: bool, grounding: bool, reasoning: str).
            """
            prompt = JUDGE_PROMPT_TEMPLATE.format(expected=expected, actual=actual)
            try:
                response = rag.chat_client.generate_text(
                    system_prompt=JUDGE_SYSTEM_PROMPT,
                    user_message=prompt,
                    max_tokens=1500,
                    temperature=0.0,
                )
                lines = {}
                for line in response.strip().splitlines():
                    line = line.strip()
                    for key in ["RELEVANCE", "CORRECTNESS", "COMPLETENESS", "GROUNDING", "REASONING"]:
                        if line.upper().startswith(key):
                            _, _, v = line.partition(":")
                            lines[key] = v.strip()
                            break

                def parse(key):
                    return lines.get(key, "NO").upper().startswith("YES")

                relevance = parse("RELEVANCE")
                correctness = parse("CORRECTNESS")
                completeness = parse("COMPLETENESS")
                grounding = parse("GROUNDING")
                overall = relevance and correctness and completeness and grounding
                reasoning = lines.get("REASONING", response.strip()[:100])
                return overall, relevance, correctness, completeness, grounding, reasoning
            except Exception as e:
                return False, False, False, False, False, f"Judge error: {str(e)[:80]}"

        # This function evaluates a single row — runs in a thread
        def evaluate_row(row_data):
            idx, row = row_data
            question = str(row["question"])
            expected_answer = str(row["expected_answer"]).strip().upper() if str(row["expected_answer"]).strip().upper() == "SHOULD REFUSE" else str(row["expected_answer"]).strip()
            expected_source = (
                str(row["expected_source"]).strip()
                if pd.notna(row["expected_source"]) else ""
            )

            start_time = time.time()
            call_failed = False

            try:
                result = rag.ask_questions(
                    question=question,
                    session_id=session_id,
                    namespace=entity_id,
                )
                actual_answer = result["answer"] or ""
                sources = result["sources"] if isinstance(result.get("sources"), list) else []
            except Exception as e:
                actual_answer = f"ERROR: {str(e)}"
                sources = []
                call_failed = True

            latency_seconds = round(time.time() - start_time, 2)

            top_cosine_score = round(max(s["relevance_score"] for s in sources), 3) if sources else 0.0
            source_filenames = ", ".join(s["filename"] for s in sources)
            is_refusal_question = expected_answer.upper() == "SHOULD REFUSE"

            correct_refusal = refusal_judge(actual_answer) if is_refusal_question else False

            if is_refusal_question or call_failed:
                llm_pass, relevance, correctness, completeness, grounding, llm_reasoning = False, False, False, False, False, "N/A"
            else:
                llm_pass, relevance, correctness, completeness, grounding, llm_reasoning = llm_judge(expected_answer, actual_answer)

            source_hit = (expected_source in source_filenames) if expected_source else None

            if call_failed:
                passed = False
            elif is_refusal_question:
                passed = correct_refusal
            elif expected_source:
                passed = bool(llm_pass and source_hit)
            else:
                passed = llm_pass

            return {
                "idx": idx,
                "question": question,
                "expected_answer": expected_answer,
                "expected_source": expected_source,
                "actual_answer": actual_answer,
                "source_filenames": source_filenames,
                "top_cosine_score": top_cosine_score,
                "source_hit": source_hit,
                "llm_pass": llm_pass,
                "relevance": relevance,
                "correctness": correctness,
                "completeness": completeness,
                "grounding": grounding,
                "llm_reasoning": llm_reasoning,
                "correct_refusal": correct_refusal,
                "is_refusal_question": is_refusal_question,
                "latency_seconds": latency_seconds,
                "passed": passed,
                "call_failed": call_failed,
            }

        # Run parallel evaluation
        results_dict = {}
        completed_count = 0

        with st.status(f"Running {total} questions in parallel (max {max_workers} at a time)...", expanded=True) as status:
            progress_bar = st.progress(0)
            counter_text = st.empty()
            results_container = st.empty()

            row_data_list = [(i, row) for i, row in df.iterrows()]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(evaluate_row, rd): rd for rd in row_data_list}

                for future in as_completed(futures):
                    completed_count += 1
                    result = future.result()
                    results_dict[result["idx"]] = result

                    progress_bar.progress(completed_count / total)
                    counter_text.markdown(
                        f"**{completed_count} of {total} complete** "
                        f"— {total - completed_count} remaining"
                    )

                    current_df = pd.DataFrame(list(results_dict.values()))
                    current_df = current_df.sort_values("idx").reset_index(drop=True)
                    pass_so_far = int(current_df["passed"].sum())
                    fail_so_far = completed_count - pass_so_far
                    results_container.markdown(
                        f"**Live results:** ✅ {pass_so_far} passing · ❌ {fail_so_far} failing"
                    )

            # Sort by original order and store
            final_results = [results_dict[k] for k in sorted(results_dict.keys())]
            final_df = pd.DataFrame(final_results).drop(columns=["idx"])

            status.update(
                label=f"✅ Evaluation complete — {total} questions evaluated",
                state="complete"
            )
            st.session_state["eval_results"] = final_df
            st.session_state["eval_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")


#  MAIN AREA — SECTION 4: Results Display 
if st.session_state.get("eval_results") is not None and not st.session_state["eval_results"].empty:
    res = st.session_state["eval_results"]

    st.divider()
    st.subheader("📈 Evaluation Results")
    st.caption(f"Run at: {st.session_state.get('eval_timestamp', '')}")

    #  4A: Summary Metrics 
    total_passed = int(res["passed"].sum())
    total_failed = int((~res["passed"]).sum())
    pass_rate = total_passed / len(res) * 100
    avg_latency = res["latency_seconds"].mean()

    source_rows = res[res["source_hit"].notna()]
    source_hit_rate = f"{source_rows['source_hit'].mean() * 100:.1f}%" if not source_rows.empty else "N/A"

    refusal_rows = res[res["is_refusal_question"] == True]
    refusal_accuracy = (
        f"{refusal_rows['correct_refusal'].mean() * 100:.1f}%"
        if not refusal_rows.empty else "N/A"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("✅ Pass Rate", f"{pass_rate:.1f}%", delta=f"{total_passed} passed / {total_failed} failed")
    m2.metric("📄 Source Hit Rate", source_hit_rate)
    m3.metric("⏱️ Avg Latency", f"{avg_latency:.2f}s")

    if not refusal_rows.empty:
        st.metric("🚫 Refusal Accuracy", refusal_accuracy)

    #  4B: Results Table 
    st.subheader("Per-Question Breakdown")

    fc1, fc2 = st.columns(2)
    filter_failures = fc1.checkbox("Show failures only")
    filter_refusals = fc2.checkbox("Show refusal questions only")

    display_df = res.copy()
    if filter_failures:
        display_df = display_df[display_df["passed"] == False]
    if filter_refusals:
        display_df = display_df[display_df["is_refusal_question"] == True]

    display_df = display_df.copy()
    display_df["question"] = display_df["question"].str[:80]
    display_df["passed"] = display_df["passed"].map({True: "✅ Pass", False: "❌ Fail"})
    display_df["llm_pass"] = display_df["llm_pass"].map({True: "✅ YES", False: "❌ NO"})
    display_df["relevance"] = display_df.apply(
        lambda r: "—" if r["is_refusal_question"] else ("✅" if r["relevance"] else "❌"), axis=1
    )
    display_df["correctness"] = display_df.apply(
        lambda r: ("✅" if r["correct_refusal"] else "❌") if r["is_refusal_question"] else ("✅" if r["correctness"] else "❌"), axis=1
    )
    display_df["completeness"] = display_df.apply(
        lambda r: ("✅" if r["correct_refusal"] else "❌") if r["is_refusal_question"] else ("✅" if r["completeness"] else "❌"), axis=1
    )
    display_df["grounding"] = display_df.apply(
        lambda r: "—" if r["is_refusal_question"] else ("✅" if r["grounding"] else "❌"), axis=1
    )
    display_df["source_hit"] = display_df["source_hit"].apply(
        lambda x: "✅" if x is True else ("❌" if x is False else "—")
    )
    display_df["correct_refusal"] = display_df["correct_refusal"].map({True: "✅", False: "❌"})

    st.dataframe(
        display_df[["question", "passed", "relevance", "correctness", "completeness", "grounding",
                    "source_hit", "latency_seconds", "source_filenames"]].rename(columns={
            "question": "Question",
            "passed": "Result",
            "relevance": "Relevance",
            "correctness": "Correctness",
            "completeness": "Completeness",
            "grounding": "Grounding",
            "source_hit": "Src Hit",
            "latency_seconds": "Latency(s)",
            "source_filenames": "Sources",
        }).reset_index(drop=True),
        use_container_width=True,
        height=400,
    )

    #  4C: Failure Analysis 
    st.subheader("🔍 Failure Analysis")
    failures = res[res["passed"] == False]

    # Benchmark breakdown -- always visible
    non_refusal = res[res["is_refusal_question"] == False]
    refusal = res[res["is_refusal_question"] == True]
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Relevance", f"{non_refusal['relevance'].mean()*100:.0f}%" if not non_refusal.empty else "N/A")
    b2.metric("Correctness", f"{non_refusal['correctness'].mean()*100:.0f}%" if not non_refusal.empty else "N/A")
    b3.metric("Completeness", f"{non_refusal['completeness'].mean()*100:.0f}%" if not non_refusal.empty else "N/A")
    b4.metric("Grounding", f"{non_refusal['grounding'].mean()*100:.0f}%" if not non_refusal.empty else "N/A")
    b5.metric("Refusal Accuracy", f"{refusal['correct_refusal'].mean()*100:.0f}%" if not refusal.empty else "N/A")
    st.divider()

    if failures.empty:
        st.success("🎉 All questions passed!")
    else:
        st.warning(f"{len(failures)} question(s) failed. Review below.")

    for _, row in res.iterrows():
        icon = "✅" if row["passed"] else "❌"
        with st.expander(f"{icon} {str(row['question'])[:60]}"):
            st.markdown(f"**Question:** {row['question']}")
            st.markdown(f"**Expected:** {row['expected_answer']}")
            st.markdown(f"**Actual Answer:** {row['actual_answer']}")
            st.markdown(f"**Sources Returned:** {row['source_filenames'] if row['source_filenames'] else 'None'}")
            st.markdown(f"**Latency:** {row['latency_seconds']}s | **Source Hit:** {row['source_hit']}")
            st.markdown(f"**Judge:** Relevance {row['relevance']} | Correctness {row['correctness']} | Completeness {row['completeness']} | Grounding {row['grounding']}")
            st.markdown(f"**Reasoning:** {row['llm_reasoning']}")

    #  4D: Download 
    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            label="⬇️ Download Full Results (CSV)",
            data=res.to_csv(index=False).encode(),
            file_name="agora_eval_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            label="⬇️ Download Failures Only (CSV)",
            data=failures.to_csv(index=False).encode(),
            file_name="agora_eval_failures.csv",
            mime="text/csv",
            use_container_width=True,
            disabled=failures.empty,
        )


#  MAIN AREA — SECTION 5: Reset 
st.divider()
if st.button("🔄 Clear Results and Start Over"):
    for key in ["eval_results", "eval_timestamp"]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()
