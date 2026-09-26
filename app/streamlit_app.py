import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# Make project root importable
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from src.rag import ask
from src.multimodal import ask_multimodal


RESULTS_DIR = ROOT / "results"


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="University Handbook RAG",
    page_icon="📚",
    layout="wide",
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("📚 University Handbook RAG Assistant")

st.markdown(
    """
    **Generative AI End-Term Project**

    Compare local RAG responses using **SmolLM2-135M**
    and **Llama 3.2-1B-Instruct (Q4_K_M)** over the
    Vijaybhoomi University Research Incubation Handbook.
    """
)


# --------------------------------------------------
# TABS
# --------------------------------------------------

chat_tab, evaluation_tab, multimodal_tab = st.tabs(
    [
        "💬 RAG Chat",
        "📊 Evaluation",
        "🖼️ Multimodal RAG",
    ]
)


# ==================================================
# TAB 1 — RAG CHAT
# ==================================================

with chat_tab:

    st.subheader("Ask the University Handbook")

    model_display = st.selectbox(
        "Select Model",
        [
            "SmolLM2-135M",
            "Llama 3.2-1B Q4_K_M",
        ],
    )

    model_key = (
        "smollm"
        if model_display == "SmolLM2-135M"
        else "llama"
    )

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What is the purpose of the "
            "Research Incubation program?"
        ),
        height=100,
    )

    ask_button = st.button(
        "🔍 Ask Question",
        type="primary",
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                f"Generating answer using {model_display}..."
            ):

                try:

                    result = ask(
                        question,
                        model_key=model_key,
                        top_k=3,
                    )

                    st.success("Answer generated successfully.")

                    # ------------------------------
                    # ANSWER
                    # ------------------------------

                    st.subheader("Answer")

                    st.write(
                        result["answer"]
                    )

                    # ------------------------------
                    # METRICS
                    # ------------------------------

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Model",
                            model_display,
                        )

                    with col2:
                        st.metric(
                            "Latency",
                            f"{result['latency_seconds']} s",
                        )

                    with col3:
                        pages = [
                            x["page"]
                            for x in result["contexts"]
                        ]

                        st.metric(
                            "Retrieved Pages",
                            str(pages),
                        )

                    # ------------------------------
                    # CONTEXT
                    # ------------------------------

                    st.subheader(
                        "📖 Retrieved Handbook Context"
                    )

                    for context in result["contexts"]:

                        page = context["page"]

                        with st.expander(
                            f"Handbook Page {page}"
                        ):

                            st.write(
                                context["text"]
                            )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )


# ==================================================
# TAB 2 — EVALUATION
# ==================================================

with evaluation_tab:

    st.subheader(
        "📊 RAG Evaluation & Model Comparison"
    )

    smollm_file = (
        RESULTS_DIR /
        "smollm_evaluation.csv"
    )

    llama_file = (
        RESULTS_DIR /
        "llama_evaluation.csv"
    )

    if not smollm_file.exists() or not llama_file.exists():

        st.warning(
            "Evaluation results have not been generated yet."
        )

    else:

        smollm_df = pd.read_csv(
            smollm_file
        )

        llama_df = pd.read_csv(
            llama_file
        )

        # ------------------------------
        # SUMMARY
        # ------------------------------

        def calculate_summary(df, model_name):

            return {
                "Model": model_name,

                "Context Precision":
                    df["context_precision"].mean(),

                "Context Recall":
                    df["context_recall"].mean(),

                "MRR":
                    df["mrr"].mean(),

                "Answer Similarity":
                    df["answer_similarity"].mean(),

                "Keyword Coverage":
                    df["keyword_coverage"].mean(),

                "Groundedness":
                    df["groundedness_proxy"].mean(),

                "Answer Relevancy":
                    df["answer_relevancy_proxy"].mean(),

                "Avg Latency (s)":
                    df["latency_seconds"].mean(),
            }


        smollm_summary = calculate_summary(
            smollm_df,
            "SmolLM2-135M",
        )

        llama_summary = calculate_summary(
            llama_df,
            "Llama 3.2-1B Q4_K_M",
        )

        comparison = pd.DataFrame(
            [
                smollm_summary,
                llama_summary,
            ]
        )

        st.markdown(
            """
            The same handbook, embedding model, vector database,
            retrieval configuration, and evaluation questions are
            used for both language models.
            """
        )

        st.dataframe(
            comparison.style.format(
                {
                    "Context Precision": "{:.3f}",
                    "Context Recall": "{:.3f}",
                    "MRR": "{:.3f}",
                    "Answer Similarity": "{:.3f}",
                    "Keyword Coverage": "{:.3f}",
                    "Groundedness": "{:.3f}",
                    "Answer Relevancy": "{:.3f}",
                    "Avg Latency (s)": "{:.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        # ------------------------------
        # METRIC CHART
        # ------------------------------

        st.subheader(
            "Metric Comparison"
        )

        metric_columns = [
            "Context Precision",
            "Context Recall",
            "MRR",
            "Answer Similarity",
            "Keyword Coverage",
            "Groundedness",
            "Answer Relevancy",
        ]

        chart_df = comparison[
            ["Model"] + metric_columns
        ].set_index("Model")

        st.bar_chart(
            chart_df
        )

        # ------------------------------
        # LATENCY
        # ------------------------------

        st.subheader(
            "Average Generation Latency"
        )

        latency_df = comparison[
            ["Model", "Avg Latency (s)"]
        ].set_index("Model")

        st.bar_chart(
            latency_df
        )

        # ------------------------------
        # INDIVIDUAL RESULTS
        # ------------------------------

        st.subheader(
            "Question-Level Results"
        )

        model_choice = st.selectbox(
            "View results for",
            [
                "SmolLM2-135M",
                "Llama 3.2-1B Q4_K_M",
            ],
        )

        selected_df = (
            smollm_df
            if model_choice == "SmolLM2-135M"
            else llama_df
        )

        display_columns = [
            "id",
            "question",
            "retrieved_pages",
            "context_precision",
            "context_recall",
            "mrr",
            "answer_similarity",
            "groundedness_proxy",
            "latency_seconds",
        ]

        st.dataframe(
            selected_df[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


# ==================================================
# TAB 3 — MULTIMODAL RAG
# ==================================================

with multimodal_tab:

    st.subheader(
        "🖼️ Multimodal RAG"
    )

    st.markdown(
        """
        This pipeline extends the text-only RAG system by providing
        the retrieved handbook text **and corresponding page images**
        to **SmolVLM-256M-Instruct**.
        """
    )

    mm_question = st.text_area(
        "Enter your multimodal question",
        placeholder=(
            "Example: What grades and grade points are "
            "specified for the Research Incubation course?"
        ),
        height=100,
        key="multimodal_question",
    )

    mm_button = st.button(
        "🖼️ Run Multimodal RAG",
        type="primary",
    )

    if mm_button:

        if not mm_question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Retrieving text + images and generating answer with SmolVLM..."
            ):

                try:

                    result = ask_multimodal(
                        mm_question,
                        top_k=3,
                    )

                    st.success(
                        "Multimodal answer generated."
                    )

                    # ------------------------------
                    # ANSWER
                    # ------------------------------

                    st.subheader("Answer")

                    st.write(
                        result["answer"]
                    )

                    # ------------------------------
                    # METRICS
                    # ------------------------------

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Latency",
                            f"{result['latency_seconds']} s",
                        )

                    with col2:

                        st.metric(
                            "Retrieved Pages",
                            str(result["pages"]),
                        )

                    # ------------------------------
                    # RETRIEVED IMAGES
                    # ------------------------------

                    st.subheader(
                        "🖼️ Retrieved Handbook Pages"
                    )

                    image_paths = result.get(
                        "image_paths",
                        []
                    )

                    if image_paths:

                        image_cols = st.columns(
                            len(image_paths)
                        )

                        for i, image_path in enumerate(
                            image_paths
                        ):

                            with image_cols[i]:

                                st.caption(
                                    f"Retrieved Page "
                                    f"{result['pages'][i]}"
                                )

                                if Path(
                                    image_path
                                ).exists():

                                    st.image(
                                        image_path,
                                        use_container_width=True,
                                    )

                    # ------------------------------
                    # RETRIEVED TEXT
                    # ------------------------------

                    st.subheader(
                        "📖 Retrieved Text Context"
                    )

                    for context in result["contexts"]:

                        page = context["page"]

                        with st.expander(
                            f"Handbook Page {page}"
                        ):

                            st.write(
                                context["text"]
                            )

                except Exception as e:

                    st.error(
                        f"Multimodal RAG error: {e}"
                    )

