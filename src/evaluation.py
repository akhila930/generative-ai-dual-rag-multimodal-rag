import json
import re
from pathlib import Path
from difflib import SequenceMatcher

from .config import QUESTIONS_JSON, RESULTS_DIR
from .rag import ask


STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "in", "on",
    "for", "with", "what", "which", "who", "how", "when", "where",
    "does", "do", "be", "this", "that", "their", "its", "as", "from",
    "by", "or", "at", "into", "will", "can", "at", "about"
}


def tokens(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def token_set(text):
    return set(tokens(text))


# -----------------------------
# RETRIEVAL METRICS
# -----------------------------

def context_precision(retrieved, gold_pages):
    """
    Precision of retrieved pages.
    Measures how many retrieved pages are actually relevant.
    """
    if not retrieved:
        return 0.0

    relevant = [
        int(x["page"]) in set(gold_pages)
        for x in retrieved
    ]

    return round(
        sum(relevant) / len(relevant),
        4
    )


def context_recall(retrieved, gold_pages):
    """
    Recall of relevant handbook pages.
    """
    if not gold_pages:
        return 0.0

    retrieved_pages = {
        int(x["page"])
        for x in retrieved
    }

    gold = set(gold_pages)

    return round(
        len(retrieved_pages & gold) / len(gold),
        4
    )


def reciprocal_rank(retrieved, gold_pages):
    """
    Mean Reciprocal Rank component.
    Finds the rank of the first relevant page.
    """
    gold = set(gold_pages)

    for rank, item in enumerate(retrieved, start=1):
        if int(item["page"]) in gold:
            return round(1 / rank, 4)

    return 0.0


# -----------------------------
# ANSWER METRICS
# -----------------------------

def answer_similarity(answer, reference):
    """
    Text similarity between generated answer
    and the expected/reference answer.
    """
    if not answer or not reference:
        return 0.0

    return round(
        SequenceMatcher(
            None,
            answer.lower().strip(),
            reference.lower().strip()
        ).ratio(),
        4
    )


def keyword_coverage(answer, reference):
    """
    Measures how many important reference keywords
    appear in the generated answer.
    """
    ref_tokens = token_set(reference)
    ans_tokens = token_set(answer)

    if not ref_tokens:
        return 0.0

    return round(
        len(ref_tokens & ans_tokens) / len(ref_tokens),
        4
    )


def groundedness(answer, retrieved):
    """
    Lightweight groundedness proxy.

    Checks what fraction of answer keywords
    can be found in the retrieved handbook context.

    This is a proxy, not a full LLM-based faithfulness metric.
    """
    if not answer or not retrieved:
        return 0.0

    context = " ".join(
        item["text"]
        for item in retrieved
    )

    answer_tokens = token_set(answer)
    context_tokens = token_set(context)

    if not answer_tokens:
        return 0.0

    return round(
        len(answer_tokens & context_tokens)
        / len(answer_tokens),
        4
    )


def answer_relevancy(answer, question):
    """
    Measures overlap between question concepts
    and generated answer concepts.
    """
    q_tokens = token_set(question)
    a_tokens = token_set(answer)

    if not q_tokens:
        return 0.0

    return round(
        len(q_tokens & a_tokens) / len(q_tokens),
        4
    )


# -----------------------------
# EVALUATION
# -----------------------------

def evaluate_answer(answer, reference, question, contexts):

    similarity = answer_similarity(
        answer,
        reference
    )

    coverage = keyword_coverage(
        answer,
        reference
    )

    grounded = groundedness(
        answer,
        contexts
    )

    relevancy = answer_relevancy(
        answer,
        question
    )

    # Combined answer quality proxy
    correctness = round(
        (similarity + coverage) / 2,
        4
    )

    return {
        "answer_similarity": similarity,
        "keyword_coverage": coverage,
        "answer_correctness_proxy": correctness,
        "answer_relevancy_proxy": relevancy,
        "groundedness_proxy": grounded,
    }


def run_evaluation(model_key):

    questions = json.loads(
        QUESTIONS_JSON.read_text(
            encoding="utf-8"
        )
    )

    rows = []

    for index, item in enumerate(questions, start=1):

        q = item["question"]

        gold_pages = item.get(
            "gold_pages",
            []
        )

        reference = item.get(
            "reference_answer",
            ""
        )

        print(
            f"\n[{index}/{len(questions)}] {q}"
        )

        result = ask(
            q,
            model_key=model_key,
            top_k=3
        )

        retrieval_precision = context_precision(
            result["contexts"],
            gold_pages
        )

        retrieval_recall = context_recall(
            result["contexts"],
            gold_pages
        )

        mrr = reciprocal_rank(
            result["contexts"],
            gold_pages
        )

        answer_metrics = evaluate_answer(
            result["answer"],
            reference,
            q,
            result["contexts"]
        )

        rows.append({

            "id": item["id"],

            "question": q,

            "model": model_key,

            "answer": result["answer"],

            "retrieved_pages": [
                x["page"]
                for x in result["contexts"]
            ],

            # Retrieval
            "context_precision": retrieval_precision,

            "context_recall": retrieval_recall,

            "mrr": mrr,

            # Answer
            **answer_metrics,

            # System
            "latency_seconds":
                result["latency_seconds"],
        })

        print(
            f"Pages: {[x['page'] for x in result['contexts']]}"
        )

        print(
            f"Precision: {retrieval_precision:.3f} | "
            f"Recall: {retrieval_recall:.3f} | "
            f"MRR: {mrr:.3f} | "
            f"Groundedness: {answer_metrics['groundedness_proxy']:.3f} | "
            f"Latency: {result['latency_seconds']}s"
        )

    RESULTS_DIR.mkdir(
        exist_ok=True
    )

    # JSON
    json_path = (
        RESULTS_DIR /
        f"{model_key}_evaluation.json"
    )

    json_path.write_text(
        json.dumps(
            rows,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # CSV
    try:

        import pandas as pd

        df = pd.DataFrame(rows)

        csv_path = (
            RESULTS_DIR /
            f"{model_key}_evaluation.csv"
        )

        df.to_csv(
            csv_path,
            index=False
        )

        # Summary
        summary = {
            "model": model_key,
            "questions": len(rows),
            "avg_context_precision":
                round(
                    df["context_precision"].mean(),
                    4
                ),
            "avg_context_recall":
                round(
                    df["context_recall"].mean(),
                    4
                ),
            "avg_mrr":
                round(
                    df["mrr"].mean(),
                    4
                ),
            "avg_answer_similarity":
                round(
                    df["answer_similarity"].mean(),
                    4
                ),
            "avg_keyword_coverage":
                round(
                    df["keyword_coverage"].mean(),
                    4
                ),
            "avg_groundedness":
                round(
                    df["groundedness_proxy"].mean(),
                    4
                ),
            "avg_answer_relevancy":
                round(
                    df["answer_relevancy_proxy"].mean(),
                    4
                ),
            "avg_latency_seconds":
                round(
                    df["latency_seconds"].mean(),
                    3
                ),
        }

        summary_path = (
            RESULTS_DIR /
            f"{model_key}_summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2
            ),
            encoding="utf-8"
        )

        print("\n==============================")
        print(f"{model_key.upper()} SUMMARY")
        print("==============================")

        for key, value in summary.items():
            print(f"{key}: {value}")

    except ImportError:

        print(
            "pandas not installed; "
            "CSV/summary generation skipped."
        )

    return rows


if __name__ == "__main__":

    for model in [
        "smollm",
        "llama"
    ]:

        print(
            f"\n\n========== "
            f"RUNNING {model.upper()} "
            f"=========="
        )

        run_evaluation(model)