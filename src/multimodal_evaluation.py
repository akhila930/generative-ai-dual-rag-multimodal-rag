import json
import time
from pathlib import Path

from .multimodal import ask_multimodal


QUESTIONS = [
    {
        "id": "MM01",
        "question": "What information is shown on page 3 of the Research Incubation Handbook?",
        "expected_keywords": [
            "Introduction",
            "Research Track",
            "Innovation Track"
        ],
        "expected_page": 3
    },
    {
        "id": "MM02",
        "question": "What grades and grade points are specified for the Research Incubation course?",
        "expected_keywords": [
            "A+",
            "10",
            "A",
            "9",
            "B",
            "8",
            "C",
            "7",
            "D",
            "6"
        ],
        "expected_page": 11
    }
]


def keyword_coverage(answer, keywords):
    answer_lower = answer.lower()

    matched = [
        keyword
        for keyword in keywords
        if keyword.lower() in answer_lower
    ]

    return len(matched) / len(keywords) if keywords else 0


def evaluate():
    results = []

    for item in QUESTIONS:

        print("\n" + "=" * 70)
        print(item["id"])
        print(item["question"])
        print("=" * 70)

        start = time.time()

        result = ask_multimodal(
            item["question"],
            top_k=2
        )

        total_time = time.time() - start

        answer = result["answer"]

        coverage = keyword_coverage(
            answer,
            item["expected_keywords"]
        )

        retrieved_pages = [
            c["page"]
            for c in result["contexts"]
        ]

        page_retrieved = (
            item["expected_page"]
            in retrieved_pages
        )

        record = {
            "id": item["id"],
            "question": item["question"],
            "expected_page": item["expected_page"],
            "retrieved_pages": retrieved_pages,
            "expected_keywords": item["expected_keywords"],
            "keyword_coverage": round(coverage, 4),
            "expected_page_retrieved": page_retrieved,
            "answer": answer,
            "latency_seconds": round(
                total_time,
                3
            )
        }

        results.append(record)

        print("\nANSWER:")
        print(answer)

        print("\nRETRIEVED PAGES:")
        print(retrieved_pages)

        print("\nKEYWORD COVERAGE:")
        print(round(coverage, 4))

        print("\nEXPECTED PAGE RETRIEVED:")
        print(page_retrieved)

        print("\nLATENCY:")
        print(round(total_time, 3), "seconds")

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    output_file = (
        output_dir /
        "multimodal_evaluation.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("Evaluation complete.")
    print(f"Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    evaluate()
