"""Evaluate retrieval, relevance-gate, and generation behavior."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate import generate_answer
from src.index_store import VaultIndex
from src.retrieval import is_relevant

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.json"
RESULTS_PATH = EVAL_DIR / "results.json"

TOP_K = 4


def load_dataset() -> list[dict]:
    """Load the labeled evaluation dataset."""
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def reciprocal_rank(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> float:
    """Return reciprocal rank of the first relevant retrieved source."""
    relevant = set(relevant_sources)

    for rank, source in enumerate(retrieved_sources, start=1):
        if source in relevant:
            return 1.0 / rank

    return 0.0


CITATION_START_PATTERN = re.compile(
    r"\(Note:\s*",
    re.IGNORECASE,
)


def extract_citations(answer: str) -> list[str]:
    """Extract citation bodies, supporting parentheses inside note titles."""
    citations = []

    for match in CITATION_START_PATTERN.finditer(answer):
        depth = 1
        index = match.end()

        while index < len(answer) and depth:
            if answer[index] == "(":
                depth += 1
            elif answer[index] == ")":
                depth -= 1
            index += 1

        if depth == 0:
            citation = answer[match.end() : index - 1].strip()
            citations.append(citation)

    return citations


def citation_note_title(citation: str) -> str:
    """Extract the note title from a citation.

    Citations may include section information such as:
    'Hybrid Search | Section: Combining Scores'
    """
    return citation.split("| Section:", 1)[0].strip()


def evaluate_citations(answer: str, retrieved_sources: list[str]) -> dict:
    citations = extract_citations(answer)

    citation_note_titles = [
        citation_note_title(citation)
        for citation in citations
    ]

    valid_citations = [
        title
        for title in citation_note_titles
        if title in retrieved_sources
    ]

    return {
        "has_citation": bool(citations),
        "citation_count": len(citations),
        "citations": citations,
        "citation_note_titles": citation_note_titles,
        "valid_citation_count": len(valid_citations),
        "all_citations_valid": (
            bool(citations)
            and len(valid_citations) == len(citations)
        ),
    }


def evaluate_query(index: VaultIndex, case: dict) -> dict:
    """Evaluate one query."""
    hits = index.query(case["question"], top_k=TOP_K)

    retrieved_sources = [hit["note_title"] for hit in hits]
    relevant_sources = set(case["relevant_sources"])

    hit_at_k = (
        bool(relevant_sources.intersection(retrieved_sources))
        if case["answerable"]
        else True
    )

    mrr = (
        reciprocal_rank(retrieved_sources, list(relevant_sources))
        if case["answerable"]
        else 0.0
    )

    gate_prediction = is_relevant(hits)

    result = {
        "id": case["id"],
        "question": case["question"],
        "answerable": case["answerable"],
        "gate_prediction": gate_prediction,
        "gate_correct": gate_prediction == case["answerable"],
        "hit_at_k": hit_at_k,
        "mrr": mrr,
        "retrieved_sources": retrieved_sources,
        "generation": {
            "attempted": False,
            "success": False,
            "provider": None,
            "answer": None,
            "citation_count": 0,
            "has_citation": False,
            "valid_citation_count": 0,
            "all_citations_valid": False,
        },
        "scores": [
            {
                "note_title": hit["note_title"],
                "heading": hit["heading"],
                "dense_score": round(hit["dense_score"], 4),
                "bm25_score": round(hit["bm25_score"], 4),
                "rrf_score": round(hit["rrf_score"], 4),
            }
            for hit in hits
        ],
    }

    if not gate_prediction:
        return result

    result["generation"]["attempted"] = True

    try:
        generated = generate_answer(case["question"], hits)
        answer = generated["answer"]
        provider = generated["provider"]

        citation_metrics = evaluate_citations(
            answer,
            retrieved_sources,
        )

        result["generation"].update(
            {
                "success": True,
                "provider": provider,
                "answer": answer,
                **citation_metrics,
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["generation"]["error"] = str(exc)

    return result


def calculate_metrics(results: list[dict]) -> dict:
    """Calculate aggregate retrieval, gate, and generation metrics."""
    answerable = [result for result in results if result["answerable"]]
    unanswerable = [result for result in results if not result["answerable"]]

    hit_count = sum(result["hit_at_k"] for result in answerable)
    mrr_sum = sum(result["mrr"] for result in answerable)

    gate_correct = sum(result["gate_correct"] for result in results)

    answerable_accepted = sum(
        result["gate_prediction"] for result in answerable
    )

    unanswerable_rejected = sum(
        not result["gate_prediction"] for result in unanswerable
    )

    generated = [
    result["generation"]
    for result in results
    if result.get("generation", {}).get("attempted", False)
    ]

    successful_generations = sum(
        generation["success"] for generation in generated
    )

    cited_answers = sum(
        generation["has_citation"]
        for generation in generated
        if generation["success"]
    )

    fully_valid_citations = sum(
        generation["all_citations_valid"]
        for generation in generated
        if generation["success"]
    )

    return {
        "query_count": len(results),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "retrieval": {
            f"hit_at_{TOP_K}": (
                hit_count / len(answerable) if answerable else 0.0
            ),
            "mrr": mrr_sum / len(answerable) if answerable else 0.0,
        },
        "relevance_gate": {
            "accuracy": gate_correct / len(results) if results else 0.0,
            "answerable_recall": (
                answerable_accepted / len(answerable)
                if answerable
                else 0.0
            ),
            "unanswerable_rejection": (
                unanswerable_rejected / len(unanswerable)
                if unanswerable
                else 0.0
            ),
        },
        "generation": {
            "attempted": len(generated),
            "successful": successful_generations,
            "success_rate": (
                successful_generations / len(generated)
                if generated
                else 0.0
            ),
            "answers_with_citations": cited_answers,
            "citation_coverage": (
                cited_answers / successful_generations
                if successful_generations
                else 0.0
            ),
            "answers_with_fully_valid_citations": fully_valid_citations,
            "citation_validity": (
                fully_valid_citations / successful_generations
                if successful_generations
                else 0.0
            ),
        },
    }


def main() -> None:
    """Run the evaluation suite."""
    load_dotenv()

    dataset = load_dataset()

    index = VaultIndex()

    if not index.load_if_exists("vault"):
        index.build("vault")

    results = [evaluate_query(index, case) for case in dataset]
    metrics = calculate_metrics(results)

    output = {
        "config": {
            "top_k": TOP_K,
            "embedding_model": index.index_metadata.get(
                "embedding_model"
            ),
            "index_schema_version": index.index_metadata.get(
                "schema_version"
            ),
        },
        "metrics": metrics,
        "queries": results,
    }

    with RESULTS_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print("Evaluation complete.")
    print(f"Queries: {metrics['query_count']}")
    print(
        f"Hit@{TOP_K}: "
        f"{metrics['retrieval'][f'hit_at_{TOP_K}']:.3f}"
    )
    print(f"MRR: {metrics['retrieval']['mrr']:.3f}")
    print(
        "Gate accuracy: "
        f"{metrics['relevance_gate']['accuracy']:.3f}"
    )
    print(
        "Answerable recall: "
        f"{metrics['relevance_gate']['answerable_recall']:.3f}"
    )
    print(
        "Unanswerable rejection: "
        f"{metrics['relevance_gate']['unanswerable_rejection']:.3f}"
    )
    print(
        "Generation success: "
        f"{metrics['generation']['success_rate']:.3f}"
    )
    print(
        "Citation coverage: "
        f"{metrics['generation']['citation_coverage']:.3f}"
    )
    print(
        "Citation validity: "
        f"{metrics['generation']['citation_validity']:.3f}"
    )
    print(f"Detailed results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()