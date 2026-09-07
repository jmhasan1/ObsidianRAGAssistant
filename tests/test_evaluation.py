from evaluation.evaluate import (
    calculate_metrics,
    citation_note_title,
    evaluate_citations,
    extract_citations,
    reciprocal_rank,
    )


def test_reciprocal_rank():
    assert reciprocal_rank(
        ["Wrong", "Correct", "Other"],
        ["Correct"],
    ) == 0.5


def test_reciprocal_rank_returns_zero_when_missing():
    assert reciprocal_rank(
        ["Wrong", "Other"],
        ["Correct"],
    ) == 0.0


def test_calculate_metrics():
    results = [
        {
            "answerable": True,
            "gate_prediction": True,
            "gate_correct": True,
            "hit_at_k": True,
            "mrr": 1.0,
        },
        {
            "answerable": True,
            "gate_prediction": False,
            "gate_correct": False,
            "hit_at_k": True,
            "mrr": 0.5,
        },
        {
            "answerable": False,
            "gate_prediction": False,
            "gate_correct": True,
            "hit_at_k": True,
            "mrr": 0.0,
        },
    ]

    metrics = calculate_metrics(results)

    assert metrics["retrieval"]["hit_at_4"] == 1.0
    assert metrics["retrieval"]["mrr"] == 0.75
    assert metrics["relevance_gate"]["accuracy"] == 2 / 3
    assert metrics["relevance_gate"]["answerable_recall"] == 0.5
    assert metrics["relevance_gate"]["unanswerable_rejection"] == 1.0

def test_citation_evaluation_accepts_retrieved_source():
    answer = (
        "Hybrid search combines dense and lexical retrieval "
        "(Note: Hybrid Search)."
    )

    result = evaluate_citations(
        answer,
        ["Hybrid Search", "Retrieval-Augmented Generation (RAG)"],
    )

    assert result["has_citation"]
    assert result["valid_citation_count"] == 1
    assert result["all_citations_valid"]


def test_citation_evaluation_rejects_unknown_source():
    answer = "Paris is the capital of France (Note: France)."

    result = evaluate_citations(
        answer,
        ["Hybrid Search", "Vector Databases"],
    )
    assert result["has_citation"]
    assert result["valid_citation_count"] == 0
    assert not result["all_citations_valid"]


def test_citation_evaluation_detects_missing_citation():
    answer = "Hybrid search combines lexical and dense retrieval."

    result = evaluate_citations(
        answer,
        ["Hybrid Search"],
    )

    assert not result["has_citation"]
    assert not result["all_citations_valid"]

def test_citation_note_title_extracts_note_from_section():
    assert (
        citation_note_title("Hybrid Search | Section: Combining Scores")
        == "Hybrid Search"
    )


def test_citation_note_title_preserves_title_without_section():
    assert citation_note_title("Hybrid Search") == "Hybrid Search"


def test_evaluate_citations_accepts_retrieved_note_with_section_citation():
    result = evaluate_citations(
        "RRF combines ranked retrieval results. "
        "(Note: Hybrid Search | Section: Combining Scores)",
        ["Hybrid Search", "DocuMind Project"],
    )

    assert result["citation_count"] == 1
    assert result["has_citation"] is True
    assert result["valid_citation_count"] == 1
    assert result["all_citations_valid"] is True
    assert result["citations"] == [
        "Hybrid Search | Section: Combining Scores"
    ]
    assert result["citation_note_titles"] == ["Hybrid Search"]


def test_evaluate_citations_rejects_note_not_in_retrieved_sources():
    result = evaluate_citations(
        "This is supported by another note. "
        "(Note: Retrieval-Augmented Generation (RAG) | Section: Core Stages)",
        ["Hybrid Search", "DocuMind Project"],
    )

    assert result["citation_count"] == 1
    assert result["valid_citation_count"] == 0
    assert result["all_citations_valid"] is False

def test_extract_citations():
    answer = (
        "Hybrid search combines dense and lexical retrieval "
        "(Note: Hybrid Search)."
    )

    assert extract_citations(answer) == ["Hybrid Search"]

def test_extract_citations_supports_parentheses_in_note_titles():
    answer = (
        "The answer is supported by "
        "(Note: Retrieval-Augmented Generation (RAG))."
    )

    assert extract_citations(answer) == [
        "Retrieval-Augmented Generation (RAG)"
    ]