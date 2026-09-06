from src.retrieval import BM25, rank_fused_results, tokenize


def test_tokenize_normalizes_text():
    assert tokenize("Hybrid Search with BM25") == [
        "hybrid",
        "search",
        "with",
        "bm25",
    ]


def test_bm25_prefers_exact_term_match():
    documents = [
        "Hybrid search combines dense and lexical retrieval.",
        "Dense embeddings provide semantic similarity.",
        "BM25 is a lexical retrieval method.",
    ]

    bm25 = BM25(documents)
    ranking = bm25.rank("BM25 lexical retrieval")

    assert ranking[0] == 2


def test_rrf_combines_rankings():
    rankings = [
        [0, 1, 2],
        [1, 0, 2],
    ]

    results = rank_fused_results(rankings)

    assert results[0][0] in {0, 1}
    assert results[0][1] > results[-1][1]

def test_bm25_and_dense_rankings_can_be_fused():
    dense_ranking = [2, 0, 1]
    lexical_ranking = [1, 2, 0]

    results = rank_fused_results(
        [dense_ranking, lexical_ranking],
        limit=3,
    )

    indexes = [index for index, _score in results]

    assert set(indexes) == {0, 1, 2}
    
def test_bm25_returns_scores_for_all_documents():
    documents = [
        "Hybrid search combines dense and lexical retrieval.",
        "Dense embeddings provide semantic similarity.",
        "BM25 is a lexical retrieval method.",
    ]

    bm25 = BM25(documents)
    scores = bm25.score("BM25 lexical retrieval")

    assert len(scores) == len(documents)
    assert scores[2] > scores[0]
    assert scores[2] > scores[1]