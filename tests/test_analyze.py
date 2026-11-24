from ragscope import (
    NEUTRAL,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    Chunk,
    Thresholds,
    Trace,
    analyze,
)
from ragscope.demo import demo_traces

PARIS = Chunk(id="paris", text="Paris is the capital of France. It has a population of 2.1 million people.")
BERLIN = Chunk(id="berlin", text="Berlin is the capital of Germany and its largest city.")


def labels(a):
    return [s.label for s in a.sentences]


def test_grounded_answer():
    t = Trace(query="What is the capital of France?", chunks=[PARIS, BERLIN],
              answer="The capital of France is Paris.")
    a = analyze(t)
    assert labels(a) == [SUPPORTED]
    assert a.sentences[0].sources == [0]
    assert a.verdict == "grounded"
    assert a.groundedness == 1.0
    assert a.chunks[0].used and not a.chunks[1].used


def test_hallucinated_sentence_is_flagged():
    t = Trace(query="What is the capital of France?", chunks=[PARIS],
              answer="Paris is the capital of France. The Eiffel Tower was built by Gustave Eiffel.")
    a = analyze(t)
    assert labels(a) == [SUPPORTED, UNSUPPORTED]
    assert "Eiffel" in a.sentences[1].missing_terms
    assert any(i.code == "unsupported" for i in a.issues)
    # more than half of the factual content is unsupported
    assert a.groundedness < 0.5 and a.verdict == "ungrounded"


def test_wrong_number_is_flagged_even_when_words_match():
    t = Trace(query="Population of Paris?", chunks=[PARIS],
              answer="Paris has a population of 3.4 million people.")
    a = analyze(t)
    s = a.sentences[0]
    assert s.label == UNSUPPORTED
    assert s.missing_numbers == ["3.4"]
    assert s.support > 0.5  # lexically close; only the number gives it away


def test_formatted_numbers_match():
    chunk = Chunk(text="The plan allows 10,000 requests per day.")
    t = Trace(query="daily limit?", chunks=[chunk], answer="You get 10000 requests per day.")
    assert labels(analyze(t)) == [SUPPORTED]


def test_citation_markers_are_not_treated_as_numbers():
    t = Trace(query="capital?", chunks=[PARIS], answer="Paris is the capital of France [1].")
    assert labels(analyze(t)) == [SUPPORTED]


def test_filler_sentences_are_neutral_and_ignored():
    t = Trace(query="capital?", chunks=[PARIS],
              answer="Paris is the capital of France. Hope this helps!")
    a = analyze(t)
    assert labels(a) == [SUPPORTED, NEUTRAL]
    assert a.groundedness == 1.0


def test_synthesis_across_chunks_is_credited():
    t = Trace(query="capitals", chunks=[PARIS, BERLIN],
              answer="Paris and Berlin are both capital cities of France and Germany.")
    a = analyze(t)
    assert a.sentences[0].label in (SUPPORTED, PARTIAL)
    assert set(a.sentences[0].sources) == {0, 1}


def test_no_chunks():
    a = analyze(Trace(query="anything", chunks=[], answer="Something made up."))
    assert labels(a) == [UNSUPPORTED]
    assert any(i.code == "no-context" for i in a.issues)
    assert a.verdict == "ungrounded"


def test_empty_answer():
    a = analyze(Trace(query="q", chunks=[PARIS], answer=""))
    assert a.verdict == "empty"
    assert a.groundedness is None
    assert any(i.code == "empty-answer" for i in a.issues)


def test_duplicates_and_retrieval_gap():
    dup = Chunk(id="dup", text=PARIS.text)
    t = Trace(query="How do I rotate keys?", chunks=[PARIS, dup], answer="Paris is the capital of France.")
    a = analyze(t)
    assert a.chunks[1].duplicate_of == 0
    codes = {i.code for i in a.issues}
    assert {"duplicate", "retrieval-gap"} <= codes
    assert "rotate" in a.missing_query_terms


def test_thresholds_are_configurable():
    t = Trace(query="q", chunks=[PARIS], answer="Paris is a lovely city in France.")
    strict = analyze(t, thresholds=Thresholds(supported=0.99, partial=0.98))
    assert strict.sentences[0].label == UNSUPPORTED


def test_semantic_embedder_rescues_paraphrase():
    chunk = Chunk(text="Request logs are retained for 30 days.")
    t = Trace(query="log retention", chunks=[chunk], answer="We hold onto logs for a month.")
    assert analyze(t).sentences[0].label != SUPPORTED

    def fake_embed(texts):  # every text maps to the same vector -> cosine 1.0
        return [[1.0, 0.0] for _ in texts]

    assert analyze(t, embed=fake_embed).sentences[0].label == SUPPORTED


def test_semantic_score_does_not_hide_wrong_numbers():
    chunk = Chunk(text="Request logs are retained for 30 days.")
    t = Trace(query="log retention", chunks=[chunk], answer="Logs are retained for 90 days.")
    a = analyze(t, embed=lambda texts: [[1.0, 0.0] for _ in texts])
    assert a.sentences[0].label == UNSUPPORTED


def test_demo_scenarios():
    results = {t.id: analyze(t) for t in demo_traces()}
    assert results["demo-grounded"].verdict == "grounded"
    wrong = results["demo-wrong-number"]
    assert wrong.sentences[0].missing_numbers == ["90"]
    assert wrong.sentences[2].label == UNSUPPORTED
    assert "rotate" in results["demo-retrieval-gap"].missing_query_terms
    noisy = results["demo-noisy-context"]
    assert {i.code for i in noisy.issues} >= {"duplicate", "rerank", "unused-context"}


def test_to_dict_is_json_serialisable():
    import json

    a = analyze(demo_traces()[1])
    json.dumps(a.to_dict())
