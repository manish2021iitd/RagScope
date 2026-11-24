from ragscope import text as T


def test_stem_groups_inflections():
    assert T.stem("retrieved") == T.stem("retrieves") == T.stem("retrieve")
    assert T.stem("retries") == T.stem("retried")
    assert T.stem("embeddings") == T.stem("embedding") == T.stem("embed")
    assert T.stem("processes") == T.stem("processing") == T.stem("process")
    assert T.stem("running") == "run"


def test_numbers_are_normalized():
    assert T.content_keys("10,000 requests and 40% off at 3.50") == ["10000", "request", "40", "3.5"]


def test_alphanumeric_words_stay_together():
    assert "s3" in T.content_keys("store it in S3")
    assert "3" not in T.content_keys("store it in S3")


def test_negation_is_kept():
    assert "not" in T.content_keys("this is not supported")


def test_sentence_split_handles_abbreviations_and_lists():
    text = "Use a reranker, e.g. a cross-encoder. It helps a lot!\n- First item\n- Second item"
    spans = T.sentence_spans(text)
    sents = [text[s:e] for s, e in spans]
    assert sents == [
        "Use a reranker, e.g. a cross-encoder.",
        "It helps a lot!",
        "- First item",
        "- Second item",
    ]


def test_sentence_split_keeps_last_sentence():
    text = "One sentence here. Another one there. And a final one"
    assert len(T.sentence_spans(text)) == 3


def test_mask_markers_removes_citations_but_keeps_offsets():
    s = "Logs are kept for 30 days [1]."
    masked = T.mask_markers(s)
    assert len(masked) == len(s)
    assert "1" not in T.content_keys(masked)
    assert "30" in T.content_keys(masked)


def test_segments_cover_whole_text():
    s = "Hello, world! 42 apples."
    assert "".join(piece for piece, _ in T.segments(s)) == s


def test_query_keys_drop_question_scaffolding():
    assert T.query_keys("How long are logs kept?") == ["log", "kept"]
    assert T.query_keys("How many retries?") == ["retry"]
