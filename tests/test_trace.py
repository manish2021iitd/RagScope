import json
from types import SimpleNamespace

import pytest

from ragscope import Chunk, Trace, Tracer, load_traces, save_traces


def test_chunk_from_str_and_dict():
    assert Chunk.from_any("hello", 0).id == "chunk-1"
    c = Chunk.from_any({"content": "x", "score": "0.5", "url": "http://a"}, 3)
    assert c.text == "x" and c.score == 0.5 and c.id == "chunk-4"
    assert c.metadata["url"] == "http://a"


def test_chunk_from_langchain_like_document():
    doc = SimpleNamespace(page_content="body", metadata={"source": "a.md"}, id=None)
    c = Chunk.from_any(doc, 0)
    assert (c.text, c.source, c.id) == ("body", "a.md", "chunk-1")
    c2 = Chunk.from_any((doc, 0.42), 1)
    assert c2.score == 0.42


def test_chunk_from_llamaindex_like_node():
    node = SimpleNamespace(get_content=lambda: "node text", node_id="n1", metadata={"file_name": "f.pdf"})
    c = Chunk.from_any(SimpleNamespace(node=node, score=0.9), 0)
    assert (c.text, c.id, c.score, c.source) == ("node text", "n1", 0.9, "f.pdf")


def test_unknown_chunk_type_raises():
    with pytest.raises(TypeError):
        Chunk.from_any(42, 0)


def test_roundtrip_jsonl(tmp_path):
    t = Trace(query="q", answer="a").retrieved(["one", {"text": "two", "score": 1}])
    p = tmp_path / "t.jsonl"
    save_traces([t, t], p)
    loaded = load_traces(p)
    assert len(loaded) == 2
    assert loaded[0].chunks[1].text == "two"
    assert loaded[0].id == t.id


def test_load_json_with_alternate_field_names(tmp_path):
    p = tmp_path / "t.json"
    p.write_text(json.dumps([{"question": "q", "contexts": ["c1", "c2"], "response": "r"}]))
    (t,) = load_traces(p)
    assert t.query == "q" and t.answer == "r" and [c.text for c in t.chunks] == ["c1", "c2"]


def test_bad_jsonl_line_reports_line_number(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"query": "ok"}\n{not json}\n')
    with pytest.raises(ValueError, match=":2:"):
        load_traces(p)


def test_tracer_context_manager_writes_on_error(tmp_path):
    tracer = Tracer(tmp_path / "out.jsonl")
    with pytest.raises(RuntimeError):
        with tracer.trace("q", user="u1") as t:
            t.retrieved(["ctx"])
            raise RuntimeError("llm timeout")
    (saved,) = load_traces(tmp_path / "out.jsonl")
    assert saved.metadata["error"] == "RuntimeError: llm timeout"
    assert saved.metadata["user"] == "u1"
    assert "total" in saved.timings


def test_tracer_log_and_timed(tmp_path):
    tracer = Tracer(tmp_path / "out.jsonl")
    with tracer.trace("q") as t:
        with t.timed("retrieval"):
            t.retrieved(["c"])
        t.answered(SimpleNamespace(content="answer from a chat model"))
    tracer.log("q2", ["c"], "a2")
    traces = load_traces(tmp_path / "out.jsonl")
    assert [t.query for t in traces] == ["q", "q2"]
    assert traces[0].answer == "answer from a chat model"
    assert "retrieval" in traces[0].timings


def test_disabled_tracer_writes_nothing(tmp_path):
    tracer = Tracer(tmp_path / "out.jsonl", enabled=False)
    tracer.log("q", [], "a")
    assert not (tmp_path / "out.jsonl").exists()
