"""Trace format and the tiny logging API used to capture RAG runs.

A *trace* is one question -> retrieved chunks -> answer round trip. Traces are
stored as JSON Lines, one trace per line, so they can be appended to from a
running app and inspected later.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

PathLike = Union[str, Path]


@dataclass
class Chunk:
    """One retrieved piece of context."""

    text: str
    id: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_any(cls, obj: Any, index: int = 0) -> "Chunk":
        """Build a Chunk from a str, dict, Chunk, LangChain ``Document``,
        ``(Document, score)`` tuple or LlamaIndex ``NodeWithScore``."""
        if isinstance(obj, Chunk):
            chunk = Chunk(obj.text, obj.id, obj.score, obj.source, dict(obj.metadata))
        elif isinstance(obj, str):
            chunk = cls(text=obj)
        elif isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], (int, float)):
            # LangChain similarity_search_with_score -> (Document, score)
            chunk = cls.from_any(obj[0], index)
            chunk.score = float(obj[1])
        elif isinstance(obj, dict):
            text = obj.get("text", obj.get("page_content", obj.get("content")))
            if text is None:
                raise ValueError(f"chunk dict #{index} has no 'text'/'page_content'/'content' key")
            known = {"text", "page_content", "content", "id", "score", "source", "metadata"}
            meta = dict(obj.get("metadata") or {})
            meta.update({k: v for k, v in obj.items() if k not in known})
            chunk = cls(
                text=str(text),
                id=obj.get("id"),
                score=obj.get("score"),
                source=obj.get("source") or meta.get("source"),
                metadata=meta,
            )
        elif hasattr(obj, "node") and hasattr(obj, "score"):
            # LlamaIndex NodeWithScore
            node = obj.node
            text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
            meta = dict(getattr(node, "metadata", None) or {})
            chunk = cls(
                text=str(text),
                id=getattr(node, "node_id", None) or getattr(node, "id_", None),
                score=obj.score,
                source=meta.get("file_name") or meta.get("source"),
                metadata=meta,
            )
        elif hasattr(obj, "page_content"):
            # LangChain Document
            meta = dict(getattr(obj, "metadata", None) or {})
            chunk = cls(
                text=str(obj.page_content),
                id=getattr(obj, "id", None) or meta.get("id"),
                score=meta.get("score"),
                source=meta.get("source"),
                metadata=meta,
            )
        else:
            raise TypeError(f"don't know how to turn {type(obj).__name__} into a Chunk")
        if chunk.id is None:
            chunk.id = f"chunk-{index + 1}"
        else:
            chunk.id = str(chunk.id)
        if chunk.score is not None:
            chunk.score = float(chunk.score)
        return chunk


@dataclass
class Trace:
    """One RAG round trip."""

    query: str
    chunks: List[Chunk] = field(default_factory=list)
    answer: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)

    # -- building -----------------------------------------------------------
    def retrieved(self, chunks: Iterable[Any]) -> "Trace":
        """Record retrieved chunks (any format :meth:`Chunk.from_any` accepts)."""
        start = len(self.chunks)
        self.chunks.extend(Chunk.from_any(c, start + i) for i, c in enumerate(chunks))
        return self

    def answered(self, answer: Any) -> "Trace":
        """Record the generated answer (a str or an object with ``.content``/``.text``)."""
        if not isinstance(answer, str):
            answer = getattr(answer, "content", None) or getattr(answer, "text", None) or str(answer)
        self.answer = answer
        return self

    @contextmanager
    def timed(self, step: str) -> Iterator[None]:
        """``with trace.timed("retrieval"): ...`` records the step's duration in ms."""
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[step] = round((time.perf_counter() - t0) * 1000, 2)

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trace":
        query = data.get("query", data.get("question", data.get("input")))
        if query is None:
            raise ValueError("trace has no 'query' (or 'question'/'input') field")
        chunk_data = data.get("chunks", data.get("contexts", data.get("context", [])))
        if isinstance(chunk_data, str):
            chunk_data = [chunk_data]
        answer = data.get("answer", data.get("response", data.get("output", "")))
        kwargs: Dict[str, Any] = {}
        for key in ("id", "timestamp"):
            if data.get(key) is not None:
                kwargs[key] = data[key]
        return cls(
            query=str(query),
            chunks=[Chunk.from_any(c, i) for i, c in enumerate(chunk_data or [])],
            answer=str(answer or ""),
            metadata=dict(data.get("metadata") or {}),
            timings=dict(data.get("timings") or {}),
            **kwargs,
        )


def load_traces(path: PathLike) -> List[Trace]:
    """Load traces from ``.jsonl`` (one per line) or ``.json`` (object or list)."""
    p = Path(path)
    raw = p.read_text(encoding="utf-8")
    stripped = raw.lstrip()
    if stripped.startswith("[") or (p.suffix == ".json" and stripped.startswith("{")):
        data = json.loads(raw)
        items = data if isinstance(data, list) else data.get("traces", [data])
    else:
        items = []
        for n, line in enumerate(raw.splitlines(), 1):
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{n}: invalid JSON ({exc.msg})") from None
    return [Trace.from_dict(item) for item in items]


def save_traces(traces: Iterable[Trace], path: PathLike, *, append: bool = False) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a" if append else "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")


class Tracer:
    """Append traces to a JSONL file from inside your RAG pipeline.

    >>> tracer = Tracer("traces.jsonl")                         # doctest: +SKIP
    >>> with tracer.trace(question) as t:                       # doctest: +SKIP
    ...     t.retrieved(retriever.invoke(question))
    ...     t.answered(llm.invoke(prompt))
    """

    def __init__(self, path: PathLike = "ragscope_traces.jsonl", *, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        self._lock = threading.Lock()

    def write(self, trace: Trace) -> Trace:
        if self.enabled:
            with self._lock:
                save_traces([trace], self.path, append=True)
        return trace

    def log(self, query: str, chunks: Iterable[Any], answer: Any, **metadata: Any) -> Trace:
        """One-shot logging when you already have all three pieces."""
        trace = Trace(query=query, metadata=metadata).retrieved(chunks).answered(answer)
        return self.write(trace)

    @contextmanager
    def trace(self, query: str, **metadata: Any) -> Iterator[Trace]:
        """Context manager; the trace is written on exit, even if an error is raised."""
        t = Trace(query=query, metadata=metadata)
        t0 = time.perf_counter()
        try:
            yield t
        except BaseException as exc:
            t.metadata["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            t.timings.setdefault("total", round((time.perf_counter() - t0) * 1000, 2))
            self.write(t)
