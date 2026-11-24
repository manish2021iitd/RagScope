"""Attribution and diagnostics: which chunks support which answer sentences,
and what looks wrong with a trace.

The default scorer is lexical (stemmed unigram + bigram overlap) plus a strict
check that every number in the answer appears in the context. It needs no
models and no API keys, is fully deterministic and is fast enough to run in
CI. Pass ``embed=`` to add a semantic score that catches paraphrases.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from . import text as T
from .trace import Trace

SUPPORTED = "supported"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
NEUTRAL = "neutral"  # no factual content ("Hope this helps!")

Embedder = Callable[[List[str]], Sequence[Sequence[float]]]


@dataclass
class Thresholds:
    supported: float = 0.6   # support score at or above which a sentence is "supported"
    partial: float = 0.35    # ... at or above which it is "partial"
    source: float = 0.3      # min chunk score for a chunk to count as a source of a sentence
    duplicate: float = 0.85  # Jaccard similarity above which two chunks are duplicates
    semantic_floor: float = 0.25  # cosine similarity mapped to 0 support
    semantic_span: float = 0.5    # cosine range mapped onto 0..1 support


@dataclass
class SentenceResult:
    index: int
    text: str
    start: int
    end: int
    label: str
    support: float
    reason: str
    sources: List[int] = field(default_factory=list)       # chunk indices, best first
    chunk_scores: List[float] = field(default_factory=list)
    missing_terms: List[str] = field(default_factory=list)  # content words in no chunk
    missing_numbers: List[str] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)


@dataclass
class ChunkResult:
    index: int
    id: str
    rank: int
    score: Optional[float]
    source: Optional[str]
    query_terms: List[str]
    query_coverage: float
    used: bool
    cited_by: List[int]
    duplicate_of: Optional[int] = None


@dataclass
class Issue:
    severity: str  # "error" | "warning" | "info"
    code: str
    message: str


@dataclass
class TraceAnalysis:
    trace: Trace
    sentences: List[SentenceResult]
    chunks: List[ChunkResult]
    issues: List[Issue]
    groundedness: Optional[float]
    context_utilization: Optional[float]
    query_coverage: Optional[float]
    missing_query_terms: List[str]
    verdict: str

    @property
    def unsupported(self) -> List[SentenceResult]:
        return [s for s in self.sentences if s.label == UNSUPPORTED]

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["trace"] = self.trace.to_dict()
        return d


def _overlap(s_keys: set, s_bi: set, c_keys: set, c_bi: set) -> float:
    if not s_keys:
        return 0.0
    uni = len(s_keys & c_keys) / len(s_keys)
    if not s_bi:
        return uni
    bi = len(s_bi & c_bi) / len(s_bi)
    return 0.7 * uni + 0.3 * bi


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def analyze(
    trace: Trace,
    *,
    thresholds: Optional[Thresholds] = None,
    embed: Optional[Embedder] = None,
) -> TraceAnalysis:
    th = thresholds or Thresholds()
    chunks = trace.chunks
    answer = trace.answer or ""

    # -- precompute chunk features -------------------------------------------
    c_keys_list = [T.content_keys(c.text) for c in chunks]
    c_keys = [set(k) for k in c_keys_list]
    c_bi = [T.bigrams(k) for k in c_keys_list]
    all_keys = set().union(*c_keys) if c_keys else set()
    all_numbers = {k for k in all_keys if T.is_number_key(k)}

    # -- sentences -----------------------------------------------------------
    spans = T.sentence_spans(answer)
    sent_texts = [answer[s:e] for s, e in spans]

    semantic: Optional[List[List[float]]] = None
    if embed is not None and sent_texts and chunks:
        vecs = list(embed(sent_texts + [c.text for c in chunks]))
        s_vecs, c_vecs = vecs[: len(sent_texts)], vecs[len(sent_texts):]
        semantic = [
            [
                max(0.0, min(1.0, (_cosine(sv, cv) - th.semantic_floor) / th.semantic_span))
                for cv in c_vecs
            ]
            for sv in s_vecs
        ]

    sentences: List[SentenceResult] = []
    for i, ((start, end), stext) in enumerate(zip(spans, sent_texts)):
        keys = T.content_keys(stext, drop_filler=True, mask=True)
        key_set = set(keys)
        if not key_set:
            sentences.append(SentenceResult(i, stext, start, end, NEUTRAL, 1.0, "no factual content"))
            continue
        s_bi = T.bigrams(keys)
        scores = [_overlap(key_set, s_bi, ck, cb) for ck, cb in zip(c_keys, c_bi)]
        if semantic is not None:
            scores = [max(lex, sem) for lex, sem in zip(scores, semantic[i])]
        best = max(scores, default=0.0)
        union = len(key_set & all_keys) / len(key_set)
        support = max(best, 0.85 * union)

        forms = T.surface_forms(T.mask_markers(stext))
        missing = [forms.get(k, k) for k in dict.fromkeys(keys) if k not in all_keys]
        missing_numbers = [forms.get(k, k) for k in dict.fromkeys(keys)
                           if T.is_number_key(k) and k not in all_numbers]

        if missing_numbers:
            label = UNSUPPORTED
            reason = "number" + ("s " if len(missing_numbers) > 1 else " ") + ", ".join(
                repr(n) for n in missing_numbers) + " not found in any retrieved chunk"
        elif support >= th.supported:
            label, reason = SUPPORTED, "matches retrieved context"
        elif support >= th.partial:
            label, reason = PARTIAL, "only partly matches retrieved context"
        else:
            label, reason = UNSUPPORTED, "not supported by any retrieved chunk"

        cutoff = max(th.source, 0.6 * best)
        sources = sorted((j for j, sc in enumerate(scores) if sc >= cutoff and sc > 0),
                         key=lambda j: -scores[j])
        sentences.append(SentenceResult(
            index=i, text=stext, start=start, end=end, label=label,
            support=round(support, 3), reason=reason, sources=sources,
            chunk_scores=[round(s, 3) for s in scores], missing_terms=missing,
            missing_numbers=missing_numbers, keys=list(dict.fromkeys(keys)),
        ))

    # -- chunks --------------------------------------------------------------
    q_keys = T.query_keys(trace.query)
    q_forms = T.surface_forms(trace.query)
    chunk_results: List[ChunkResult] = []
    for j, c in enumerate(chunks):
        matched = [k for k in q_keys if k in c_keys[j]]
        cited_by = [s.index for s in sentences if j in s.sources]
        dup = next((k for k in range(j) if T.jaccard(c_keys[k], c_keys[j]) >= th.duplicate), None)
        chunk_results.append(ChunkResult(
            index=j, id=c.id or f"chunk-{j + 1}", rank=j + 1, score=c.score, source=c.source,
            query_terms=[q_forms.get(k, k) for k in matched],
            query_coverage=round(len(matched) / len(q_keys), 3) if q_keys else 0.0,
            used=bool(cited_by), cited_by=cited_by, duplicate_of=dup,
        ))

    # -- metrics -------------------------------------------------------------
    judged = [s for s in sentences if s.label != NEUTRAL]
    weights = {SUPPORTED: 1.0, PARTIAL: 0.5, UNSUPPORTED: 0.0}
    if judged:
        total = sum(len(s.keys) for s in judged)
        groundedness: Optional[float] = round(
            sum(weights[s.label] * len(s.keys) for s in judged) / total, 3)
    else:
        groundedness = None
    utilization = round(sum(c.used for c in chunk_results) / len(chunks), 3) if chunks else None
    missing_q = [q_forms.get(k, k) for k in q_keys if k not in all_keys]
    q_cov = round(1 - len(missing_q) / len(q_keys), 3) if q_keys else None

    issues = _issues(trace, sentences, chunk_results, missing_q)

    if not judged:
        verdict = "empty"
    elif groundedness is not None and groundedness < 0.5:
        verdict = "ungrounded"
    elif any(s.label == UNSUPPORTED for s in judged) or (groundedness or 0) < 0.75:
        verdict = "partially grounded"
    else:
        verdict = "grounded"

    return TraceAnalysis(
        trace=trace, sentences=sentences, chunks=chunk_results, issues=issues,
        groundedness=groundedness, context_utilization=utilization, query_coverage=q_cov,
        missing_query_terms=missing_q, verdict=verdict,
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _issues(trace: Trace, sentences: List[SentenceResult], chunks: List[ChunkResult],
            missing_q: List[str]) -> List[Issue]:
    issues: List[Issue] = []
    if not trace.chunks:
        issues.append(Issue("error", "no-context", "No chunks were retrieved for this query."))
    if not trace.answer.strip():
        issues.append(Issue("warning", "empty-answer", "The answer is empty."))
    if trace.metadata.get("error"):
        issues.append(Issue("error", "pipeline-error", f"Pipeline raised {trace.metadata['error']}"))

    unsupported = [s for s in sentences if s.label == UNSUPPORTED]
    if unsupported:
        nums = sorted({n for s in unsupported for n in s.missing_numbers})
        msg = (f"{_plural(len(unsupported), 'answer sentence')} not supported by the retrieved "
               f"context (sentence " + ", ".join(str(s.index + 1) for s in unsupported) + ").")
        if nums:
            msg += " Numbers not in context: " + ", ".join(nums) + "."
        issues.append(Issue("error", "unsupported", msg))
    partial = [s for s in sentences if s.label == PARTIAL]
    if partial:
        issues.append(Issue("warning", "partial",
                            f"{_plural(len(partial), 'sentence')} only partially supported "
                            f"(sentence " + ", ".join(str(s.index + 1) for s in partial) + ")."))

    if missing_q and trace.chunks:
        q_total = len(T.query_keys(trace.query)) or 1
        sev = "warning" if len(missing_q) / q_total >= 0.5 or len(missing_q) >= 2 else "info"
        issues.append(Issue(sev, "retrieval-gap",
                            "Query terms not found in any chunk: " + ", ".join(missing_q)
                            + ". The retriever may have missed the relevant document."))
    off_topic = [c for c in chunks if c.query_coverage == 0 and not c.used]
    if off_topic and any(c.query_terms for c in chunks):
        issues.append(Issue("warning", "off-topic",
                            ("Chunks " if len(off_topic) > 1 else "Chunk ")
                            + ", ".join(f"#{c.rank}" for c in off_topic)
                            + (" share" if len(off_topic) > 1 else " shares")
                            + " no terms with the query and " + ("were" if len(off_topic) > 1 else "was")
                            + " not used."))
    dups = [c for c in chunks if c.duplicate_of is not None]
    for c in dups:
        issues.append(Issue("warning", "duplicate",
                            f"Chunk #{c.rank} is a near-duplicate of chunk #{c.duplicate_of + 1}; "
                            "it wastes context window."))
    if chunks and sentences:
        unused = [c for c in chunks if not c.used]
        if unused:
            sev = "warning" if len(unused) > len(chunks) / 2 else "info"
            issues.append(Issue(sev, "unused-context",
                                f"{len(unused)} of {len(chunks)} retrieved chunks were never used "
                                "in the answer (" + ", ".join(f"#{c.rank}" for c in unused) + ")."))
        if len(chunks) > 3:
            counts: Dict[int, int] = {}
            for s in sentences:
                if s.sources and s.label != NEUTRAL:
                    counts[s.sources[0]] = counts.get(s.sources[0], 0) + 1
            if counts:
                top = max(counts, key=lambda j: (counts[j], -j))
                if chunks[top].rank > 3:
                    issues.append(Issue("info", "rerank",
                                        f"The chunk that supports most of the answer was ranked "
                                        f"#{chunks[top].rank}. A reranker may improve results."))
    return issues


def analyze_all(traces: Sequence[Trace], **kwargs) -> List[TraceAnalysis]:
    return [analyze(t, **kwargs) for t in traces]


def sentence_transformers_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Embedder:
    """Return an embedder backed by sentence-transformers (``pip install ragscope[semantic]``)."""
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "Semantic scoring needs sentence-transformers: pip install 'ragscope[semantic]'"
        ) from exc
    model = SentenceTransformer(model_name)

    def _embed(texts: List[str]) -> Sequence[Sequence[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    return _embed
