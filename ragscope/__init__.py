"""ragscope: see inside your RAG pipeline.

    from ragscope import Tracer, analyze

    tracer = Tracer("traces.jsonl")
    with tracer.trace(question) as t:
        t.retrieved(docs)        # strings, dicts, LangChain Documents, LlamaIndex nodes
        t.answered(answer)

Then run ``ragscope report traces.jsonl --open``.
"""

__version__ = "0.1.0"

from .analyze import (  # noqa: E402
    NEUTRAL,
    PARTIAL,
    SUPPORTED,
    UNSUPPORTED,
    ChunkResult,
    Issue,
    SentenceResult,
    Thresholds,
    TraceAnalysis,
    analyze,
    analyze_all,
)
from .report import build_report, write_report  # noqa: E402
from .trace import Chunk, Trace, Tracer, load_traces, save_traces  # noqa: E402

__all__ = [
    "__version__", "Chunk", "Trace", "Tracer", "load_traces", "save_traces",
    "analyze", "analyze_all", "Thresholds", "TraceAnalysis", "SentenceResult", "ChunkResult",
    "Issue", "build_report", "write_report",
    "SUPPORTED", "PARTIAL", "UNSUPPORTED", "NEUTRAL",
]
