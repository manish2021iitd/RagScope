"""Plain-terminal rendering of an analysis (no dependencies; honours NO_COLOR)."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import List, Optional, TextIO

from .analyze import NEUTRAL, PARTIAL, SUPPORTED, UNSUPPORTED, TraceAnalysis

_CODES = {"red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35",
          "cyan": "36", "dim": "2", "bold": "1"}


class _Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, *styles: str) -> str:
        if not self.enabled or not styles:
            return text
        codes = ";".join(_CODES[s] for s in styles)
        return f"\033[{codes}m{text}\033[0m"


def _use_color(stream: TextIO, color: Optional[bool]) -> bool:
    if color is not None:
        return color
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    return hasattr(stream, "isatty") and stream.isatty()


_LABEL = {
    SUPPORTED: ("✓", "green"),
    PARTIAL: ("~", "yellow"),
    UNSUPPORTED: ("✗", "red"),
    NEUTRAL: ("·", "dim"),
}
_VERDICT = {"grounded": "green", "partially grounded": "yellow", "ungrounded": "red", "empty": "dim"}
_SEVERITY = {"error": ("✗", "red"), "warning": ("!", "yellow"), "info": ("i", "blue")}


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{round(x * 100)}%"


def render(analysis: TraceAnalysis, *, color: Optional[bool] = None,
           stream: Optional[TextIO] = None, verbose: bool = False) -> str:
    stream = stream or sys.stdout
    st = _Style(_use_color(stream, color))
    width = max(60, min(shutil.get_terminal_size((100, 20)).columns, 120))
    t = analysis.trace
    out: List[str] = []

    title = f" trace {t.id} "
    if t.metadata.get("scenario"):
        title += f"· {t.metadata['scenario']} "
    out.append(st("━━" + title + "━" * max(0, width - len(title) - 2), "bold"))
    out.append(st("Q: ", "bold") + t.query)
    used = sum(c.used for c in analysis.chunks)
    out.append(
        st(analysis.verdict.upper(), "bold", _VERDICT.get(analysis.verdict, "dim"))
        + st(f"   groundedness {_pct(analysis.groundedness)}"
             f"   context used {used}/{len(analysis.chunks)}"
             f"   query coverage {_pct(analysis.query_coverage)}", "dim")
    )

    out.append("")
    out.append(st("Answer", "bold"))
    for s in analysis.sentences:
        sym, colour = _LABEL[s.label]
        prefix = f" {st(sym, colour)} [{s.index + 1}] "
        body = textwrap.fill(s.text, width=width - 8, subsequent_indent=" " * 7)
        out.append(prefix + (st(body, "dim") if s.label == NEUTRAL else body))
        if s.label == NEUTRAL:
            continue
        detail = []
        if s.sources:
            detail.append("from " + ", ".join(
                f"#{j + 1} ({s.chunk_scores[j]:.2f})" for j in s.sources[:3]))
        if s.label != SUPPORTED and s.missing_terms:
            detail.append("not in context: " + ", ".join(s.missing_terms[:8]))
        if s.missing_numbers:
            detail.append(st(s.reason, "red"))
        elif verbose:
            detail.append(f"support {s.support:.2f}")
        if detail:
            out.append(" " * 7 + st("↳ " + " · ".join(detail), "dim"))

    out.append("")
    out.append(st("Retrieved chunks", "bold"))
    for c, chunk in zip(analysis.chunks, t.chunks):
        score = "" if c.score is None else f" score {c.score:.2f}"
        if c.used:
            tag = st("used by " + ", ".join(f"[{i + 1}]" for i in c.cited_by), "green")
        else:
            tag = st("unused", "dim")
        extra = ""
        if c.duplicate_of is not None:
            extra = st(f"  duplicate of #{c.duplicate_of + 1}", "yellow")
        out.append(f" #{c.rank} {st(c.id, 'cyan')}{st(score, 'dim')}  {tag}{extra}")
        snippet = " ".join(chunk.text.split())
        limit = width - 9
        if not verbose and len(snippet) > limit:
            snippet = snippet[: limit - 1] + "…"
        out.append(st(textwrap.fill(snippet, width=width - 4, initial_indent="    ",
                                    subsequent_indent="    "), "dim"))

    if analysis.issues:
        out.append("")
        out.append(st("Issues", "bold"))
        for issue in analysis.issues:
            sym, colour = _SEVERITY[issue.severity]
            out.append(f" {st(sym, colour)} " + textwrap.fill(
                issue.message, width=width - 4, subsequent_indent="   "))
    out.append("")
    return "\n".join(out)


def render_summary(analyses: List[TraceAnalysis], *, color: Optional[bool] = None,
                   stream: Optional[TextIO] = None) -> str:
    stream = stream or sys.stdout
    st = _Style(_use_color(stream, color))
    scored = [a.groundedness for a in analyses if a.groundedness is not None]
    avg = sum(scored) / len(scored) if scored else None
    flagged = sum(1 for a in analyses if a.unsupported)
    counts = {}
    for a in analyses:
        counts[a.verdict] = counts.get(a.verdict, 0) + 1
    verdicts = "  ".join(st(f"{k} {v}", _VERDICT.get(k, "dim")) for k, v in counts.items())
    return (st(f"{len(analyses)} trace(s)", "bold")
            + f"  avg groundedness {_pct(avg)}  "
            + st(f"{flagged} with unsupported claims", "red" if flagged else "green")
            + "\n" + verdicts + "\n")
