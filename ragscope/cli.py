"""Command-line interface: ``ragscope analyze | report | demo``."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path
from typing import List, Optional

from . import __version__
from .analyze import Thresholds, TraceAnalysis, analyze_all, sentence_transformers_embedder
from .report import write_report
from .terminal import render, render_summary
from .trace import load_traces


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--embed", metavar="MODEL", nargs="?", const="sentence-transformers/all-MiniLM-L6-v2",
                   help="add semantic matching with a sentence-transformers model "
                        "(needs: pip install 'ragscope[semantic]')")
    p.add_argument("--supported", type=float, default=Thresholds.supported, metavar="X",
                   help="support score needed to count a sentence as supported (default %(default)s)")
    p.add_argument("--partial", type=float, default=Thresholds.partial, metavar="X",
                   help="support score needed to count a sentence as partial (default %(default)s)")


def _run_analysis(args: argparse.Namespace) -> List[TraceAnalysis]:
    traces = []
    for f in args.files:
        try:
            traces.extend(load_traces(f))
        except FileNotFoundError:
            raise SystemExit(f"ragscope: no such file: {f}") from None
        except ValueError as exc:
            raise SystemExit(f"ragscope: {exc}") from None
    th = Thresholds(supported=args.supported, partial=args.partial)
    embed = sentence_transformers_embedder(args.embed) if getattr(args, "embed", None) else None
    return analyze_all(traces, thresholds=th, embed=embed)


def cmd_analyze(args: argparse.Namespace) -> int:
    analyses = _run_analysis(args)
    if args.json:
        print(json.dumps([a.to_dict() for a in analyses], indent=2, ensure_ascii=False))
    else:
        shown = [a for a in analyses if not args.only_flagged or a.unsupported or a.verdict != "grounded"]
        for a in shown:
            print(render(a, color=args.color, verbose=args.verbose))
        print(render_summary(analyses, color=args.color))

    failed = []
    if args.fail_under is not None:
        failed += [a for a in analyses if a.groundedness is not None and a.groundedness < args.fail_under]
    if args.fail_on_unsupported:
        failed += [a for a in analyses if a.unsupported]
    if failed:
        ids = sorted({a.trace.id for a in failed})
        print(f"ragscope: {len(ids)} trace(s) failed the check: {', '.join(ids)}", file=sys.stderr)
        return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    analyses = _run_analysis(args)
    out = write_report(analyses, args.output, title=args.title)
    print(f"Wrote {out} ({len(analyses)} trace(s))")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .analyze import analyze
    from .demo import demo_traces
    from .trace import save_traces

    traces = demo_traces()
    analyses = [analyze(t) for t in traces]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_traces(traces, out_dir / "demo_traces.jsonl")
    report = write_report(analyses, out_dir / "demo_report.html", title="ragscope demo")
    for a in analyses:
        print(render(a, color=args.color))
    print(render_summary(analyses, color=args.color))
    print(f"Traces: {out_dir / 'demo_traces.jsonl'}\nReport: {report}")
    if args.open:
        webbrowser.open(report.resolve().as_uri())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ragscope",
        description="See inside your RAG pipeline: which chunks were retrieved, which ones the "
                    "answer used, and which claims aren't supported by any of them.",
    )
    parser.add_argument("--version", action="version", version=f"ragscope {__version__}")
    out = argparse.ArgumentParser(add_help=False)
    color = out.add_mutually_exclusive_group()
    color.add_argument("--color", dest="color", action="store_const", const=True, default=None,
                       help="force coloured output")
    color.add_argument("--no-color", dest="color", action="store_const", const=False,
                       help="disable coloured output")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    a = sub.add_parser("analyze", parents=[out], help="print a diagnosis of each trace to the terminal")
    a.add_argument("files", nargs="+", help="trace files (.jsonl or .json)")
    _add_common(a)
    a.add_argument("--json", action="store_true", help="print machine-readable JSON instead")
    a.add_argument("-v", "--verbose", action="store_true", help="show full chunk text and scores")
    a.add_argument("--only-flagged", action="store_true", help="only print traces that aren't grounded")
    a.add_argument("--fail-under", type=float, metavar="X",
                   help="exit 1 if any trace's groundedness is below X (for CI)")
    a.add_argument("--fail-on-unsupported", action="store_true",
                   help="exit 1 if any answer contains an unsupported sentence (for CI)")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("report", parents=[out], help="write an interactive, self-contained HTML report")
    r.add_argument("files", nargs="+", help="trace files (.jsonl or .json)")
    _add_common(r)
    r.add_argument("-o", "--output", default="ragscope_report.html", help="output path (default %(default)s)")
    r.add_argument("--title", default="ragscope report")
    r.add_argument("--open", action="store_true", help="open the report in a browser")
    r.set_defaults(func=cmd_report)

    d = sub.add_parser("demo", parents=[out], help="run on built-in example traces and write a demo report")
    d.add_argument("-o", "--output-dir", default="ragscope_demo")
    d.add_argument("--open", action="store_true", help="open the report in a browser")
    d.set_defaults(func=cmd_demo)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
