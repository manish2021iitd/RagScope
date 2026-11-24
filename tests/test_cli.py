import json

from ragscope import Trace, save_traces
from ragscope.cli import main


def _write(tmp_path, answer):
    p = tmp_path / "t.jsonl"
    save_traces([Trace(query="capital of France?", chunks=[], answer=answer)
                 .retrieved(["Paris is the capital of France."])], p)
    return p


def test_demo_writes_report(tmp_path, capsys):
    assert main(["demo", "-o", str(tmp_path), "--no-color"]) == 0
    html = (tmp_path / "demo_report.html").read_text()
    assert "ragscope-data" in html and "demo-wrong-number" in html
    assert (tmp_path / "demo_traces.jsonl").exists()
    assert "with unsupported claims" in capsys.readouterr().out


def test_report_command(tmp_path):
    p = _write(tmp_path, "Paris is the capital of France.")
    out = tmp_path / "r.html"
    assert main(["report", str(p), "-o", str(out)]) == 0
    assert out.read_text().startswith("<!DOCTYPE html>")


def test_report_escapes_script_tags(tmp_path):
    p = tmp_path / "t.jsonl"
    save_traces([Trace(query="</script><script>alert(1)</script>", answer="x").retrieved(["y"])], p)
    out = tmp_path / "r.html"
    main(["report", str(p), "-o", str(out)])
    assert "</script><script>alert(1)" not in out.read_text()


def test_analyze_json(tmp_path, capsys):
    p = _write(tmp_path, "Paris is the capital of France.")
    assert main(["analyze", str(p), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data[0]["verdict"] == "grounded"


def test_ci_gates(tmp_path):
    good = _write(tmp_path, "Paris is the capital of France.")
    assert main(["analyze", str(good), "--no-color", "--fail-on-unsupported", "--fail-under", "0.9"]) == 0
    bad = _write(tmp_path, "Lyon is the capital of France since 1850.")
    assert main(["analyze", str(bad), "--no-color", "--fail-on-unsupported"]) == 1
    assert main(["analyze", str(bad), "--no-color", "--fail-under", "0.9"]) == 1


def test_missing_file_is_a_clean_error(tmp_path):
    import pytest

    with pytest.raises(SystemExit, match="no such file"):
        main(["analyze", str(tmp_path / "nope.jsonl")])
