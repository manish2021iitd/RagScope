# Contributing

Thanks for helping make RAG pipelines less of a black box!

## Setup

```bash
git clone https://github.com/manish2021iitd/ragscope
cd ragscope
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
```

## Guidelines

- **The core stays dependency-free.** Anything that needs a model or a heavy library goes behind an optional extra (see `semantic` in `pyproject.toml`) and an import inside the function that uses it.
- **Every flag must explain itself.** If you add a check, its `Issue` message should tell the user what to look at and, ideally, what to try next.
- **Add a test** for any change in scoring. `tests/test_analyze.py` shows the pattern: build a tiny `Trace`, run `analyze`, assert on labels.
- If you change the report, open `ragscope demo --open` and check it in both light and dark mode.

## Good first issues

- Add more abbreviations to the sentence splitter (`ragscope/text.py`)
- A `--format markdown` option for `ragscope analyze` (handy in PR comments)
- An example for your favourite framework in `examples/`
