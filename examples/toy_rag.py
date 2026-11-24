"""A complete (toy) RAG pipeline instrumented with ragscope. No API keys needed.

    python examples/toy_rag.py
    ragscope report toy_traces.jsonl --open

The "retriever" is a tiny BM25 over a handful of docs, and the "LLM" is a fake
that answers extractively but sometimes embellishes, so you can see what
ragscope flags. Swap in your real retriever and model; the tracing lines stay
the same.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ragscope import Tracer

DOCS = {
    "handbook/leave.md": "Employees get 24 days of paid leave per year. Unused leave can be carried "
                         "over, up to a maximum of 10 days.",
    "handbook/remote.md": "Remote work is allowed up to 3 days per week with manager approval. "
                          "Fully remote roles must be approved by HR.",
    "handbook/expenses.md": "Travel expenses must be submitted within 30 days with receipts. "
                            "Meals are reimbursed up to 50 dollars per day.",
    "handbook/equipment.md": "New employees receive a laptop on their first day. Laptops are "
                             "refreshed every 3 years.",
    "handbook/holidays.md": "The office is closed on public holidays. A list of holidays is "
                            "published every January.",
}


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, docs: dict, k1: float = 1.5, b: float = 0.75):
        self.ids = list(docs)
        self.texts = [docs[i] for i in self.ids]
        self.tfs = [Counter(tokenize(t)) for t in self.texts]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(w for tf in self.tfs for w in tf)
        n = len(self.texts)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 3):
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            scored.append((s, i))
        scored.sort(reverse=True)
        return [{"id": self.ids[i], "text": self.texts[i], "score": round(s, 3),
                 "source": self.ids[i]} for s, i in scored[:k]]


def fake_llm(question: str, chunks: list, embellish: str = "") -> str:
    """Answers with the best-matching sentence from the context, plus optional made-up text."""
    q = set(tokenize(question))
    sentences = [s for c in chunks for s in re.split(r"(?<=\.)\s+", c["text"])]
    best = max(sentences, key=lambda s: len(q & set(tokenize(s))))
    return f"{best} {embellish}".strip()


QUESTIONS = [
    ("How many days of paid leave do employees get?", ""),
    ("Can I work remotely?", "You can also work from another country for up to 6 months."),
    ("How long do I have to submit travel expenses?", ""),
    ("When will my laptop be replaced?", "Laptops are replaced every 2 years."),
    ("What is the parental leave policy?", "Parental leave is 26 weeks at full pay."),
]


def main() -> None:
    retriever = BM25(DOCS)
    tracer = Tracer("toy_traces.jsonl")
    tracer.path.unlink(missing_ok=True)  # start fresh each run

    for question, embellish in QUESTIONS:
        with tracer.trace(question, pipeline="toy-bm25") as t:
            with t.timed("retrieval"):
                chunks = retriever.search(question, k=3)
            t.retrieved(chunks)
            with t.timed("generation"):
                answer = fake_llm(question, chunks, embellish)
            t.answered(answer)
        print(f"Q: {question}\nA: {answer}\n")

    print(f"Wrote {tracer.path}. Now run:\n  ragscope analyze {tracer.path}\n"
          f"  ragscope report {tracer.path} --open")


if __name__ == "__main__":
    main()
