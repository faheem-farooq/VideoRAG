#!/usr/bin/env python
"""Compute precision/recall/F1@k and query latency for VideoRAG retrieval.

Ground truth is "one relevant segment per query": a query counts as a hit at
rank k if any of the top-k retrieved segments (from the correct video) overlaps
the hand-labeled [expected_start, expected_end] window. With a single relevant
item per query this reduces to success@k — precision@k = 1/rank_of_hit,
recall@k = 1 if hit else 0 — which is the standard formulation for this kind of
"does the right segment show up" retrieval eval.

Usage:
    python scripts/run_eval.py --dataset eval/dataset.json --output eval/results.json
    python scripts/run_eval.py --dataset eval/fixture/dataset.json --fake-embedder
"""

import argparse
import json
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from videorag.ingestion.chunking import chunk_transcript  # noqa: E402
from videorag.ingestion.transcription.base import TranscriptSegment  # noqa: E402
from videorag.retrieval.store import VectorStore, get_ephemeral_client  # noqa: E402


class HashingEmbedder:
    """Deterministic, dependency-free stand-in for the real multilingual model.

    Only used with --fake-embedder for the CI fixture, so pipeline/metric
    regressions are caught on every PR without downloading a model. The numbers
    it produces are not meaningful retrieval-quality numbers and must never be
    reported as such.
    """

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for word in text.lower().split():
            vec[hash(word) % self._dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


@dataclass
class QueryResult:
    query: str
    language: str
    video_id: str
    hit: bool
    rank: int | None
    precision: float
    recall: float
    f1: float
    latency_ms: float


def _load_transcript(path: Path) -> list[TranscriptSegment]:
    raw = json.loads(path.read_text())
    return [TranscriptSegment(start=r["start"], end=r["end"], text=r["text"]) for r in raw]


def _overlaps(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return a_start < b_end and b_start < a_end


def run_eval(dataset_path: Path, top_k: int, use_fake_embedder: bool) -> dict:
    dataset = json.loads(dataset_path.read_text())
    dataset_dir = dataset_path.parent

    embedder = HashingEmbedder() if use_fake_embedder else None
    if embedder is None:
        from videorag.ingestion.embedding import Embedder

        embedder = Embedder()

    client = get_ephemeral_client()
    store = VectorStore(client, collection_name=f"eval_{uuid.uuid4().hex}")

    for video in dataset["videos"]:
        segments = _load_transcript(dataset_dir / video["transcript_path"])
        chunks = chunk_transcript(segments)
        embeddings = embedder.embed([c.text for c in chunks])
        store.add_chunks(
            video_id=video["video_id"],
            chunks=chunks,
            embeddings=embeddings,
            language=video.get("language", ""),
        )

    results: list[QueryResult] = []
    for q in dataset["queries"]:
        start = time.perf_counter()
        query_embedding = embedder.embed_one(q["query"])
        retrieved = store.query(query_embedding, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000

        rank = None
        for i, seg in enumerate(retrieved):
            if seg.video_id == q["video_id"] and _overlaps(
                seg.start, seg.end, q["expected_start"], q["expected_end"]
            ):
                rank = i + 1
                break

        hit = rank is not None
        precision = (1.0 / rank) if hit else 0.0
        recall = 1.0 if hit else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if hit else 0.0

        results.append(
            QueryResult(
                query=q["query"],
                language=q["language"],
                video_id=q["video_id"],
                hit=hit,
                rank=rank,
                precision=precision,
                recall=recall,
                f1=f1,
                latency_ms=latency_ms,
            )
        )

    aggregate = {
        "num_queries": len(results),
        "top_k": top_k,
        "mean_precision": statistics.fmean(r.precision for r in results) if results else 0.0,
        "mean_recall": statistics.fmean(r.recall for r in results) if results else 0.0,
        "mean_f1": statistics.fmean(r.f1 for r in results) if results else 0.0,
        "hit_rate": statistics.fmean(1.0 if r.hit else 0.0 for r in results) if results else 0.0,
        "mean_latency_ms": statistics.fmean(r.latency_ms for r in results) if results else 0.0,
        "p95_latency_ms": (
            statistics.quantiles([r.latency_ms for r in results], n=20)[18]
            if len(results) >= 2
            else (results[0].latency_ms if results else 0.0)
        ),
    }

    return {
        "dataset": str(dataset_path),
        "embedder": "fake-hashing" if use_fake_embedder else "sentence-transformers",
        "aggregate": aggregate,
        "queries": [asdict(r) for r in results],
    }


def to_markdown_table(report: dict) -> str:
    agg = report["aggregate"]
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Queries | {agg['num_queries']} |",
        f"| top_k | {agg['top_k']} |",
        f"| Hit rate | {agg['hit_rate']:.2%} |",
        f"| Mean precision@k | {agg['mean_precision']:.3f} |",
        f"| Mean recall@k | {agg['mean_recall']:.3f} |",
        f"| Mean F1@k | {agg['mean_f1']:.3f} |",
        f"| Mean query latency | {agg['mean_latency_ms']:.1f} ms |",
        f"| p95 query latency | {agg['p95_latency_ms']:.1f} ms |",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("eval/dataset.json"))
    parser.add_argument("--output", type=Path, default=Path("eval/results.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("eval/results.md"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--fake-embedder",
        action="store_true",
        help="Use a deterministic hashing embedder instead of the real multilingual "
        "model. For CI only — do not use this to produce reported numbers.",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=None,
        help="Exit non-zero if hit_rate falls below this. Wire into CI as a regression gate.",
    )
    args = parser.parse_args()

    report = run_eval(args.dataset, args.top_k, args.fake_embedder)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    markdown = to_markdown_table(report)
    args.markdown_output.write_text(markdown + "\n")

    print(markdown)
    print(f"\nWrote {args.output} and {args.markdown_output}")

    if args.min_hit_rate is not None and report["aggregate"]["hit_rate"] < args.min_hit_rate:
        print(
            f"\nRegression: hit_rate {report['aggregate']['hit_rate']:.2%} "
            f"is below the required {args.min_hit_rate:.2%}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
