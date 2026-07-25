from videorag.retrieval.store import RetrievedSegment


class CrossEncoderReranker:
    """Optional second-stage reranker. Feature-flagged (RERANK_ENABLED) and off by
    default: multilingual cross-encoders are heavier to load and slower per-query
    than the embedding search alone, so this is a stretch addition on top of a
    retrieval path that already works and is tested without it.
    """

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self, query: str, segments: list[RetrievedSegment], top_k: int | None = None
    ) -> list[RetrievedSegment]:
        if not segments:
            return segments
        model = self._load_model()
        pairs = [(query, seg.text) for seg in segments]
        scores = model.predict(pairs)
        reranked = [
            RetrievedSegment(
                video_id=seg.video_id,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                score=float(score),
                language=seg.language,
            )
            for seg, score in zip(segments, scores, strict=True)
        ]
        reranked.sort(key=lambda s: s.score, reverse=True)
        return reranked[:top_k] if top_k else reranked
