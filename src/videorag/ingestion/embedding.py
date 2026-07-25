class Embedder:
    """Wraps a multilingual sentence-transformers model.

    LaBSE (default) maps 100+ languages into a shared embedding space, which is
    what makes cross-lingual retrieval work directly: a Hindi or Spanish query
    embeds close to the English transcript chunk that answers it, with no
    translation step required before search.
    """

    def __init__(self, model_name: str = "sentence-transformers/LaBSE") -> None:
        self._model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._load_model().get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load_model()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
