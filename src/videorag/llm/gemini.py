from videorag.retrieval.store import RetrievedSegment

_SYSTEM_INSTRUCTION = (
    "You answer a user's question using only the provided video transcript segments. "
    "Respond in the same language as the question. Be concise (2-4 sentences). "
    "If the segments don't actually answer the question, say so plainly."
)


class GeminiSynthesizer:
    """Answer synthesis only — not a translation hop.

    Retrieval already works cross-lingually via multilingual embeddings, so this
    is the only place an LLM call happens: turning the top retrieved English
    transcript segments into a direct answer in the language the user asked in.
    """

    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash") -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._model = None

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _load_model(self):
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=self._api_key)
            self._model = genai.GenerativeModel(
                self._model_name, system_instruction=_SYSTEM_INSTRUCTION
            )
        return self._model

    def synthesize(self, query: str, segments: list[RetrievedSegment]) -> str | None:
        if not self.is_configured or not segments:
            return None
        model = self._load_model()
        context = "\n\n".join(f"[{s.start:.1f}s-{s.end:.1f}s] {s.text}" for s in segments)
        prompt = f"Question: {query}\n\nTranscript segments:\n{context}"
        response = model.generate_content(prompt)
        return response.text.strip()
