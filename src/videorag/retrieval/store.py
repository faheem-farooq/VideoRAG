from dataclasses import dataclass
from typing import Any

import chromadb
from chromadb.api import ClientAPI

from videorag.ingestion.chunking import Chunk


@dataclass(frozen=True)
class RetrievedSegment:
    video_id: str
    start: float
    end: float
    text: str
    score: float
    language: str = ""


def get_http_client(host: str, port: int) -> ClientAPI:
    return chromadb.HttpClient(host=host, port=port)


def get_ephemeral_client() -> ClientAPI:
    """In-process, non-persistent client — used by unit tests, no Docker required."""
    return chromadb.EphemeralClient()


class VectorStore:
    """Thin wrapper around a Chroma collection: persists chunk embeddings with
    video_id metadata so queries can be filtered to one video or search across all
    indexed videos, and always survives process restarts (unlike the original
    in-memory FAISS index).
    """

    def __init__(self, client: ClientAPI, collection_name: str = "video_segments") -> None:
        self._collection = client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        video_id: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        language: str = "",
    ) -> None:
        if not chunks:
            return
        ids = [f"{video_id}::{i}" for i in range(len(chunks))]
        metadatas: list[dict[str, Any]] = [
            {"video_id": video_id, "start": c.start, "end": c.end, "language": language}
            for c in chunks
        ]
        documents = [c.text for c in chunks]
        self._collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        video_id: str | None = None,
    ) -> list[RetrievedSegment]:
        where = {"video_id": video_id} if video_id else None
        result = self._collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )
        if not result["ids"] or not result["ids"][0]:
            return []

        segments: list[RetrievedSegment] = []
        metadatas = result["metadatas"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        for metadata, document, distance in zip(metadatas, documents, distances, strict=True):
            segments.append(
                RetrievedSegment(
                    video_id=metadata["video_id"],
                    start=metadata["start"],
                    end=metadata["end"],
                    text=document,
                    score=1.0 - distance,
                    language=metadata.get("language", ""),
                )
            )
        return segments

    def delete_video(self, video_id: str) -> None:
        self._collection.delete(where={"video_id": video_id})

    def list_video_ids(self) -> list[str]:
        records = self._collection.get(include=["metadatas"])
        video_ids = {m["video_id"] for m in records["metadatas"]}
        return sorted(video_ids)
