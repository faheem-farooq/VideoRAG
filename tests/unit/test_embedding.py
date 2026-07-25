from unittest.mock import MagicMock, patch

import numpy as np

from videorag.ingestion.embedding import Embedder


def test_embed_returns_list_of_vectors():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.1, 0.2], [0.3, 0.4]])

    embedder = Embedder(model_name="fake-model")
    with patch.object(Embedder, "_load_model", return_value=fake_model):
        vectors = embedder.embed(["hello", "world"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    fake_model.encode.assert_called_once_with(
        ["hello", "world"], normalize_embeddings=True, convert_to_numpy=True
    )


def test_embed_empty_list_returns_empty_without_loading_model():
    embedder = Embedder(model_name="fake-model")
    with patch.object(Embedder, "_load_model") as mock_load:
        result = embedder.embed([])

    assert result == []
    mock_load.assert_not_called()


def test_embed_one_returns_single_vector():
    fake_model = MagicMock()
    fake_model.encode.return_value = np.array([[0.5, 0.6]])

    embedder = Embedder(model_name="fake-model")
    with patch.object(Embedder, "_load_model", return_value=fake_model):
        vector = embedder.embed_one("hello")

    assert vector == [0.5, 0.6]
