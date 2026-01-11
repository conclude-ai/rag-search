from typing import List
from .embedding_base import BaseEmbedding
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "huggingface",
    ComponentCategory.EMBEDDING,
    default_params={"model_name": "sentence-transformers/all-MiniLM-L6-v2"},
    description="HuggingFace sentence-transformers embeddings"
)
class HuggingFaceEmbedding(BaseEmbedding):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install `sentence-transformers` to use this embedding method")

        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
