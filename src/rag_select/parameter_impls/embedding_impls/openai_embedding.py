from typing import List
from .embedding_base import BaseEmbedding
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "openai",
    ComponentCategory.EMBEDDING,
    default_params={"model": "text-embedding-3-small"},
    description="OpenAI embeddings API"
)
class OpenAIEmbedding(BaseEmbedding):
    def __init__(self, model: str = "text-embedding-3-small", api_key: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install `openai` to use this embedding method")

        self.model = model
        self.client = OpenAI(api_key=api_key)

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]
