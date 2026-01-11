from .chunking_base import BaseChunking
from typing import List
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "sliding_window",
    ComponentCategory.CHUNKING,
    default_params={"chunk_size": 512, "chunk_overlap": 50},
    description="Simple character-based sliding window chunking"
)
class SlidingWindowChunking(BaseChunking):
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> List[str]:
        chunks = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(text):
            chunks.append(text[start:start+self.chunk_size])
            start += step
        return chunks
