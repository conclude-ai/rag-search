from typing import List, Optional
from .chunking_base import BaseChunking
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "llamaindex_sentence",
    ComponentCategory.CHUNKING,
    default_params={"chunk_size": 1024, "chunk_overlap": 200},
    description="LlamaIndex SentenceSplitter (sentence-aware chunking)"
)
class LlamaIndexSentenceChunking(BaseChunking):
    """
    Wrapper for LlamaIndex's SentenceSplitter.

    Splits text while trying to preserve sentence boundaries.
    Better for maintaining semantic coherence in chunks.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        paragraph_separator: str = "\n\n\n"
    ):
        try:
            from llama_index.core.node_parser import SentenceSplitter
        except ImportError:
            raise ImportError(
                "Install `llama-index-core` to use this chunking method: "
                "pip install llama-index-core"
            )

        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator=paragraph_separator
        )

    def chunk(self, text: str) -> List[str]:
        # SentenceSplitter works on TextNode, so we create one
        try:
            from llama_index.core.schema import TextNode
        except ImportError:
            raise ImportError("Install `llama-index-core` to use this chunking method")

        node = TextNode(text=text)
        nodes = self._splitter.get_nodes_from_documents([node])
        return [n.text for n in nodes]
