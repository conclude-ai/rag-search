from typing import List
from .chunking_base import BaseChunking
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "langchain_token",
    ComponentCategory.CHUNKING,
    default_params={"chunk_size": 500, "chunk_overlap": 50},
    description="LangChain TokenTextSplitter (token-aware chunking)"
)
class LangChainTokenChunking(BaseChunking):
    """
    Wrapper for LangChain's TokenTextSplitter.

    Splits text based on token count rather than character count.
    Useful for staying within model token limits.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        try:
            from langchain_text_splitters import TokenTextSplitter
        except ImportError:
            raise ImportError(
                "Install `langchain-text-splitters` to use this chunking method: "
                "pip install langchain-text-splitters"
            )

        self._splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoding_name=encoding_name
        )

    def chunk(self, text: str) -> List[str]:
        documents = self._splitter.create_documents([text])
        return [doc.page_content for doc in documents]
