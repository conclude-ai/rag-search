from typing import List, Optional
from .chunking_base import BaseChunking
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "langchain_recursive",
    ComponentCategory.CHUNKING,
    default_params={"chunk_size": 1000, "chunk_overlap": 200},
    description="LangChain RecursiveCharacterTextSplitter"
)
class LangChainRecursiveChunking(BaseChunking):
    """
    Wrapper for LangChain's RecursiveCharacterTextSplitter.

    Recursively splits text using a list of separators, trying larger
    separators first (paragraphs, then sentences, then words).
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "Install `langchain-text-splitters` to use this chunking method: "
                "pip install langchain-text-splitters"
            )

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators or ["\n\n", "\n", " ", ""]
        )

    def chunk(self, text: str) -> List[str]:
        documents = self._splitter.create_documents([text])
        return [doc.page_content for doc in documents]
