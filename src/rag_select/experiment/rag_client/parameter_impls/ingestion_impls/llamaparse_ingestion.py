from typing import Any, Optional
from .ingestion_base import BaseIngestion
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "llamaparse",
    ComponentCategory.INGESTION,
    default_params={"result_type": "markdown"},
    description="Document ingestion using LlamaParse (PDFs, Word docs, etc.)"
)
class LlamaParseIngestion(BaseIngestion):
    """
    Document ingestion using LlamaIndex's LlamaParse.

    Excellent for PDFs, Word documents, PowerPoint, and other complex formats.
    Requires a LlamaCloud API key.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        result_type: str = "markdown"
    ):
        """
        Args:
            api_key: LlamaCloud API key. If None, uses LLAMA_CLOUD_API_KEY env var.
            result_type: Output format - "markdown" or "text"
        """
        self.api_key = api_key
        self.result_type = result_type
        self._parser = None

    def _get_parser(self):
        """Lazy initialization of the parser."""
        if self._parser is None:
            try:
                from llama_parse import LlamaParse
            except ImportError:
                raise ImportError(
                    "Install `llama-parse` to use this ingestion method: "
                    "pip install llama-parse"
                )

            self._parser = LlamaParse(
                api_key=self.api_key,
                result_type=self.result_type
            )
        return self._parser

    def ingest(self, doc: Any) -> str:
        file_path = doc.get("file_path")
        if not file_path:
            raise ValueError("Document must have 'file_path' key for LlamaParse ingestion")

        parser = self._get_parser()
        documents = parser.load_data(file_path)

        # Combine all pages/sections into single text
        return "\n\n".join([d.text for d in documents])
