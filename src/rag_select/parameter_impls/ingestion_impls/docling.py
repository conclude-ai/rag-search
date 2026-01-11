from typing import Any
from .ingestion_base import BaseIngestion
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "docling",
    ComponentCategory.INGESTION,
    default_params={},
    description="Document ingestion using the docling library"
)
class DoclingIngestion(BaseIngestion):
    def ingest(self, doc: Any) -> str:
        try:
            from docling import parse_document
        except ImportError:
            raise ImportError("Install `docling` to use this ingestion method")
        file_path = doc["file_path"]
        return parse_document(file_path)
