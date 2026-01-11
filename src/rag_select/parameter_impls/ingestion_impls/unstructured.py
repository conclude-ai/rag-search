from typing import Any
from .ingestion_base import BaseIngestion
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "unstructured",
    ComponentCategory.INGESTION,
    default_params={},
    description="Document ingestion using the unstructured library"
)
class UnstructuredIngestion(BaseIngestion):
    def ingest(self, doc: Any) -> str:
        try:
            from unstructured.partition.auto import partition
        except ImportError:
            raise ImportError("Install `unstructured` to use this ingestion method")
        file_path = doc["file_path"]
        elements = partition(filename=file_path)
        return "\n".join([str(e) for e in elements])
