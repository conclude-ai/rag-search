from typing import Any
from .ingestion_base import BaseIngestion
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "simple",
    ComponentCategory.INGESTION,
    default_params={},
    description="Simple text file ingestion (txt, md, html)"
)
class SimpleIngestion(BaseIngestion):
    """
    Simple file-based ingestion for text, markdown, and HTML files.

    Accepts documents with either:
    - "file_path": Path to a text file
    - "text": Raw text content directly
    """

    def ingest(self, doc: Any) -> str:
        # If text is provided directly, use it
        if "text" in doc:
            return doc["text"]

        # Otherwise read from file
        file_path = doc.get("file_path")
        if not file_path:
            raise ValueError("Document must have 'text' or 'file_path' key")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Basic HTML stripping if needed
        if file_path.endswith('.html') or file_path.endswith('.htm'):
            content = self._strip_html(content)

        return content

    def _strip_html(self, html: str) -> str:
        """Basic HTML tag stripping."""
        import re
        # Remove script and style elements
        clean = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        clean = re.sub(r'<style[^>]*>.*?</style>', '', clean, flags=re.DOTALL)
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', ' ', clean)
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean)
        return clean.strip()
