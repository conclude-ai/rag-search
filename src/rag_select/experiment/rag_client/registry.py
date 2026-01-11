
from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T", bound=type)

class ComponentCategory(str, Enum):
    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    RETRIEVER = "retriever"
    RERANKER = "reranker"
    OTHER = "other"

class ComponentRegistry:
    """No-op (optional) registry for backwards compatibility.

    The codebase previously used decorators like:
        @ComponentRegistry.register("name", ComponentCategory.CHUNKING, ...)
    If you're no longer using a registry, you can keep this as a harmless shim so
    older components still import and run without modification.
    """

    _REGISTRY: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        category: ComponentCategory = ComponentCategory.OTHER,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Callable[[T], T]:
        def decorator(component_cls: T) -> T:
            # Store but do not require anything; safe to ignore.
            cls._REGISTRY.setdefault(category.value, {})[name] = {
                "cls": component_cls,
                "metadata": metadata or {},
            }
            return component_cls
        return decorator

    @classmethod
    def get(cls, name: str, category: ComponentCategory) -> Optional[Type[Any]]:
        item = cls._REGISTRY.get(category.value, {}).get(name)
        return None if item is None else item.get("cls")
