from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type


class ComponentCategory(str, Enum):
    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    RETRIEVER = "retriever"
    # Keep both spellings for backward compatibility with existing impls
    RERANKING = "reranker"
    RERANKER = "reranker"

@dataclass(frozen=True)
class ComponentSpec:
    name: str
    category: ComponentCategory
    cls: Type[Any]
    default_params: Dict[str, Any]
    description: str = ""


class ComponentRegistry:
    """Minimal registry used by parameter implementations.

    This keeps the package lightweight while still allowing discovery of available components.

    """

    _registry: Dict[ComponentCategory, Dict[str, ComponentSpec]] = {c: {} for c in ComponentCategory}

    @classmethod
    def register(
        cls,
        name: str,
        category: ComponentCategory,
        *,
        default_params: Optional[Dict[str, Any]] = None,
        description: str = "",
    ) -> Callable[[Type[Any]], Type[Any]]:
        def decorator(component_cls: Type[Any]) -> Type[Any]:
            spec = ComponentSpec(
                name=name,
                category=category,
                cls=component_cls,
                default_params=default_params or {},
                description=description or "",
            )
            cls._registry[category][name] = spec
            return component_cls
        return decorator

    @classmethod
    def get(cls, category: ComponentCategory, name: str) -> ComponentSpec:
        return cls._registry[category][name]

    @classmethod
    def list(cls, category: Optional[ComponentCategory] = None) -> Dict[str, ComponentSpec]:
        if category is None:
            out: Dict[str, ComponentSpec] = {}
            for cat, items in cls._registry.items():
                out.update({f"{cat.value}:{k}": v for k, v in items.items()})
            return out
        return cls._registry[category].copy()
