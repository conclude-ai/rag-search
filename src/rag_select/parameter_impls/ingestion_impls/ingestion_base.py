from abc import ABC, abstractmethod
from typing import Any

class BaseIngestion(ABC):
    @abstractmethod
    def ingest(self, doc: Any) -> str:
        pass
