from abc import ABC, abstractmethod
from typing import List

class BaseChunking(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass
