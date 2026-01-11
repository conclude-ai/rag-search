from typing import List, Dict, Optional
import math
from collections import Counter
from .embedding_base import BaseEmbedding
from ...registry import ComponentRegistry, ComponentCategory


@ComponentRegistry.register(
    "simple",
    ComponentCategory.EMBEDDING,
    default_params={"max_features": 1000},
    description="Simple TF-IDF based embeddings (no external dependencies)"
)
class SimpleEmbedding(BaseEmbedding):
    """
    Simple TF-IDF based embedding implementation.

    This is a baseline implementation that doesn't require any external
    dependencies. Useful for testing and development, but not recommended
    for production use cases.
    """

    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self._fitted = False

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        import re
        # Remove punctuation and split on whitespace
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()

    def _compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """Compute term frequency."""
        counts = Counter(tokens)
        total = len(tokens)
        if total == 0:
            return {}
        return {term: count / total for term, count in counts.items()}

    def fit(self, documents: List[str]) -> None:
        """
        Fit the vocabulary and IDF values from a corpus.

        Args:
            documents: List of document strings
        """
        # Tokenize all documents
        doc_tokens = [self._tokenize(doc) for doc in documents]

        # Count document frequency for each term
        df: Dict[str, int] = Counter()
        all_terms: Counter = Counter()

        for tokens in doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                df[term] += 1
            all_terms.update(tokens)

        # Select top terms by frequency
        top_terms = [term for term, _ in all_terms.most_common(self.max_features)]

        # Build vocabulary mapping
        self.vocabulary = {term: idx for idx, term in enumerate(top_terms)}

        # Compute IDF: log(N / df)
        n_docs = len(documents)
        self.idf = {
            term: math.log(n_docs / (df[term] + 1)) + 1
            for term in self.vocabulary
        }

        self._fitted = True

    def _to_vector(self, text: str) -> List[float]:
        """Convert text to TF-IDF vector."""
        if not self._fitted:
            # Auto-fit on single document (not ideal but works)
            self.fit([text])

        tokens = self._tokenize(text)
        tf = self._compute_tf(tokens)

        # Build vector
        vector = [0.0] * len(self.vocabulary)
        for term, freq in tf.items():
            if term in self.vocabulary:
                idx = self.vocabulary[term]
                vector[idx] = freq * self.idf.get(term, 1.0)

        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    def embed_text(self, text: str) -> List[float]:
        return self._to_vector(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        # Fit on the batch if not already fitted
        if not self._fitted:
            self.fit(texts)
        return [self._to_vector(text) for text in texts]
