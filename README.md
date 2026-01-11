# RAG-select

A framework for experimenting with and optimizing Retrieval-Augmented Generation (RAG) pipeline architectures.

## Overview

This package provides a framework for testing different RAG pipeline configurations and measuring their performance. It supports pluggable components for document ingestion, chunking strategies, and retrieval methods.

## Features

- **Modular Architecture**: Pluggable components across the ingestion and retrieval stack with sample wrappers over top open-source component offerings.
- **Experiment Pipeline**: Generate and test all combinations of component variants
- **LangChain Integration**: RAGArtifact extends `BaseRetriever` for seamless use with LangChain chains

## Installation

```bash
pip install rag_select
```

## License

MIT
