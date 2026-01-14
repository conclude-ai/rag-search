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

## Quick Start
To set up an experiment with RAG-select, follow these steps:

1. **Prepare Your Dataset and Documents**
   - `dataset`: This should be your evaluation set (e.g., queries and ground-truths).
   - `documents`: The corpus to be indexed and retrieved from.

2. **Define Component Variants**
   - Specify different variants for each pipeline stage. Import and instantiate implementations as needed.

   Example:
   ```python
   from rag_select.parameter_impls.chunking_impls import SlidingWindowChunking
   from rag_select.parameter_impls.embedding_impls import HuggingFaceEmbedding
   from rag_select.parameter_impls.retriever_impls import SimpleRetriever

   chunking_variants = [
       SlidingWindowChunking(chunk_size=256, chunk_overlap=20),
       SlidingWindowChunking(chunk_size=512, chunk_overlap=50),
   ]
   embedding_variants = [
       HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
   ]
   retriever_variants = [
       SimpleRetriever(top_k=3),
       SimpleRetriever(top_k=5),
   ]
   ```

3. **Run the Experiment**
   - Instantiate the `RAGExperiment` and call `.run()` to evaluate all pipeline combinations.
   - Review, compare, or rank the results.

   Example:
   ```python
   from rag_select.experiment.rag_experiment import RAGExperiment

   experiment = RAGExperiment(
       dataset=eval_dataset,
       documents=documents,
       search_space={
           "chunking": chunking_variants,
           "embedding": embedding_variants,
           "retriever": retriever_variants,
       },
       metrics=["precision@3", "precision@5", "recall@5", "mrr"],
   )

   results = experiment.run()

   # Rank pipelines by a metric
   ranked = results.rank(by="precision@5")

   # Get the best pipeline config
   best = results.get_best_pipeline()
   ```

4. **Extending the Search Space**
   - To experiment with ingestion or storage variants, include them as keys in `search_space` and provide corresponding implementations.

## License

MIT
