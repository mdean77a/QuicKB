# QuicKB: End-to-End Knowledge Base Processing Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

> From raw text to optimized embeddings in a single workflow

## Overview

QuicKB transforms unstructured text documents into production-ready knowledge bases through a complete pipeline that handles:

- Smart document chunking with multiple strategies
- Synthetic question generation for training data
- Embedding model fine-tuning with dimension optimization
- Automatic Hugging Face Hub integration

## Key Features

### Document Chunking
- Multiple chunking strategies:
  - **RecursiveTokenChunker**: Hierarchical splitting using custom separators
  - **FixedTokenChunker**: Precise token-based chunking
  - **ClusterSemanticChunker**: Content-aware semantic grouping
  - **LLMSemanticChunker**: LLM-guided natural break points
  - **KamradtModifiedChunker**: Hybrid semantic-token approach

### Training Data Generation
- Automatic question generation from chunks
- Semantic deduplication of similar questions
- Configurable similarity thresholds
- Parallel processing for speed

### Embedding Optimization
- Matryoshka embedding training (768→64D)
- Progressive dimensionality reduction
- Automatic evaluation metrics
- Direct Hugging Face Hub upload

## Installation

```bash
git clone https://github.com/AdamLucek/quickb.git
cd quickb

python -m venv quickb-env
source quickb-env/bin/activate  # Windows: quickb-env\Scripts\activate

pip install -e .
```

## Usage

1. Configure your pipeline in `config.yaml`
2. Run:
```bash
python src/main.py
```

## Configuration Guide

The pipeline is controlled through a single `config.yaml` file. Here's a complete configuration example with all available options:

```yaml
# Core Settings
path_to_knowledgebase: "./testing/knowledgebase"  # Input directory
output_path: "./output/knowledgebase-quickb.json" # Output path

# Chunking Configuration
chunker: "RecursiveTokenChunker"
chunker_arguments:
  chunk_size: 400
  chunk_overlap: 0
  separators: ["\n\n", "\n", ".", "?", "!", " ", ""]
  keep_separator: true
  is_separator_regex: false
  length_function: "character"

# Question Generation
generate_questions: true
question_output_path: "./output/train_data.json"
deduplication:
  enabled: true
  similarity_threshold: 0.8

# Hugging Face Integration
hub_username: "YourUsername"
hub_token: null     # Uses HF_TOKEN environment variable
hub_private: false

# Embedding Training
train_embedding: true
training:
  model_id: "nomic-ai/modernbert-embed-base"
  output_dir: "./output/modernbert_mtl"
  epochs: 4
  learning_rate: 2.0e-5
  matryoshka_dimensions: [768, 512, 256, 128, 64]
  batch_size: 32
  gradient_accumulation_steps: 16
  metric_for_best_model: "eval_dim_128_cosine_ndcg@10"
  push_to_hub: true
  hub_model_id: "username/model-name"
```

### Alternative Chunker Configurations

1. **Fixed Token Chunker**
```yaml
chunker: "FixedTokenChunker"
chunker_arguments:
  encoding_name: "cl100k_base"
  model_name: "text-embedding-3-large"
  chunk_size: 400
  chunk_overlap: 50
```

2. **Cluster Semantic Chunker**
```yaml
chunker: "ClusterSemanticChunker"
chunker_arguments:
  embedding_function: "openai"
  max_chunk_size: 400
  min_chunk_size: 50
```

3. **LLM Semantic Chunker**
```yaml
chunker: "LLMSemanticChunker"
chunker_arguments:
  organisation: "openai"  # or "anthropic"
  model_name: "gpt-4o"
```

4. **Kamradt Modified Chunker**
```yaml
chunker: "KamradtModifiedChunker"
chunker_arguments:
  avg_chunk_size: 400
  min_chunk_size: 50
  embedding_function: "openai"
```

## Output Format

### Knowledgebase Dataset
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text": "Section 12.1: Termination clauses...",
  "source": "docs/contracts/2024/Q1-agreement.txt"
}
```

### Training Dataset
```json
{
  "anchor": "What are the termination notice requirements?",
  "positive": "Section 12.1: Either party may terminate...",
  "question_id": "a3b8c7d0-e83a-4b5c-b12d-3f7a8d4c9e1b",
  "chunk_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

## Environment Variables

- `OPENAI_API_KEY`: Required for OpenAI embeddings and question generation
- `HF_TOKEN`: Required for Hugging Face Hub uploads
- `ANTHROPIC_API_KEY`: Optional, required only for Anthropic LLM chunking

## Citations

QuicKB builds upon these foundational works:

1. **Chunking Algorithms**
```bibtex
@software{LangChain_TextSplitters,
  author = {LangChain AI},
  title = {LangChain Text Splitters},
  url = {https://github.com/langchain-ai/langchain},
  license = {MIT}
}
@software{Chroma_Chunking,
  author = {Chroma Research},
  title = {Chunking Evaluation Framework},
  url = {https://github.com/chroma-core/chroma},
  license = {Apache-2.0}
}
```

2. **Training Approach**
```bibtex
@article{Matryoshka_Embeddings,
  title={Matryoshka Representation Learning},
  author={Kusupati, Aditya et al.},
  journal={NeurIPS 2022},
  year={2022}
}
```

**Citing QuicKB**
```bibtex
@software{Lucek_QuicKB_2024,
  author = {Łucek, Adam},
  title = {QuicKB: End-to-End Knowledge Base Processing Pipeline},
  url = {https://github.com/AdamLucek/quickb},
  version = {0.1},
  license = {MIT}
}
```

## Contributing

Contributions welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## License

MIT License - See [LICENSE](LICENSE)

## Acknowledgments

This project extends work from LangChain AI, Chroma Research, and Greg Kamradt's chunking methodologies. Special thanks to the Hugging Face team for their embedding training tools.
