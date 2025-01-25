# QuicKB 

<img src="qkb_logo.png" width=175>

Optimize Document Retrieval with Fine-tuned Knowledge Bases

## Overview

QuicKB optimizes document retrieval by creating fine-tuned knowledge bases through an end-to-end pipeline that handles document chunking, training data generation, and embedding model optimization.

## Key Features

### Document Chunking
- Token-based strategies:
  - **RecursiveTokenChunker**: Hierarchical splitting with custom separators
  - **FixedTokenChunker**: Precise length-based chunking
- Semantic strategies:
  - **LLMSemanticChunker**: Natural language break points using LLMs
  - **ClusterSemanticChunker**: Content-aware semantic grouping
  - **KamradtModifiedChunker**: Hybrid semantic-token approach

*Chunking techniques adapted from [ChromaDB's Chunking Research](https://research.trychroma.com/evaluating-chunking)*

### Training Data Generation
- LLM-powered question generation
- Cloud and local model support via [LiteLLM](https://docs.litellm.ai/docs/)
- Semantic deduplication
- Parallel processing

### Embedding Optimization
- Fine-tuned retrieval models with [Sentence Transformers](https://sbert.net/)
- Matryoshka dimension reduction (768→64D)
- Comprehensive evaluation metrics

## Installation

```bash
git clone https://github.com/ALucek/QuicKB.git
cd QuicKB

python -m venv quickb-env
source quickb-env/bin/activate  # Windows: quickb-env\Scripts\activate

pip install -e .
```

## Usage

1. Prepare your text documents in a directory
2. Configure the pipeline in `config.yaml`
3. Run:
```bash
python src/main.py
```
4. Enjoy!

## Configuration Guide

The pipeline is controlled through a single `config.yaml` file. Here's a complete configuration example:

```yaml
# Pipeline Stage Control
pipeline:
  from_stage: "CHUNK"    # Options: CHUNK, GENERATE, TRAIN
  to_stage: "TRAIN"

# Path to Knowledgebase Directory
path_to_knowledgebase: "./testing/knowledgebase"

# Base HF Hub credentials
hub_username: "AdamLucek"
hub_token: null  # or rely on HF_TOKEN environment variable

# Chunking config with upload options
chunker_config:
  output_path: "./output/knowledgebase-quickb.json"
  chunker: "RecursiveTokenChunker"
  chunker_arguments:
    chunk_size: 400
    chunk_overlap: 0
    separators: ["\n\n", "\n", ".", "?", "!", " ", ""]
    keep_separator: true
    is_separator_regex: false
    length_type: "character"
  
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_dataset_id: "AdamLucek/quickb"

# Question Generation
question_generation:
  output_path: "./output/train_data.json"
  litellm_config:
    model: "openai/gpt-4o-mini"
    model_api_base: null
    embedding_model: "text-embedding-3-large"
    embedding_api_base: null
  max_workers: 20
  deduplication_enabled: true
  similarity_threshold: 0.85

  upload_config:
    push_to_hub: true
    hub_private: false
    hub_dataset_id: "AdamLucek/quickb"

# Training config
training:
  model_settings:
    model_id: "nomic-ai/modernbert-embed-base"
    matryoshka_dimensions: [768, 512, 256, 128, 64]
    metric_for_best_model: "eval_dim_128_cosine_ndcg@10"
  
  training_arguments:
    output_path: "./output/modernbert_mtl"
    epochs: 4
    learning_rate: 2.0e-5
    batch_size: 32
    gradient_accumulation_steps: 16
    warmup_ratio: 0.1
    lr_scheduler_type: "cosine"
    optim: "adamw_torch_fused"
    tf32: true
    bf16: true
    batch_sampler: "no_duplicates"
    eval_strategy: "epoch"
    save_strategy: "epoch"
    logging_steps: 10
    save_total_limit: 3
    load_best_model_at_end: true
    report_to: "none"
    
  upload_config:
    push_to_hub: true
    hub_private: false
    hub_model_id: "AdamLucek/modernbert-embed-quickb"
```

### Alternative Chunker Configurations

1. **Fixed Token Chunker**
```yaml
chunker_config:
  output_path: "./output/fixed_token_chunks.json"
  chunker: "FixedTokenChunker"
  chunker_arguments:
    encoding_name: "cl100k_base"
    model_name: "text-embedding-3-large"  # Only needed if using model-specific tokenizer
    chunk_size: 400
    chunk_overlap: 50
    length_type: "token"
```

2. **Cluster Semantic Chunker**
```yaml
chunker_config:
  output_path: "./output/semantic_clusters.json"
  chunker: "ClusterSemanticChunker"
  chunker_arguments:
    max_chunk_size: 400    # Max tokens after clustering
    min_chunk_size: 50     # Initial split size
    length_type: "token"
    litellm_config:
      embedding_model: "text-embedding-3-large"  # Required for semantic clustering
```

3. **LLM Semantic Chunker**
```yaml
chunker_config:
  output_path: "./output/llm_semantic_chunks.json"
  chunker: "LLMSemanticChunker"
  chunker_arguments:
    length_type: "token"
    litellm_config:
      model: "openai/gpt-4o"  # LLM for split decisions
```

4. **Kamradt Modified Semantic Chunker**
```yaml
chunker_config:
  output_path: "./output/kamradt_chunks.json"
  chunker: "KamradtModifiedChunker"
  chunker_arguments:
    avg_chunk_size: 400    # Target average size
    min_chunk_size: 50     # Minimum initial split
    length_type: "token"
    litellm_config: 
      embedding_model: "text-embedding-3-large"  # For similarity calculations
```

### LiteLLM Integration

QuicKB uses [LiteLLM](https://docs.litellm.ai/docs/) for flexible model provider integration, allowing you to use any supported LLM or embedding provider for question generation and semantic chunking. This enables both cloud-based and local model deployment.

The LiteLLM configuration is managed through the `litellm_config` section in both the chunking and question generation configurations:

```yaml
litellm_config:
  model: "openai/gpt-4o"                     # LLM model identifier
  model_api_base: null                       # Optional API base URL for LLM
  embedding_model: "text-embedding-3-large"  # Embedding model identifier
  embedding_api_base: null                   # Optional API base URL for embeddings
```

**Using Local Models**:

1. Set up an OpenAI API compatible endpoint (e.g., Ollama, vLLM)
2. Configure the `model_api_base` or `embedding_api_base` in your config
3. Use the appropriate model identifier format

Example local setup:

```yaml
# For question generation
question_generation:
  litellm_config:
    model: "local/llama-7b"
    model_api_base: "http://localhost:8000"
    embedding_model: "local/bge-small"
    embedding_api_base: "http://localhost:8000"

# For semantic chunkers
chunker_config:
  chunker: "ClusterSemanticChunker"  # or other semantic chunkers
  chunker_arguments:
    litellm_config:
      model: "local/llama-7b"
      model_api_base: "http://localhost:8000"
      embedding_model: "local/bge-small"
      embedding_api_base: "http://localhost:8000"
```

For more details on setting up local models and supported providers, refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).

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

- `<PROVIDER>_API_KEY`: Required for LLM embeddings, question generation, and chunking
- `HF_TOKEN`: Required for Hugging Face Hub uploads

## Citations

QuicKB builds upon these foundational works:

ChromaDB: [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)
```bibtex
@techreport{smith2024evaluating,
  title = {Evaluating Chunking Strategies for Retrieval},
  author = {Smith, Brandon and Troynikov, Anton},
  year = {2024},
  month = {July},
  institution = {Chroma},
  url = {https://research.trychroma.com/evaluating-chunking},
}
```

Sentence Transformers
```bibtext
@inproceedings{reimers-2019-sentence-bert,
  title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
  author = "Reimers, Nils and Gurevych, Iryna",
  booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
  month = "11",
  year = "2019",
  publisher = "Association for Computational Linguistics",
  url = "https://arxiv.org/abs/1908.10084",
}
```

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

Todo List:

- pydantic v2 fields warning
- Transformers verions with torch handling
- Custom Model Card (Using base from SBERT currently)
- CPU training support

## License

MIT License - See [LICENSE](LICENSE)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
