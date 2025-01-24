# QuicKB: End-to-End Knowledge Base Processing Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)

> From raw text to optimized embeddings in a single workflow

## Overview

QuicKB provides a complete pipeline for transforming unstructured text documents into:

1. **Optimized Chunked Knowledge Bases**  
2. **Synthetic Training Data**  
3. **Fine-tuned Embedding Models**  

## Key Features

**Core Pipeline**
- 6 available chunking strategies with hybrid approaches
- Synthetic question generation
- Retrieval model training
- Matryoshka dimension training (768→64D)
- Built-in Hugging Face Hub integration

## Installation

```bash
git clone https://github.com/AdamLucek/quickb.git
cd quickb

python -m venv quickb-env
source quickb-env/bin/activate  # Windows: quickb-env\Scripts\activate

pip install -e .
```

## Configuration (`config.yaml`)

```yaml
# Full Pipeline Example
path_to_knowledgebase: "./testing/knowledgebase"
chunker: "RecursiveTokenChunker"
chunker_arguments:
  chunk_size: 400
  chunk_overlap: 0
  separators: ["\n\n", "\n", ".", "?", "!", " ", ""]
  keep_separator: true
  is_separator_regex: false
  length_function: "character"
output_path: "./output/knowledgebase-quickb-legal.json"
generate_questions: true
question_output_path: "./output/train_data.json"
deduplication:
  enabled: true
  similarity_threshold: 0.8

hub_username: AdamLucek  
hub_token: null     
hub_private: false

train_embedding: true
training:
  model_id: "nomic-ai/modernbert-embed-base"
  output_dir: "./output/modernbert_legal_mtl"
  epochs: 4
  learning_rate: 2.0e-5
  matryoshka_dimensions: [768, 512, 256, 128, 64]
  batch_size: 32
  gradient_accumulation_steps: 16
  metric_for_best_model: "eval_dim_128_cosine_ndcg@10"
  push_to_hub: true
  hub_model_id: "AdamLucek/modernbert-embed-base-quickb-legal"
```

## Usage

**Full Pipeline Execution**
```bash
python src/main.py
```

**Key Components**
1. **Chunking** (`src/chunking/`)
   - Registry pattern for custom implementations
   - Hybrid semantic+token-based approaches

2. **Training** (`src/training/`)
   - Multi-stage Matryoshka training
   - Automatic metric tracking
   - Hugging Face integration

3. **Hub Upload** (`src/hub_upload/`)
   - Dataset cards with chunking metadata
   - Model cards with training dimensions

## Output Artifacts

**Knowledge Base** (`output/knowledgebase.json`)
```json
{
  "id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "text": "Section 12.1: Termination clauses...",
  "source": "docs/contracts/2024/Q1-agreement.txt"
}
```

**Training Data** (`output/train_data.json`)
```json
{
  "anchor": "What are the termination notice requirements?",
  "positive": "Section 12.1: Either party may terminate...",
  "question_id": "a3b8c7d0-e83a-4b5c-b12d-3f7a8d4c9e1b",
  "chunk_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6"
}
```

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

## License

MIT License - See [LICENSE](LICENSE)

---

**Roadmap**
XXX

**Acknowledgments**  
This project extends work from LangChain AI, Chroma Research, and Greg Kamradt's chunking methodologies. Special thanks to the Hugging Face team for their embedding training tools.