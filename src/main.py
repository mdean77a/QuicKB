import yaml
import json
import os
from pathlib import Path
from src.chunking import (
    ClusterSemanticChunker,
    LLMSemanticChunker,
    FixedTokenChunker,
    RecursiveTokenChunker,
    KamradtModifiedChunker
)
from src.chunking.utils import EmbeddingManager

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def process_files(config: dict):
    # Initialize chunker with automatic embedding/token resolution
    chunker_class = globals()[config['chunker']]
    args = config.get('chunker_arguments', {})
    
    # Resolve string-based embedders/counters
    for key in ['embedding_function', 'length_function']:
        if key in args and isinstance(args[key], str):
            resolver = EmbeddingManager.get_embedder if 'embedding' in key else EmbeddingManager.get_token_counter
            args[key] = resolver(args[key])
    
    chunker = chunker_class(**args)
    
    # Process files
    base_path = Path(config['path_to_knowledgebase'])
    results = []
    
    for file_path in base_path.rglob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = chunker.split_text(f.read())
            
        results.append({
            "path": str(file_path.relative_to(base_path)),
            "chunks": chunks,
            "count": len(chunks)
        })
    
    # Save output
    output_path = Path(config['output_path'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    config = load_config("config.yaml")
    process_files(config)
    print(f"Chunking complete! Results saved to {config['output_path']}")