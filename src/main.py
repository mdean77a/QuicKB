import yaml
import json
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any
from chunking import ChunkerRegistry
from embeddings.base_embedder import EmbeddingManager

logger = logging.getLogger(__name__)

class ChunkerConfig(BaseModel):
    path_to_knowledgebase: str = Field(..., description="Path to knowledge base directory")
    chunker: str = Field(..., description="Name of chunker to use")
    chunker_arguments: Dict[str, Any] = Field(default_factory=dict)
    output_path: str = Field(..., description="Output file path")

def load_config(config_path: str) -> ChunkerConfig:
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    return ChunkerConfig(**raw_config)

def process_files(config: ChunkerConfig):
    try:
        chunker_class = ChunkerRegistry.get_chunker(config.chunker)
    except ValueError as e:
        logger.error(str(e))
        raise

    args = config.chunker_arguments.copy()
    
    for key in ['embedding_function', 'length_function']:
        if key in args and isinstance(args[key], str):
            resolver = EmbeddingManager.get_embedder if 'embedding' in key else EmbeddingManager.get_token_counter
            args[key] = resolver(args[key])
    
    chunker = chunker_class(**args)
    base_path = Path(config.path_to_knowledgebase)
    results = []
    
    for file_path in base_path.rglob('*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunker.split_text(text)
            results.append({
                "path": str(file_path.relative_to(base_path)),
                "chunks": chunks,
                "count": len(chunks)
            })
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting chunking process...")
    try:
        config = load_config("config.yaml")
        process_files(config)
        logger.info(f"Chunking complete! Results saved to {config.output_path}")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise