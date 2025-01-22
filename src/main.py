import yaml
import json
import logging
import os
import uuid
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

from chunking import ChunkerRegistry
from embeddings.base_embedder import EmbeddingManager
from synth_dataset.question_generator import QuestionGenerator

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class ChunkerConfig(BaseModel):
    path_to_knowledgebase: str = Field(..., description="Path to knowledge base directory")
    chunker: str = Field(..., description="Name of chunker to use")
    chunker_arguments: Dict[str, Any] = Field(default_factory=dict)
    output_path: str = Field(..., description="Output file path")
    generate_questions: bool = Field(False, description="Enable synthetic question generation")
    question_output_path: Optional[str] = Field(None, description="Output path for questions")

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

    # Handle string-based references to embedder / token counters if present
    for key in ['embedding_function', 'length_function']:
        if key in args and isinstance(args[key], str):
            resolver = (EmbeddingManager.get_embedder
                        if 'embedding' in key else
                        EmbeddingManager.get_token_counter)
            args[key] = resolver(args[key])

    chunker = chunker_class(**args)
    base_path = Path(config.path_to_knowledgebase)
    results = []

    # Update the file processing loop
    for file_path in base_path.rglob('*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunker.split_text(text)
            results.append({
                "path": str(file_path.relative_to(base_path)),
                "chunks": [
                    {
                        "id": str(uuid.uuid4()),  # Add UUID for each chunk
                        "text": chunk,
                    } for chunk in chunks
                ],
                "count": len(chunks)
            })
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # After chunk generation
    # In process_files() where you generate questions:
    if config.generate_questions:
        # Create mapping from chunk text to list of (chunk_id, source_file)
        text_to_chunks = {}
        for file in results:
            for chunk in file["chunks"]:
                text = chunk['text']
                if text not in text_to_chunks:
                    text_to_chunks[text] = []
                text_to_chunks[text].append({
                    "id": chunk['id'],
                    "source_file": file["path"]
                })

        # Generate questions for unique chunks only
        unique_texts = list(text_to_chunks.keys())
        generator = QuestionGenerator(
            prompt_path="src/prompts/question_generation.txt",
            api_key=os.getenv("OPENAI_API_KEY")
        )
        questions = generator.generate_for_chunks(unique_texts)

        # Build question data with chunk IDs
        question_data = []
        for q in questions:
            chunk_text = q.get("chunk_text")
            if not chunk_text:
                logger.warning("Question missing 'chunk_text' field")
                continue
            if chunk_text not in text_to_chunks:
                logger.warning(f"Question generated for non-existent chunk: {chunk_text[:50]}...")
                continue
            
            # Create a question entry for each matching chunk
            for chunk_info in text_to_chunks[chunk_text]:
                question_data.append({
                    "question_id": q["id"],
                    "chunk_id": chunk_info["id"],
                    "question": q["question"],
                    "answer_location": q["answer_location"],
                    "source_file": chunk_info["source_file"]
                })

        output_path = Path(config.question_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(question_data, f, indent=2, ensure_ascii=False)

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