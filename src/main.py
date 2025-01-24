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
from hub_upload.dataset_pusher import DatasetPusher

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class ChunkerConfig(BaseModel):
    path_to_knowledgebase: str = Field(..., description="Path to knowledge base directory")
    chunker: str = Field(..., description="Name of chunker to use")
    chunker_arguments: Dict[str, Any] = Field(default_factory=dict)
    output_path: str = Field(..., description="File path for saving the knowledgebase dataset")
    generate_questions: bool = Field(False, description="Enable synthetic question generation")
    question_output_path: Optional[str] = Field(
        None, description="File path for saving the training dataset (questions)"
    )
    deduplication: Optional[Dict[str, Any]] = Field(
        default_factory=lambda: {
            "enabled": True, 
            "similarity_threshold": 0.85
        },
        description="Configuration for question deduplication"
    )
    hub_username: Optional[str] = Field(None, description="Hugging Face username")
    hub_token: Optional[str] = Field(None, description="Hugging Face token (will use HF_TOKEN env var if not provided)")
    hub_private: bool = Field(True, description="Whether to create a private repository")
    train_embedding: bool = Field(False, description="Whether to train an embedding model or not")


def load_config(config_path: str) -> ChunkerConfig:
    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)
    return ChunkerConfig(**raw_config)


def process_files(config: ChunkerConfig):
    """
    Main function for:
      1) Chunking all .txt files in `config.path_to_knowledgebase`.
      2) Optionally generating synthetic questions.
      3) Saving two separate datasets:
         - Knowledgebase dataset (id, text, source)
         - Training dataset (anchor, positive, question_id, chunk_id)
    """

    # --------------------------------------------------------------------------
    # 1) GET THE CHUNKER CLASS
    # --------------------------------------------------------------------------
    try:
        chunker_class = ChunkerRegistry.get_chunker(config.chunker)
    except ValueError as e:
        logger.error(str(e))
        raise

    # Make a copy of the chunker arguments so we can safely modify them
    args = config.chunker_arguments.copy()

    # Resolve any string references to embedder or token counter
    for key in ['embedding_function', 'length_function']:
        if key in args and isinstance(args[key], str):
            resolver = (
                EmbeddingManager.get_embedder
                if 'embedding' in key
                else EmbeddingManager.get_token_counter
            )
            args[key] = resolver(args[key])

    # Instantiate the chunker
    chunker = chunker_class(**args)

    # --------------------------------------------------------------------------
    # 2) CHUNK ALL .TXT FILES
    # --------------------------------------------------------------------------
    base_path = Path(config.path_to_knowledgebase)
    results = []

    for file_path in base_path.rglob('*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunker.split_text(text)
            results.append({
                "path": str(file_path.relative_to(base_path)),
                "chunks": [
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk,
                    }
                    for chunk in chunks
                ],
            })
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    # --------------------------------------------------------------------------
    # 3) CREATE THE KNOWLEDGEBASE DATASET (id, text, source)
    #    This will be saved to config.output_path
    # --------------------------------------------------------------------------
    knowledgebase_records = []
    for file_entry in results:
        source_path = file_entry["path"]
        for chunk in file_entry["chunks"]:
            knowledgebase_records.append({
                "id": chunk["id"],
                "text": chunk["text"],
                "source": source_path
            })

    kb_output_path = Path(config.output_path)
    kb_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(kb_output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledgebase_records, f, indent=2, ensure_ascii=False)

    logger.info(f"Knowledgebase dataset saved to {config.output_path}")

    # --------------------------------------------------------------------------
    # 4) IF QUESTION GENERATION IS ENABLED, CREATE THE TRAINING DATASET
    #    (anchor, positive, question_id, chunk_id)
    #    This will be saved to config.question_output_path
    # --------------------------------------------------------------------------
    if config.generate_questions and config.question_output_path:

        # Build a text -> [ (chunk_id, source_file) ] map
        text_to_chunks_map = {}
        for file_entry in results:
            for chunk in file_entry["chunks"]:
                text_val = chunk["text"]
                if text_val not in text_to_chunks_map:
                    text_to_chunks_map[text_val] = []
                text_to_chunks_map[text_val].append({
                    "id": chunk["id"],
                    "source_file": file_entry["path"]
                })

        # We only need unique text values
        unique_texts = list(text_to_chunks_map.keys())

        # Create question generator with deduplication settings
        dedup_config = config.deduplication or {}
        generator = QuestionGenerator(
            prompt_path="src/prompts/question_generation.txt",
            api_key=os.getenv("OPENAI_API_KEY"),
            dedup_enabled=dedup_config.get("enabled", True),
            similarity_threshold=dedup_config.get("similarity_threshold", 0.92)
        )

        # Generate questions
        question_entries = generator.generate_for_chunks(unique_texts)
        original_question_count = sum(len(questions) for questions in generator._question_cache.values())

        # Build the training dataset
        # anchor = question
        # positive = chunk text
        # question_id = q["id"]
        # chunk_id = chunk_info["id"]
        train_records = []
        for q in question_entries:
            chunk_text = q.get("chunk_text")
            anchor = q.get("question")
            question_id = q.get("id")
            if not (chunk_text and anchor and question_id):
                continue

            if chunk_text not in text_to_chunks_map:
                # Should never happen unless chunk_text changed
                logger.warning(f"Question generated for non-existent chunk: {chunk_text[:50]}...")
                continue

            # For each chunk that matches chunk_text, create a train record
            for chunk_info in text_to_chunks_map[chunk_text]:
                train_records.append({
                    "anchor": anchor,
                    "positive": chunk_text,
                    "question_id": question_id,
                    "chunk_id": chunk_info["id"]
                })

        # Save the training dataset
        train_output_path = Path(config.question_output_path)
        train_output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_records, f, indent=2, ensure_ascii=False)

        logger.info(f"Training dataset saved to {config.question_output_path}")

    kb_output_path = Path(config.output_path)
    kb_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(kb_output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledgebase_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Knowledgebase dataset saved to {config.output_path}")

    # After saving train_records to JSON (if generated)
    if config.generate_questions and config.question_output_path:
        train_output_path = Path(config.question_output_path)
        train_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(train_output_path, 'w', encoding='utf-8') as f:
            json.dump(train_records, f, indent=2, ensure_ascii=False)
        logger.info(f"Training dataset saved to {config.question_output_path}")

    # Upload to Hub if username is provided
    if config.hub_username:
        try:
            # Initialize pusher
            pusher = DatasetPusher(
                username=config.hub_username,
                token=config.hub_token
            )
            
            # Get repository name from output path if not specified
            repository_name = Path(config.output_path).stem
            
            # Collect chunker info
            chunker_info = {
                'chunker_name': config.chunker,
                'chunker_params': config.chunker_arguments
            }
            
            # Collect question generation info if enabled
            question_gen_info = None
            if config.generate_questions:
                question_gen_info = {
                    'model_name': "gpt-4o-mini",
                    'similarity_threshold': config.deduplication.get('similarity_threshold', 0.85),
                    'num_questions': original_question_count,  # Original count
                    'num_deduped': len(train_records)          # After deduplication
                }
            
            # Push dataset
            pusher.push_dataset(
                repository_name=repository_name,
                knowledgebase_path=config.output_path,
                chunker_info=chunker_info,
                train_path=config.question_output_path if config.generate_questions else None,
                question_gen_info=question_gen_info,
                private=config.hub_private
            )
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face Hub: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting chunking process...")
    try:
        config = load_config("config.yaml")
        process_files(config)
        logger.info("Done chunking")

      # If we want to automatically train embeddings:
        from training.train import main as train_main
        if config.train_embedding:
            logger.info("Starting embedding training pipeline...")
            train_main("config.yaml")
            logger.info("Embedding training complete!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise