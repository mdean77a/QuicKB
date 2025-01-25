from typing import List, Dict, Optional, Any
import logging
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import os
import json
import uuid
from enum import Enum, auto

from chunking import ChunkerRegistry
from embeddings.base_embedder import EmbeddingManager
from synth_dataset.question_generator import QuestionGenerator
from hub_upload.dataset_pusher import DatasetPusher

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class PipelineStage(Enum):
    CHUNK = auto()
    GENERATE = auto()
    TRAIN = auto()
    UPLOAD = auto()

class TrainingConfig(BaseModel):
    model_id: str = "nomic-ai/modernbert-embed-base"
    output_dir: str
    epochs: int = 4
    learning_rate: float = 2.0e-5
    matryoshka_dimensions: list[int] = [768, 512, 256, 128, 64]
    batch_size: int = 32
    gradient_accumulation_steps: int = 16
    metric_for_best_model: str = "eval_dim_128_cosine_ndcg@10"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

class DeduplicationConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.85

class PipelineConfig(BaseModel):
    pipeline: Dict[str, str] = Field(..., description="Contains from_stage and to_stage")
    
    # Document chunking
    path_to_knowledgebase: str
    chunker: str
    chunker_arguments: Dict[str, Any] = Field(default_factory=dict)
    output_path: str

    # Synthetic question generation
    question_output_path: Optional[str]
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)

    # Hugging Face Hub
    hub_username: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private: bool = True

    # Training
    training: Optional[TrainingConfig] = None

@field_validator('chunker')
@classmethod
def validate_chunker_type(cls, v, info):
    # Only validate if we're doing chunking (no existing chunks)
    if not os.path.exists(info.data.get('output_path', '')):
        if v not in ChunkerRegistry._chunkers:
            raise ValueError(f"Unknown chunker: {v}")
    return v

def load_pipeline_config(config_path: str = "config.yaml") -> PipelineConfig:
    """Load and validate pipeline configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        raw_config = yaml.safe_load(f)
    return PipelineConfig(**raw_config)

def process_chunks(config: PipelineConfig) -> List[Dict[str, Any]]:
    """Process documents into chunks."""
    from chunking import ChunkerRegistry
    
    # Get chunker class
    chunker_class = ChunkerRegistry.get_chunker(config.chunker)
    
    # Make a copy of chunker arguments
    args = config.chunker_arguments.copy()
    
    # Resolve embedder or token counter references
    for key in ['embedding_function', 'length_function']:
        if key in args and isinstance(args[key], str):
            from embeddings.base_embedder import EmbeddingManager
            resolver = (
                EmbeddingManager.get_embedder 
                if 'embedding' in key 
                else EmbeddingManager.get_token_counter
            )
            args[key] = resolver(args[key])
    
    # Initialize chunker
    chunker = chunker_class(**args)
    
    # Process files
    base_path = Path(config.path_to_knowledgebase)
    results = []
    
    for file_path in base_path.rglob('*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunker.split_text(text)
            source_path = str(file_path.relative_to(base_path))
            
            # Create records for each chunk
            for chunk in chunks:
                results.append({
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "source": source_path
                })
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Save results
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return results

def generate_questions(
    config: PipelineConfig, 
    kb_dataset: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    from synth_dataset.question_generator import QuestionGenerator
    import logging
    
    logger = logging.getLogger(__name__)
    
    generator = QuestionGenerator(
        prompt_path="src/prompts/question_generation.txt",
        api_key=os.getenv("OPENAI_API_KEY"),
        dedup_enabled=config.deduplication.enabled,
        similarity_threshold=config.deduplication.similarity_threshold
    )
    
    # Get unique texts and build a text-to-id mapping
    text_to_chunk_map = {}
    for item in kb_dataset:
        text_val = item["text"]
        if text_val not in text_to_chunk_map:
            text_to_chunk_map[text_val] = []
        text_to_chunk_map[text_val].append(item["id"])
    
    unique_texts = list(text_to_chunk_map.keys())
    logger.info(f"Found {len(unique_texts)} unique chunks")
    
    # Generate questions
    questions = generator.generate_for_chunks(unique_texts)
    logger.info(f"Generated {len(questions)} questions after deduplication")
    
    # Track metrics
    metrics = {
        "num_questions_original": sum(len(generator._question_cache[chunk]) for chunk in generator._question_cache),
        "num_questions_deduped": len(questions)
    }
    logger.info(f"Question generation metrics: {metrics}")
    
    # Create training records with better error handling
    train_records = []
    skipped_questions = 0
    for q in questions:
        chunk_text = q.get("chunk_text")
        if not chunk_text:
            skipped_questions += 1
            continue
            
        chunk_ids = text_to_chunk_map.get(chunk_text, [])
        if not chunk_ids:
            skipped_questions += 1
            logger.warning(f"Could not find chunk_id for question: {q['question'][:100]}...")
            continue
            
        # Create a record for each matching chunk
        for chunk_id in chunk_ids:
            train_records.append({
                "anchor": q["question"],
                "positive": chunk_text,
                "question_id": q["id"],
                "chunk_id": chunk_id
            })
    
    logger.info(f"Created {len(train_records)} training records (skipped {skipped_questions} questions)")
    
    # Save results with error handling
    if config.question_output_path:
        output_path = Path(config.question_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(train_records, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved training records to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save training records: {str(e)}")
    
    return train_records, metrics

def train_model(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]],
    train_dataset: List[Dict[str, Any]]
):
    """Train the embedding model."""
    from training.train import main as train_main
    train_main(config)

def upload_to_hub(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]],
    train_dataset: Optional[List[Dict[str, Any]]] = None,
    question_metrics: Optional[Dict[str, int]] = None
):
    """Upload datasets to Hugging Face Hub."""
    from hub_upload.dataset_pusher import DatasetPusher
    
    if not config.hub_username:
        logger.warning("No 'hub_username' specified, skipping upload.")
        return
        
    try:
        # Initialize pusher
        pusher = DatasetPusher(
            username=config.hub_username,
            token=config.hub_token
        )
        
        # Get repository name from output path
        repository_name = Path(config.output_path).stem
        
        # Collect chunker info
        chunker_info = {
            'chunker_name': config.chunker,
            'chunker_params': config.chunker_arguments
        }
        
        # Collect question generation info if enabled
        question_gen_info = None
        question_gen_info = {
            'model_name': "gpt-4o-mini",
            'similarity_threshold': config.deduplication.similarity_threshold,
            'num_questions': question_metrics['num_questions_original'] if question_metrics else len(train_dataset),
            'num_deduped': question_metrics['num_questions_deduped'] if question_metrics else len(train_dataset)
        }
    
        # Push dataset
        pusher.push_dataset(
            repository_name=repository_name,
            knowledgebase_path=config.output_path,
            chunker_info=chunker_info,
            train_path=config.question_output_path if train_dataset else None,
            question_gen_info=question_gen_info,
            private=config.hub_private
        )
    except Exception as e:
        logger.error(f"Failed to upload to Hugging Face Hub: {str(e)}")

def run_pipeline(config: PipelineConfig):
    """
    Run the QuicKB pipeline from `from_stage` up to and including `to_stage`.
    The stages are defined by config.pipeline["from_stage"] and config.pipeline["to_stage"].
    """
    from_stage = PipelineStage[config.pipeline["from_stage"]]
    to_stage = PipelineStage[config.pipeline["to_stage"]]

    # 1. CHUNK
    if from_stage.value <= PipelineStage.CHUNK.value <= to_stage.value:
        logger.info("Running CHUNK stage.")
        kb_dataset = process_chunks(config)
    else:
        logger.info("Skipping CHUNK stage. Loading existing chunks.")
        with open(config.output_path, "r", encoding="utf-8") as f:
            kb_dataset = json.load(f)

    # 2. GENERATE
    train_dataset = None
    question_metrics = None
    if from_stage.value <= PipelineStage.GENERATE.value <= to_stage.value:
        logger.info("Running GENERATE stage.")
        train_dataset, question_metrics = generate_questions(config, kb_dataset)
    else:
        logger.info("Skipping GENERATE stage.")
        # If we skip generate, we can load from disk if we need it for TRAIN
        if (to_stage.value >= PipelineStage.TRAIN.value) and config.question_output_path:
            with open(config.question_output_path, "r", encoding="utf-8") as f:
                train_dataset = json.load(f)

    # 3. TRAIN
    if from_stage.value <= PipelineStage.TRAIN.value <= to_stage.value:
        logger.info("Running TRAIN stage.")
        if not config.training:
            raise ValueError("No training config found, cannot run TRAIN stage.")
        train_model(config, kb_dataset, train_dataset)
    else:
        logger.info("Skipping TRAIN stage.")

    # 4. UPLOAD
    if from_stage.value <= PipelineStage.UPLOAD.value <= to_stage.value:
        logger.info("Running UPLOAD stage.")
        # Optionally only do it if there's a hub_username
        if config.hub_username:
            upload_to_hub(
                config,
                kb_dataset=kb_dataset,
                train_dataset=train_dataset,
                question_metrics=question_metrics
            )
        else:
            logger.info("No hub_username configured. Skipping UPLOAD.")
    else:
        logger.info("Skipping UPLOAD stage.")

    logger.info("Pipeline complete!")

def run_pipeline_from_stage(config: PipelineConfig, start_stage: str):
    """Run pipeline starting from specified stage."""
    if start_stage == PipelineStage.CHUNK:
        run_pipeline(config)
    elif start_stage == PipelineStage.GENERATE:
        # Load existing chunks and continue
        with open(config.output_path, 'r', encoding='utf-8') as f:
            kb_dataset = json.load(f)
        train_dataset = generate_questions(config, kb_dataset)
        
        if config.train_embedding:
            train_model(config, kb_dataset, train_dataset)
        if config.hub_username:
            upload_to_hub(config, kb_dataset, train_dataset)
            
    elif start_stage == PipelineStage.TRAIN:
        # Load existing data and train
        with open(config.output_path, 'r', encoding='utf-8') as f:
            kb_dataset = json.load(f)
        with open(config.question_output_path, 'r', encoding='utf-8') as f:
            train_dataset = json.load(f)
            
        train_model(config, kb_dataset, train_dataset)
        if config.hub_username:
            upload_to_hub(config, kb_dataset, train_dataset)
            
    elif start_stage == PipelineStage.UPLOAD:
        # Just upload existing data
        with open(config.output_path, 'r', encoding='utf-8') as f:
            kb_dataset = json.load(f)
        train_dataset = None
        if config.question_output_path:
            with open(config.question_output_path, 'r', encoding='utf-8') as f:
                train_dataset = json.load(f)
        upload_to_hub(config, kb_dataset, train_dataset)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        config = load_pipeline_config("config.yaml")
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise