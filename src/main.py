from typing import List, Dict, Optional, Any
import logging
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, field_validator
import os
import json
import uuid
from dataclasses import dataclass

from chunking import ChunkerRegistry
from embeddings.base_embedder import EmbeddingManager
from synth_dataset.question_generator import QuestionGenerator
from hub_upload.dataset_pusher import DatasetPusher

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

@dataclass
class PipelineStage:
    CHUNK = "chunk"
    GENERATE = "generate"
    TRAIN = "train"
    UPLOAD = "upload"

class TrainingConfig(BaseModel):
    model_id: str = "nomic-ai/modernbert-embed-base"
    output_dir: str = Field(..., description="Directory for model outputs")
    epochs: int = 4
    learning_rate: float = 2.0e-5
    matryoshka_dimensions: List[int] = [768, 512, 256, 128, 64]
    batch_size: int = 32
    gradient_accumulation_steps: int = 16
    metric_for_best_model: str = "eval_dim_128_cosine_ndcg@10"
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

class DeduplicationConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.85

class PipelineConfig(BaseModel):
    # Core settings
    path_to_knowledgebase: str = Field(..., description="Path to knowledge base directory")
    chunker: str = Field(..., description="Name of chunker to use")
    chunker_arguments: Dict[str, Any] = Field(default_factory=dict)
    output_path: str = Field(..., description="File path for saving the knowledgebase dataset")
    
    # Question generation settings
    generate_questions: bool = False
    question_output_path: Optional[str] = None
    deduplication: DeduplicationConfig = Field(default_factory=DeduplicationConfig)

    # Hub settings
    hub_username: Optional[str] = None
    hub_token: Optional[str] = None
    hub_private: bool = True

    # Training settings
    train_embedding: bool = False
    training: Optional[TrainingConfig] = None

@field_validator('training')
@classmethod
def validate_training_config(cls, v, info):
    if info.data.get('train_embedding', False):
        if v is None:
            raise ValueError("Training configuration required when train_embedding is True")
    return v

@field_validator('question_output_path')
@classmethod
def validate_question_path(cls, v, info):
    if info.data.get('generate_questions', False) and not v:
        raise ValueError("question_output_path required when generate_questions is True")
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
    """Generate synthetic questions from chunks."""
    from synth_dataset.question_generator import QuestionGenerator
    
    # Create question generator
    generator = QuestionGenerator(
        prompt_path="src/prompts/question_generation.txt",
        api_key=os.getenv("OPENAI_API_KEY"),
        dedup_enabled=config.deduplication.enabled,
        similarity_threshold=config.deduplication.similarity_threshold
    )
    
    # Get unique texts
    unique_texts = list({item["text"] for item in kb_dataset})
    
    # Generate questions
    questions = generator.generate_for_chunks(unique_texts)
    
    # Track metrics
    metrics = {
        "num_questions_original": sum(len(generator._question_cache[chunk]) for chunk in generator._question_cache),
        "num_questions_deduped": len(questions)
    }
    
    # Create training records
    train_records = []
    for q in questions:
        train_records.append({
            "anchor": q["question"],
            "positive": q["chunk_text"],
            "question_id": q["id"],
            "chunk_id": next(
                item["id"] for item in kb_dataset 
                if item["text"] == q["chunk_text"]
            )
        })
    
    # Save results
    if config.question_output_path:
        output_path = Path(config.question_output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(train_records, f, indent=2, ensure_ascii=False)
    
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
        if config.generate_questions and train_dataset:
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

def validate_pipeline_config(config: PipelineConfig):
    """Validate pipeline configuration and dependencies."""
    if config.train_embedding and not config.generate_questions:
        raise ValueError(
            "Training requires synthetic question generation. "
            "Please enable generate_questions or disable train_embedding."
        )
    
    if config.generate_questions and not config.question_output_path:
        raise ValueError("Question generation enabled but no output path specified")

def run_pipeline(config: PipelineConfig):
    """Run the QuicKB pipeline with validated configuration."""
    logger.info("Starting QuicKB pipeline...")
    
    try:
        # Step 1: Chunking (Always required)
        logger.info("Starting document chunking...")
        kb_dataset = process_chunks(config)
        logger.info(f"Created {len(kb_dataset)} chunks")
        
        # Step 2: Question Generation (Optional)
        train_dataset = None
        question_metrics = None
        if config.generate_questions:
            logger.info("Generating synthetic questions...")
            train_dataset, question_metrics = generate_questions(config, kb_dataset)
            logger.info(f"Generated {len(train_dataset)} training examples")
        
        # Step 3: Embedding Training (Optional)
        if config.train_embedding:
            if train_dataset is None:
                raise RuntimeError("Training enabled but no training data available")
            logger.info("Training embedding model...")
            train_model(config, kb_dataset, train_dataset)
            
        # Optional: Upload to Hub
        if config.hub_username:
            logger.info("Uploading to Hugging Face Hub...")
            upload_to_hub(config, kb_dataset, train_dataset, question_metrics)
            
        logger.info("Pipeline complete!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

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
        validate_pipeline_config(config)
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise