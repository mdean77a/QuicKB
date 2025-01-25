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
    push_to_hub: bool = False
    
    # Training settings
    train_embedding: bool = False
    training: Optional[TrainingConfig] = None

    # Field to indicate using existing chunks
    use_existing_chunks: bool = False
    
    @field_validator('output_path')
    @classmethod
    def validate_chunks(cls, v, info):
        if info.data.get('use_existing_chunks', False):
            if not os.path.exists(v):
                raise ValueError(f"Specified existing chunks file {v} does not exist")
            # Verify it's valid JSON with expected structure
            try:
                with open(v, 'r') as f:
                    chunks = json.load(f)
                if not isinstance(chunks, list) or not all('text' in c for c in chunks):
                    raise ValueError("Existing chunks file has invalid format")
            except json.JSONDecodeError:
                raise ValueError("Existing chunks file is not valid JSON")
        return v

@field_validator('training')
@classmethod
def validate_training_config(cls, v, info):
    if info.data.get('train_embedding', False):
        if v is None:
            raise ValueError("Training configuration required when train_embedding is True")
        # Check if required files exist
        if not os.path.exists(info.data.get('output_path', '')):
            raise ValueError("Existing knowledgebase required for training")
        if not os.path.exists(info.data.get('question_output_path', '')):
            raise ValueError("Existing training data required for training")
    return v

@field_validator('question_output_path')
@classmethod
def validate_question_path(cls, v, info):
    if info.data.get('generate_questions', False):
        if not v:
            raise ValueError("question_output_path required when generate_questions is True")
        # If we're only generating questions, check if input chunks exist
        if os.path.exists(info.data.get('output_path', '')):
            # Verify it's valid JSON with expected structure
            try:
                with open(info.data.get('output_path'), 'r') as f:
                    chunks = json.load(f)
                if not isinstance(chunks, list) or not all('text' in c for c in chunks):
                    raise ValueError("Existing knowledgebase has invalid format")
            except json.JSONDecodeError:
                raise ValueError("Existing knowledgebase is not valid JSON")
    return v

@field_validator('hub_username')
@classmethod
def validate_hub_upload(cls, v, info):
    if v:
        # Check if we have required files for upload
        if not os.path.exists(info.data.get('output_path', '')):
            raise ValueError("Knowledgebase file required for Hub upload")
        if info.data.get('generate_questions') and not os.path.exists(info.data.get('question_output_path', '')):
            raise ValueError("Training data file required for Hub upload when generate_questions is True")
    return v

@field_validator('path_to_knowledgebase')
@classmethod
def validate_input_path(cls, v, info):
    # Only check if we're doing chunking (no existing chunks)
    if not os.path.exists(info.data.get('output_path', '')):
        if not os.path.exists(v):
            raise ValueError(f"Input directory {v} does not exist")
        if not any(Path(v).rglob('*.txt')):
            raise ValueError(f"No .txt files found in {v}")
    return v

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


def run_pipeline(
    config: PipelineConfig,
    from_stage: PipelineStage = PipelineStage.CHUNK,
    to_stage: PipelineStage = PipelineStage.UPLOAD
):
    """
    Run the QuicKB pipeline from `from_stage` up to and including `to_stage`.
    
    Stages are:
    1) CHUNK        -> Convert input text files into chunked JSON
    2) GENERATE     -> Generate synthetic QnA from chunks
    3) TRAIN        -> Fine-tune an embedding model using QnA
    4) UPLOAD       -> Upload dataset (and optionally model) to HF Hub
    
    This function avoids re-running unnecessary steps and gracefully handles
    partial pipeline runs if the relevant data already exists on disk.
    """
    # -------------------------------------------------------------------------
    # 1. CHUNKING or LOADING EXISTING CHUNKS
    # -------------------------------------------------------------------------
    kb_dataset = None
    
    # If we are skipping chunk stage (from_stage > CHUNK or user wants partial pipeline),
    # ensure we have an existing chunks file to load
    if from_stage.value > PipelineStage.CHUNK.value:
        # We are not chunking, so we must load existing chunks
        logger.info("Skipping CHUNK stage. Loading existing chunks.")
        if not Path(config.output_path).exists():
            raise RuntimeError(
                f"Cannot skip chunking: existing chunks file {config.output_path} not found."
            )
        with open(config.output_path, "r", encoding="utf-8") as f:
            kb_dataset = json.load(f)
        logger.info(f"Loaded {len(kb_dataset)} chunks from {config.output_path}")
    else:
        # We are at or before the chunk stage
        if config.use_existing_chunks and Path(config.output_path).exists():
            logger.info("use_existing_chunks=True, loading chunks from disk.")
            with open(config.output_path, "r", encoding="utf-8") as f:
                kb_dataset = json.load(f)
            logger.info(f"Loaded {len(kb_dataset)} chunks from {config.output_path}")
        else:
            logger.info("Running CHUNK stage (creating chunks).")
            kb_dataset = process_chunks(config)
            logger.info(f"Created {len(kb_dataset)} chunks.")
    
    # If the last stage we want to run is CHUNK, we're done.
    if to_stage == PipelineStage.CHUNK:
        logger.info("Ending pipeline after CHUNK stage.")
        return

    # -------------------------------------------------------------------------
    # 2. QUESTION GENERATION
    # -------------------------------------------------------------------------
    train_dataset = None
    question_metrics = None
    
    if from_stage.value > PipelineStage.GENERATE.value:
        # Skipping question generation, so load existing training data if we plan to do training next
        logger.info("Skipping GENERATE stage. Checking if existing QnA data is needed.")
        if config.train_embedding or to_stage.value >= PipelineStage.TRAIN.value:
            # We'll need question data for training
            if not Path(config.question_output_path).exists():
                raise RuntimeError(
                    f"Cannot skip question generation: no existing training data {config.question_output_path} found."
                )
            with open(config.question_output_path, "r", encoding="utf-8") as f:
                train_dataset = json.load(f)
            logger.info(f"Loaded {len(train_dataset)} QnA records from {config.question_output_path}")
    else:
        # We are at or before the GENERATE stage
        if config.generate_questions:
            logger.info("Running GENERATE stage (synthetic QnA creation).")
            train_dataset, question_metrics = generate_questions(config, kb_dataset)
            logger.info(f"Generated {len(train_dataset)} QnA records.")
        else:
            logger.info("generate_questions=False, skipping QnA creation.")
            # If we skip generating but still want to train or upload,
            # we must have an existing training file.
            if config.train_embedding or to_stage == PipelineStage.TRAIN:
                if not Path(config.question_output_path).exists():
                    raise RuntimeError(
                        "Cannot train without question data. Either enable generate_questions or provide existing QnA JSON."
                    )
                with open(config.question_output_path, "r", encoding="utf-8") as f:
                    train_dataset = json.load(f)
                logger.info(f"Loaded {len(train_dataset)} QnA records from {config.question_output_path}")
    
    # If the last stage we want to run is GENERATE, we can optionally upload or just end here:
    if to_stage == PipelineStage.GENERATE:
        logger.info("Ending pipeline after GENERATE stage.")
        return

    # -------------------------------------------------------------------------
    # 3. TRAIN EMBEDDING MODEL
    # -------------------------------------------------------------------------
    if from_stage.value > PipelineStage.TRAIN.value:
        # Skipping training stage. Possibly going directly to upload stage
        logger.info("Skipping TRAIN stage.")
    else:
        # We are at or before the TRAIN stage
        if config.train_embedding:
            logger.info("Running TRAIN stage (embedding model fine-tuning).")
            if not train_dataset:
                raise RuntimeError("Cannot train embedding model: no training data available.")
            train_model(config, kb_dataset, train_dataset)
        else:
            logger.info("train_embedding=False, skipping training stage.")
    
    if to_stage == PipelineStage.TRAIN:
        logger.info("Ending pipeline after TRAIN stage.")
        return

    # -------------------------------------------------------------------------
    # 4. UPLOAD TO HUB
    # -------------------------------------------------------------------------
    # If from_stage.value > PipelineStage.UPLOAD.value, skip uploading altogether
    if from_stage.value > PipelineStage.UPLOAD.value:
        logger.info("Skipping UPLOAD stage. Pipeline complete.")
        return
    
    if config.push_to_hub and config.hub_username:
        logger.info("Running UPLOAD stage (pushing dataset and/or model to HF Hub).")
        # If we still don't have `train_dataset` but the user wants to upload them, load from disk
        if config.generate_questions and train_dataset is None:
            with open(config.question_output_path, "r", encoding="utf-8") as f:
                train_dataset = json.load(f)
        
        upload_to_hub(
            config,
            kb_dataset=kb_dataset,
            train_dataset=train_dataset,
            question_metrics=question_metrics
        )
    else:
        logger.info("push_to_hub=False or no hub_username provided; skipping upload.")

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
        validate_pipeline_config(config)
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise