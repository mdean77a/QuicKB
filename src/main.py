import logging
import os
import json
import uuid
from enum import Enum, auto
from pathlib import Path
from typing import List, Dict, Optional, Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator
from datasets import load_dataset, Dataset

from chunking import ChunkerRegistry
from hub_upload.dataset_pusher import DatasetPusher
from synth_dataset.question_generator import QuestionGenerator
from training.train import main as train_main

logger = logging.getLogger(__name__)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

class PipelineStage(Enum):
    CHUNK = auto()
    GENERATE = auto()
    TRAIN = auto()

class BatchSamplers(str, Enum):
    BATCH_SAMPLER = "batch_sampler"
    NO_DUPLICATES = "no_duplicates"
    GROUP_BY_LABEL = "group_by_label"

class LiteLLMConfig(BaseModel):
    """Configuration for LiteLLM model and embedding settings."""
    model_config = ConfigDict(extra='forbid')
    
    model: str = "openai/gpt-4o"
    model_api_base: Optional[str] = None
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_api_base: Optional[str] = None

class InputDatasetConfig(BaseModel):
    """Configuration for input dataset source."""
    model_config = ConfigDict(extra='forbid')

    dataset_source: str = "local" # Options: "local", "hub"
    hub_dataset_id: Optional[str] = None
    local_knowledgebase_path: Optional[str] = None 
    local_train_dataset_path: Optional[str] = None 

class UploadConfig(BaseModel):
    """Configuration for Hugging Face Hub uploads."""
    model_config = ConfigDict(extra='forbid')
    
    push_to_hub: bool = False
    hub_private: bool = False
    hub_dataset_id: Optional[str] = None
    hub_model_id: Optional[str] = None

class ChunkerConfig(BaseModel):
    """Configuration for text chunking."""
    model_config = ConfigDict(extra='forbid')
    
    chunker: str
    chunker_arguments: Dict[str, Any]
    output_path: str
    upload_config: Optional[UploadConfig] = None

    @property
    def litellm_config(self) -> Optional[LiteLLMConfig]:
        """Extract ModelConfig from chunker_arguments if present."""
        if "model_config" in self.chunker_arguments:
            return LiteLLMConfig.model_validate(self.chunker_arguments["litellm_config"])
        return None

class QuestionGeneratorConfig(BaseModel):
    """Configuration for question generation."""
    model_config = ConfigDict(extra='forbid')
    
    output_path: str
    input_dataset_config: InputDatasetConfig
    litellm_config: LiteLLMConfig
    max_workers: int = 20
    deduplication_enabled: bool = True
    dedup_embedding_batch_size: int = 500
    similarity_threshold: float = 0.85
    upload_config: Optional[UploadConfig] = None
    llm_calls_per_minute: Optional[int] = 15
    embedding_calls_per_minute: Optional[int] = 15

class ModelSettings(BaseModel):
    """Settings for the embedding model training."""
    model_config = ConfigDict(extra='forbid')
    
    model_id: str
    matryoshka_dimensions: List[int] = [768, 512, 256, 128, 64]
    metric_for_best_model: str = "eval_dim_128_cosine_ndcg@10"
    max_seq_length: int = 768
    trust_remote_code: bool = False

class TrainingArguments(BaseModel):
    """Arguments for model training."""
    model_config = ConfigDict(extra='forbid')
    
    # Required parameters
    output_path: str
    
    # Basic training parameters
    epochs: int = 4
    batch_size: int = 32
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2.0e-5
    
    # Learning rate scheduler settings
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    
    # Optimizer settings
    optim: str = "adamw_torch_fused"
    
    # Hardware optimization flags
    tf32: bool = True
    bf16: bool = True
    
    # Batch sampling strategy
    batch_sampler: BatchSamplers = BatchSamplers.NO_DUPLICATES
    
    # Training and evaluation strategy
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Reporting
    report_to: str = "none"

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    model_config = ConfigDict(extra='forbid')
    
    model_settings: ModelSettings
    training_arguments: TrainingArguments
    upload_config: Optional[UploadConfig] = None
    train_dataset_config: InputDatasetConfig

class PipelineConfig(BaseModel):
    """Main configuration for the QuicKB pipeline."""
    model_config = ConfigDict(extra='forbid')
    
    pipeline: Dict[str, str]
    hub_username: Optional[str] = None
    hub_token: Optional[str] = None
    path_to_knowledgebase: str
    chunker_config: ChunkerConfig
    question_generation: Optional[QuestionGeneratorConfig] = None
    training: Optional[TrainingConfig] = None

def load_dataset_from_local(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from a local JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"Dataset file not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON in {file_path}")
        raise

def load_dataset_from_hub(hub_dataset_id: str, config_name: str) -> List[Dict[str, Any]]:
    """Load dataset from Hugging Face Hub."""
    try:
        try:
            dataset = load_dataset(hub_dataset_id, config_name, split="train") # Try loading 'train' split first
        except ValueError:
            dataset = load_dataset(hub_dataset_id, config_name, split="knowledgebase") # Fallback to 'knowledgebase'

        return dataset.to_list() # Convert HF Dataset to list of dicts for compatibility
    except Exception as e:
        logger.error(f"Error loading dataset from Hugging Face Hub: {hub_dataset_id}, config: {config_name}. Error: {e}")
        raise

def load_pipeline_config(config_path: str | Path = "config.yaml") -> PipelineConfig:
    """Load and validate pipeline configuration."""
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return PipelineConfig.model_validate(config_data)
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {str(e)}")
        raise

def process_chunks(config: PipelineConfig) -> List[Dict[str, Any]]:
    """Process documents into chunks and optionally upload to Hub."""
    
    # Get chunker class
    chunker_class = ChunkerRegistry.get_chunker(config.chunker_config.chunker)

    # Make a copy of chunker arguments
    args = config.chunker_config.chunker_arguments.copy()
    
    # Initialize chunker
    chunker = chunker_class(**args)

    logger.info(f"Initialized Chunker: {config.chunker_config.chunker}")
    
    # Process files
    base_path = Path(config.path_to_knowledgebase)
    results = []
    total_chunks = 0
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
            
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            total_chunks += len(chunks)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue

    logger.info(f"Created {total_chunks} chunks in total")
    
    # Save results
    output_path = Path(config.chunker_config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Handle upload if configured
    if (config.hub_username and 
        config.chunker_config.upload_config and 
        config.chunker_config.upload_config.push_to_hub):
        try:
            # Initialize pusher
            pusher = DatasetPusher(
                username=config.hub_username,
                token=config.hub_token
            )
            
            # Get repository id from config or fallback to path
            repository_id = (config.chunker_config.upload_config.hub_dataset_id 
                           or f"{config.hub_username}/{Path(config.chunker_config.output_path).stem}")
            
            # Collect chunker info
            chunker_info = {
                'chunker_name': config.chunker_config.chunker,
                'chunker_params': config.chunker_config.chunker_arguments
            }
            
            # Push dataset
            pusher.push_dataset(
                hub_dataset_id=repository_id,
                knowledgebase_path=config.chunker_config.output_path,
                chunker_info=chunker_info,
                private=config.chunker_config.upload_config.hub_private
            )
            logger.info(f"Successfully uploaded chunks to Hub: {repository_id}")
        except Exception as e:
            logger.error(f"Failed to upload chunks to Hub: {str(e)}")
    
    return results

def generate_questions(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate questions and optionally upload to Hub."""

    if not config.question_generation:
        raise ValueError("Question generation config is required but not provided")
    
    generator = QuestionGenerator(
        prompt_path="src/prompts/question_generation.txt",
        llm_model=config.question_generation.litellm_config.model,
        embedding_model=config.question_generation.litellm_config.embedding_model,
        dedup_enabled=config.question_generation.deduplication_enabled,
        similarity_threshold=config.question_generation.similarity_threshold,
        max_workers=config.question_generation.max_workers,
        model_api_base=config.question_generation.litellm_config.model_api_base,
        embedding_api_base=config.question_generation.litellm_config.embedding_api_base,
        embedding_batch_size=config.question_generation.dedup_embedding_batch_size,
        llm_calls_per_minute=config.question_generation.llm_calls_per_minute,
        embedding_calls_per_minute=config.question_generation.embedding_calls_per_minute
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
    
    # Create training records
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
    
    # Save results
    if config.question_generation.output_path:
        output_path = Path(config.question_generation.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(train_records, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved training records to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save training records: {str(e)}")

    # Handle upload if configured
    if (config.hub_username and 
        config.question_generation.upload_config and 
        config.question_generation.upload_config.push_to_hub):
        try:            
            # Initialize pusher
            pusher = DatasetPusher(
                username=config.hub_username,
                token=config.hub_token
            )
            
            # Get repository id from config or fallback to maintaining consistency with chunks
            repository_id = (
                config.question_generation.upload_config.hub_dataset_id or
                (config.chunker_config.upload_config.hub_dataset_id if config.chunker_config.upload_config else None) or
                f"{config.hub_username}/{Path(config.chunker_config.output_path).stem}"
            )
            
            # Collect question generation info
            question_gen_info = {
                'model_name': config.question_generation.litellm_config.model,
                'similarity_threshold': config.question_generation.similarity_threshold,
                'num_questions': metrics['num_questions_original'],
                'num_deduped': metrics['num_questions_deduped']
            }
            
            # If using same repository as chunks, include chunker info
            chunker_info = None
            if (not config.question_generation.upload_config.hub_dataset_id or
                config.question_generation.upload_config.hub_dataset_id ==
                config.chunker_config.upload_config.hub_dataset_id):
                chunker_info = {
                    'chunker_name': config.chunker_config.chunker,
                    'chunker_params': config.chunker_config.chunker_arguments
                }

            # Push dataset
            pusher.push_dataset(
                hub_dataset_id=repository_id,
                knowledgebase_path=None,
                chunker_info=chunker_info,
                train_path=config.question_generation.output_path,
                question_gen_info=question_gen_info,
                private=config.question_generation.upload_config.hub_private
            )
            logger.info(f"Successfully uploaded train dataset to Hub: {repository_id}")
        except Exception as e:
            logger.error(f"Failed to upload train dataset to Hub: {str(e)}")
    
    return train_records, metrics

def train_model(config: PipelineConfig, kb_dataset: List[Dict[str, Any]], train_dataset: List[Dict[str, Any]]):
    """Train the embedding model."""
    train_main(config, train_dataset=train_dataset, kb_dataset=kb_dataset)

def upload_to_hub(
    config: PipelineConfig,
    kb_dataset: List[Dict[str, Any]],
    train_dataset: Optional[List[Dict[str, Any]]] = None,
    question_metrics: Optional[Dict[str, int]] = None
):
    """Upload datasets to Hugging Face Hub."""
    
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
            'model_name': config.question_generation.model,
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

    kb_dataset = None
    train_dataset = None
    question_metrics = None

    # 1. CHUNK
    if from_stage.value <= PipelineStage.CHUNK.value <= to_stage.value:
        logger.info("Running CHUNK stage.")
        kb_dataset = process_chunks(config)
    else:
        logger.info("Skipping CHUNK stage.")

    # Load knowledgebase dataset if GENERATE or TRAIN stage is being run and dataset source is configured
    if to_stage.value >= PipelineStage.GENERATE.value and config.question_generation and config.question_generation.input_dataset_config.dataset_source in ["hub", "local"]:
        logger.info("Loading knowledgebase dataset for GENERATE/TRAIN stage.")
        input_config = config.question_generation.input_dataset_config
        if input_config.dataset_source == "hub":
            logger.info(f"Loading knowledgebase dataset from Hub: {input_config.hub_dataset_id}")
            kb_dataset = load_dataset_from_hub(input_config.hub_dataset_id, "knowledgebase")
        elif input_config.dataset_source == "local":
            local_kb_path = input_config.local_knowledgebase_path or config.chunker_config.output_path # Prioritize explicit path
            logger.info(f"Loading knowledgebase dataset from local path: {local_kb_path}")
            kb_dataset = load_dataset_from_local(local_kb_path)


    # 2. GENERATE
    if from_stage.value <= PipelineStage.GENERATE.value <= to_stage.value:
        logger.info("Running GENERATE stage.")
        # Pass kb_dataset directly (already loaded)
        train_dataset, question_metrics = generate_questions(config, kb_dataset)
    else:
        logger.info("Skipping GENERATE stage.")
        # Load train dataset based on training.train_dataset_config if needed for TRAIN stage
        if (to_stage.value >= PipelineStage.TRAIN.value and
                config.training and config.training.train_dataset_config.dataset_source == "hub"):
            logger.info(f"Loading training dataset from Hub: {config.training.train_dataset_config.hub_dataset_id}")
            train_dataset = load_dataset_from_hub(config.training.train_dataset_config.hub_dataset_id, "train")
        elif (to_stage.value >= PipelineStage.TRAIN.value and
              config.training and config.training.train_dataset_config.dataset_source == "local"): # Check training config for local path
            train_dataset_config = config.training.train_dataset_config
            local_train_path = train_dataset_config.local_train_dataset_path or config.question_generation.output_path # Prioritize explicit path
            logger.info(f"Loading training dataset from local path: {local_train_path}")
            train_dataset = load_dataset_from_local(local_train_path) # Use training dataset local path


    # 3. TRAIN
    if from_stage.value <= PipelineStage.TRAIN.value <= to_stage.value:
        logger.info("Running TRAIN stage.")
        if not config.training:
            raise ValueError("No training config found, cannot run TRAIN stage.")
        train_model(config, kb_dataset, train_dataset)
    else:
        logger.info("Skipping TRAIN stage.")

    logger.info("Pipeline complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        config = load_pipeline_config("config.yaml")
        run_pipeline(config)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise