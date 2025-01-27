import os
import json
import logging
import torch
import yaml
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.training_args import BatchSamplers
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

def load_main_config(config_path_or_obj):
    """Load configuration from path or use provided config object."""
    if isinstance(config_path_or_obj, (str, bytes, os.PathLike)):
        with open(config_path_or_obj, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return config_path_or_obj.model_dump()

def build_evaluation_structures(kb_dataset, test_dataset, kb_id_field="id", kb_text_field="text"):
    corpus = {row[kb_id_field]: row[kb_text_field] for row in kb_dataset}
    queries = {row["id"]: row["anchor"] for row in test_dataset}
    relevant_docs = {}
    for row in test_dataset:
        q_id = row["id"]
        if "global_chunk_id" not in row:
            logger.warning(f"Missing 'global_chunk_id': {row}")
            continue
        if q_id not in relevant_docs:
            relevant_docs[q_id] = []
        relevant_docs[q_id].append(row["global_chunk_id"])
    return corpus, queries, relevant_docs

def format_evaluation_results(title: str, results: dict, dim_list: list, metrics: list = None):
    if metrics is None:
        metrics = [
            "ndcg@10", "mrr@10", "map@100",
            "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
            "precision@1", "precision@3", "precision@5", "precision@10",
            "recall@1", "recall@3", "recall@5", "recall@10",
        ]

    # Calculate required widths for alignment
    max_dim_length = max(len(str(dim)) for dim in dim_list)
    value_format_length = 5  # '0.000' is 5 characters
    column_width = max(max_dim_length, value_format_length)
    metric_width = max(len(metric) for metric in metrics) if metrics else 10

    # Prepare dimension headers (left-aligned strings)
    dim_header = "  ".join(f"{str(dim):<{column_width}}" for dim in dim_list)
    
    # Create header line and dynamically determine separator length
    header_line = f"{'Metric':{metric_width}}  {dim_header}"
    separator = "-" * len(header_line)

    output = [
        f"\n{title}",
        separator,
        header_line,
        separator
    ]

    # Populate each metric row (values right-aligned)
    for metric in metrics:
        values = []
        for dim in dim_list:
            key = f"dim_{dim}_cosine_{metric}"
            val = results.get(key, 0.0)
            values.append(f"{val:>{column_width}.3f}")  # Right-align values
        metric_line = f"{metric:{metric_width}}  {'  '.join(values)}"
        output.append(metric_line)
    
    # Final row
    output.append(separator)
    
    return "\n".join(output)

def run_baseline_eval(model, evaluator, dim_list):
    results = evaluator(model)
    print(format_evaluation_results("Before Training (Baseline) Results", results, dim_list))
    return results

def run_final_eval(model, evaluator, dim_list):
    results = evaluator(model)
    print(format_evaluation_results("After Training (Fine-Tuned) Results", results, dim_list))
    return results

def save_metrics_to_file(before: dict, after: dict, dim_list: list, path="metrics_comparison.txt"):
    metrics = [
        "ndcg@10", "mrr@10", "map@100",
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
    ]
    
    with open(path, "w", encoding="utf-8") as f:
        title = "Model Performance Metrics Comparison"
        f.write(f"\n{title}\n")
        f.write("=" * len(title) + "\n\n")
        
        # Calculate column widths
        dim_strs = [f"{dim:,}" for dim in dim_list]  # Format with thousands separators
        dim_width = max(len(d) for d in dim_strs)
        dim_header = "Dimension"
        dim_width = max(dim_width, len(dim_header))
        
        num_width = 8  # Width for before/after/delta columns
        headers = ["Before", "After", "Δ"]
        header_width = max(len(h) for h in headers)
        num_width = max(num_width, header_width + 1)  # +1 for negative sign space

        for metric in metrics:
            # Metric header
            f.write(f"{metric}\n")
            f.write("-" * len(metric) + "\n")
            
            # Table header
            header = (f"{dim_header:>{dim_width}} │ "
                      f"{headers[0]:>{num_width}} │ "
                      f"{headers[1]:>{num_width}} │ "
                      f"{headers[2]:>{num_width}}")
            separator = ("─" * dim_width + "─┼─" +
                         "─" * num_width + "─┼─" +
                         "─" * num_width + "─┼─" +
                         "─" * num_width)
            f.write(f"{header}\n{separator}\n")
            
            # Dimension rows
            for dim in dim_list:
                key = f"dim_{dim}_cosine_{metric}"
                try:
                    b_val = before.get(key, 0.0)
                    a_val = after.get(key, 0.0)
                    delta = a_val - b_val
                except KeyError:
                    b_val = a_val = delta = 0.0
                
                # Format dimension with thousands separators
                formatted_dim = f"{dim:,}"
                row = (f"{formatted_dim:>{dim_width}} │ "
                       f"{b_val:>{num_width}.3f} │ "
                       f"{a_val:>{num_width}.3f} │ "
                       f"{delta:>{num_width}.3f}")
                f.write(f"{row}\n")
            
            f.write("\n")  # Space between metrics

def main(config):
    """Main training function."""
    if not hasattr(config, 'training') or not config.training:
        raise ValueError("Training configuration is required but not provided")

    # Get paths from config
    kb_path = config.chunker_config.output_path
    train_path = config.question_generation.output_path if config.question_generation else None

    # Verify files exist
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledgebase file not found: {kb_path}")
    if train_path and not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found: {train_path}")

    # Load knowledge base data
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    # Load and prepare training dataset
    train_dataset_full = load_dataset("json", data_files=train_path, split="train")
    if "id" not in train_dataset_full.column_names:
        train_dataset_full = train_dataset_full.add_column("id", list(range(len(train_dataset_full))))
    if "chunk_id" in train_dataset_full.column_names:
        train_dataset_full = train_dataset_full.rename_column("chunk_id", "global_chunk_id")
    train_dataset_full = train_dataset_full.shuffle()
    dataset_split = train_dataset_full.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    test_dataset = dataset_split["test"]
    logger.info(f"Train size: {len(train_dataset)} | Test size: {len(test_dataset)}")

    # Build evaluation structures
    corpus, queries, relevant_docs = build_evaluation_structures(
        kb_dataset=kb_data,
        test_dataset=test_dataset,
        kb_id_field="id",
        kb_text_field="text"
    )

    # Setup evaluators
    dim_list = config.training.model_settings.matryoshka_dimensions
    evaluators = []
    for d in dim_list:
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"dim_{d}",
                score_functions={"cosine": cos_sim},
                truncate_dim=d
            )
        )
    evaluator = SequentialEvaluator(evaluators)

    # Initialize base model and run baseline evaluation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = SentenceTransformer(config.training.model_settings.model_id, device=device, trust_remote_code=config.training.model_settings.trust_remote_code)
    base_model.max_seq_length = config.training.model_settings.max_seq_length
    baseline_results = run_baseline_eval(base_model, evaluator, dim_list)

    logger.info("Re-initializing for training.")
    model = SentenceTransformer(
        config.training.model_settings.model_id,
        device=device,
        trust_remote_code=config.training.model_settings.trust_remote_code,
        model_kwargs={"attn_implementation": "sdpa"},
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Fine-tuned with [QuicKB](https://github.com/ALucek/QuicKB)",
        ),
    )
    model.max_seq_length = config.training.model_settings.max_seq_length

    # Setup loss functions
    base_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=dim_list
    )

    # Setup training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=config.training.training_arguments.output_path,
        num_train_epochs=config.training.training_arguments.epochs,
        per_device_train_batch_size=config.training.training_arguments.batch_size,
        gradient_accumulation_steps=config.training.training_arguments.gradient_accumulation_steps,
        learning_rate=config.training.training_arguments.learning_rate,
        warmup_ratio=config.training.training_arguments.warmup_ratio,
        lr_scheduler_type=config.training.training_arguments.lr_scheduler_type,
        optim=config.training.training_arguments.optim,
        tf32=config.training.training_arguments.tf32,
        bf16=config.training.training_arguments.bf16,
        batch_sampler=config.training.training_arguments.batch_sampler.value,
        eval_strategy=config.training.training_arguments.eval_strategy,
        save_strategy=config.training.training_arguments.save_strategy,
        logging_steps=config.training.training_arguments.logging_steps,
        save_total_limit=config.training.training_arguments.save_total_limit,
        load_best_model_at_end=config.training.training_arguments.load_best_model_at_end,
        metric_for_best_model=config.training.model_settings.metric_for_best_model,
        report_to=config.training.training_arguments.report_to,
    )

    # Prepare final training dataset and trainer
    final_train_dataset = train_dataset.select_columns(["anchor", "positive"])
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=final_train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()
    trainer.save_model()
    
    # Evaluate fine-tuned model
    fine_tuned_model = SentenceTransformer(config.training.training_arguments.output_path, device=device, trust_remote_code=config.training.model_settings.trust_remote_code)
    fine_tuned_model.max_seq_length = config.training.model_settings.max_seq_length
    final_results = run_final_eval(fine_tuned_model, evaluator, dim_list)

    # Save metrics
    save_metrics_to_file(
        baseline_results, 
        final_results, 
        dim_list, 
        path=f"{config.training.training_arguments.output_path}/metrics_comparison.txt"
    )

    # Handle model upload if configured
    if (config.training.upload_config and 
        config.training.upload_config.push_to_hub):
            
        if not config.hub_token and not os.getenv("HF_TOKEN"):
            logger.warning("No HF_TOKEN in env or config, attempting login anyway.")
                
        if not config.training.upload_config.hub_model_id:
            logger.warning("No hub_model_id specified, skipping upload")
        else:
            logger.info(f"Pushing model to HF Hub: {config.training.upload_config.hub_model_id}")
            trainer.model.push_to_hub(
                config.training.upload_config.hub_model_id, 
                exist_ok=True, 
                private=config.training.upload_config.hub_private
            )
            logger.info("Upload complete!")

    logger.info("Training pipeline finished.")

if __name__ == "__main__":
    main()
