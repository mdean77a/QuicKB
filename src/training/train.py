import os
import json
import logging
import torch
import yaml
from typing import List, Dict

from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, SentenceTransformerModelCardData
from sentence_transformers import SentenceTransformerTrainingArguments, SentenceTransformerTrainer
from sentence_transformers.losses import MultipleNegativesRankingLoss, MatryoshkaLoss
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.training_args import BatchSamplers
from huggingface_hub import login

logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

def load_main_config(config_path: str = "config.yaml") -> Dict:
    """Loads the global config.yaml as a Python dict."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_evaluation_structures(kb_dataset, test_dataset, kb_id_field="id", kb_text_field="text"):
    """
    Build the dicts needed by InformationRetrievalEvaluator:
      1) corpus:  {corpus_id -> doc_text}
      2) queries: {query_id -> query_text}
      3) relevant_docs: {query_id -> [relevant_corpus_ids]}
    We rely on test_dataset having:
      - 'id' = local integer or index
      - 'anchor' = question text
      - 'global_chunk_id' = reference that matches knowledgebase 'id'
    The knowledgebase dataset has:
      - 'id'   = chunk id, a unique string (or uuid).
      - 'text' = chunk text
    """
    # Build corpus from the knowledgebase
    corpus = {
        row[kb_id_field]: row[kb_text_field]
        for row in kb_dataset
    }

    # Build queries from the test set (only test set needed for evaluation)
    queries = {
        row["id"]: row["anchor"]
        for row in test_dataset
    }

    # relevant_docs: link each query's ID to the knowledgebase IDs that are correct
    relevant_docs = {}
    for row in test_dataset:
        q_id = row["id"]
        # Because multiple queries can point to the same chunk, or vice versa,
        # we find all kb rows that match row["global_chunk_id"].
        if "global_chunk_id" not in row:
            logger.warning(f"Row is missing 'global_chunk_id': {row}")
            continue

        if q_id not in relevant_docs:
            relevant_docs[q_id] = []

        # row["global_chunk_id"] may be a single chunk ID that we match in the KB
        # We'll gather all KB entries with the same chunk id (some data has duplicates)
        # But typically there's a 1:1 mapping. We'll check for all matches.
        g_id = row["global_chunk_id"]
        # In a simpler scenario, the row["global_chunk_id"] is the EXACT knowledgebase 'id'
        relevant_docs[q_id].append(g_id)

    return corpus, queries, relevant_docs


def run_baseline_eval(model, evaluator, dim_list):
    """
    Runs a sequential evaluator across a set of dimensions.
    Prints out the results in a table format for convenience.
    """
    results = evaluator(model)
    print("\nBase Model Evaluation Results")
    print("-" * 85)
    print(f"{'Metric':15} {' '.join([f'{d:>10}d' for d in dim_list])}")
    print("-" * 85)

    metrics = [
        "ndcg@10", "mrr@10", "map@100",
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
    ]

    for metric in metrics:
        row_values = []
        for dim in dim_list:
            key = f"dim_{dim}_cosine_{metric}"
            val = results[key]
            row_values.append(val)
        print(f"{metric:15}", end="  ")
        for val in row_values:
            print(f"{val:10.4f}", end=" ")
        print()
    print("-" * 85)
    print(f"sequential_score: {results['sequential_score']:.4f}\n")

def run_final_eval(model, evaluator, dim_list):
    """
    Similar to run_baseline_eval but labeled for final/fine-tuned results.
    """
    results = evaluator(model)
    print("Fine-Tuned Model Evaluation Results")
    print("-" * 85)
    print(f"{'Metric':15} {' '.join([f'{d:>10}d' for d in dim_list])}")
    print("-" * 85)

    metrics = [
        "ndcg@10", "mrr@10", "map@100",
        "accuracy@1", "accuracy@3", "accuracy@5", "accuracy@10",
        "precision@1", "precision@3", "precision@5", "precision@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
    ]

    for metric in metrics:
        row_values = []
        for dim in dim_list:
            key = f"dim_{dim}_cosine_{metric}"
            row_values.append(results[key])
        print(f"{metric:15}", end="  ")
        for val in row_values:
            print(f"{val:10.4f}", end=" ")
        print()
    print("-" * 85)
    print(f"sequential_score: {results['sequential_score']:.4f}\n")
    return results


def main(config_path: str = "config.yaml"):
    """
    Main entry point for fine-tuning an embedding model on the question-chunk data.
    """
    logging.basicConfig(level=logging.INFO)
    cfg = load_main_config(config_path)

    if not cfg.get("train_embedding", False):
        logger.info("train_embedding is set to False (or missing). Skipping training.")
        return

    train_cfg = cfg.get("training", {})
    kb_path = cfg.get("output_path", "./output/knowledgebase.json")
    train_path = cfg.get("question_output_path", "./output/train_data.json")

    # --- Basic validations
    if not os.path.exists(kb_path):
        raise FileNotFoundError(f"Knowledgebase file not found at: {kb_path}")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data file not found at: {train_path}")

    # Load knowledgebase
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_data = json.load(f)  # List of {id, text, source}

    # Knowledgebase: id => chunk_id, text => chunk_text
    # We rely on 'id' in knowledgebase to unify with train_data's chunk_id

    # Load training data
    # The existing pipeline saves entries with structure:
    #   anchor (question), positive (chunk_text), question_id, chunk_id
    # We'll rename chunk_id -> global_chunk_id for clarity
    train_dataset_full = load_dataset("json", data_files=train_path, split="train")

    # Insert a numeric "id" for each row if missing
    if "id" not in train_dataset_full.column_names:
        train_dataset_full = train_dataset_full.add_column("id", list(range(len(train_dataset_full))))

    # Rename chunk_id => global_chunk_id
    # Because IR Evaluator uses that to link to knowledge base
    if "chunk_id" in train_dataset_full.column_names:
        train_dataset_full = train_dataset_full.rename_column("chunk_id", "global_chunk_id")

    # We want columns: anchor, positive, global_chunk_id, id
    # Make sure "anchor" & "positive" exist
    # If your data uses 'anchor'/'positive' already, we're good
    # Otherwise rename accordingly.

    # Shuffle and split 90/10 for train/test
    train_dataset_full = train_dataset_full.shuffle()
    dataset_split = train_dataset_full.train_test_split(test_size=0.1)
    train_dataset = dataset_split["train"]
    test_dataset  = dataset_split["test"]
    # Log the sizes to verify the split
    logger.info(f"Train dataset size after split: {len(train_dataset)}")
    logger.info(f"Test dataset size after split: {len(test_dataset)}")

    # --- Build a combined corpus dataset for evaluation, so that IR can see all chunk_text
    # We'll unify train + test => to have a single set of anchor/positive pairs
    # Actually, we only need all "positive" text for IR, which is the knowledge base itself,
    # but let's keep consistent with the original demonstration: we might want *all* chunk texts
    # that appear in the train data too. Usually the knowledge base is superset anyway.

    # 1) Create a "test-only" queries set => anchor from test
    # 2) The knowledge base is the corpus, we just need to ensure the ID matches global_chunk_id.

    # We'll do it exactly like the example though:
    #   corpus_dataset: combination of train+test, where 'positive' is the chunk text
    #   BUT we only do that if we want to replicate the exact approach from the notebook.
    #   If the knowledgebase is truly all chunk text, we can skip the combine.

    # For demonstration, let's do the "knowledgebase as corpus" approach:
    # Because your code sets "id" in knowledgebase, we can match that to "global_chunk_id"

    corpus_dataset = kb_data  # a list of dict {id, text, source}

    # Next, build the IR evaluators for multiple matryoshka dimensions
    # We'll create the (corpus, queries, relevant_docs) dicts from knowledgebase + test
    corpus, queries, relevant_docs = build_evaluation_structures(
        kb_dataset=corpus_dataset,
        test_dataset=test_dataset,
        kb_id_field="id",
        kb_text_field="text"
    )

    # Build a multi-dimension IR evaluator
    dim_list = train_cfg.get("matryoshka_dimensions", [768, 512, 256, 128, 64])
    from sentence_transformers.evaluation import InformationRetrievalEvaluator

    evaluators = []
    for d in dim_list:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=f"dim_{d}",
            score_functions={"cosine": cos_sim},
            truncate_dim=d
        )
        evaluators.append(ir_evaluator)

    evaluator = SequentialEvaluator(evaluators)

    # ---  BASELINE EVAL with base model
    base_model_id = train_cfg.get("model_id", "nomic-ai/modernbert-embed-base")
    logger.info(f"Loading base model: {base_model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = SentenceTransformer(
        base_model_id,
        device=device
    )
    # Evaluate the model before fine-tuning
    run_baseline_eval(base_model, evaluator, dim_list)

    # --- PREPARE TRAINING
    # We'll build a second instance of that model with a model card
    # so that we can store the updated card data if we push to hub
    logger.info("Re-initializing model for training (with model card data).")
    model = SentenceTransformer(
        base_model_id,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},  # optional for Flash Attn
        model_card_data=SentenceTransformerModelCardData(
            language="en",
            license="apache-2.0",
            model_name="Fine-tuned with MatryoshkaLoss",
        ),
    )

    # Loss function: MultipleNegativesRankingLoss -> MatryoshkaLoss
    base_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model=model,
        loss=base_loss,
        matryoshka_dims=dim_list
    )

    # Training arguments
    output_dir = train_cfg.get("output_dir", "./output/finetuned_model")
    num_epochs = train_cfg.get("epochs", 4)
    lr = train_cfg.get("learning_rate", 2e-5)
    batch_size = train_cfg.get("batch_size", 32)
    grad_accum = train_cfg.get("gradient_accumulation_steps", 16)
    metric_for_best_model = train_cfg.get("metric_for_best_model", "eval_dim_128_cosine_ndcg@10")

    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="adamw_torch_fused",
        tf32=True,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        report_to="none",
    )

    # Wrap into the SentenceTransformerTrainer
    # Our "train_dataset" must have columns: [anchor, positive]. Others are ignored or removed
    # We can keep "global_chunk_id" in case the trainer tries to remove columns anyway
    final_train_dataset = train_dataset.select_columns(["anchor", "positive"])
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=final_train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )

    logger.info("Beginning training...")
    trainer.train()

    # Save best model
    trainer.save_model()

    # Evaluate final fine-tuned model
    logger.info("Evaluating final fine-tuned model...")
    # Re-load best checkpoint
    fine_tuned_model = SentenceTransformer(output_dir, device=device)
    ft_results = run_final_eval(fine_tuned_model, evaluator, dim_list)

    # Push to Hub if desired
    if train_cfg.get("push_to_hub", False):
        # If user wants to push the model
        # Log in if needed
        HF_TOKEN = os.getenv("HF_TOKEN", "")
        if HF_TOKEN:
            login(token=HF_TOKEN)
        else:
            logger.warning("No HF_TOKEN found in environment. Attempting anyway...")

        hub_repo_id = train_cfg.get("hub_model_id", "YourUserName/modernbert-embed-ft")
        logger.info(f"Pushing model to the Hugging Face Hub: {hub_repo_id}")
        trainer.model.push_to_hub(hub_repo_id)
        logger.info("Upload complete!")

    logger.info("Training pipeline finished.")


if __name__ == "__main__":
    main()
