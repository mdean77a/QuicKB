import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from statistics import mean
from datasets import Dataset, load_dataset
from huggingface_hub import create_repo, upload_file, repo_exists
from .card_generator import DatasetCardGenerator

logger = logging.getLogger(__name__)

class DatasetPusher:
    """Handles uploading datasets to the Hugging Face Hub."""

    def __init__(self, username: str, token: Optional[str] = None):
        """Initialize with HF credentials."""
        self.username = username
        self.token = token or os.getenv("HF_TOKEN")

        if not self.token:
            raise ValueError("No Hugging Face token provided or found in environment")

        self.card_generator = DatasetCardGenerator()

    def _repository_exists(self, repo_id: str) -> bool:
        """Check if repository already exists."""
        try:
            return repo_exists(repo_id, repo_type="dataset")
        except Exception as e:
            logger.error(f"Error checking repository: {str(e)}")
            return False

    def _load_json_file(self, file_path: str) -> list:
        """Load JSON file and ensure it's a list of records."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array in {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def _load_existing_dataset_config(self, hub_dataset_id: str, config_name: str) -> Optional[List[Dict[str, Any]]]:
        """
        Load a specific configuration from an existing dataset on the Hub.

        We always load split="train" by default, because the huggingface
        datasets library typically stores everything in a 'train' split
        unless you explicitly do otherwise.
        """
        try:
            logger.info(f"Attempting to load existing '{config_name}' configuration from Hub: {hub_dataset_id}")
            dataset = load_dataset(
                hub_dataset_id,
                config_name,
                split="train",
                token=self.token
            )
            if dataset:
                logger.info(f"Successfully loaded existing '{config_name}' configuration.")
                return dataset.to_list()  # Convert HF Dataset to list of dicts
            else:
                logger.info(f"No existing '{config_name}' configuration found on Hub.")
                return None

        except FileNotFoundError:
            logger.info(f"No existing '{config_name}' configuration found on Hub (FileNotFound).")
            return None
        except Exception as e:
            logger.error(
                f"Error loading existing '{config_name}' configuration from Hub: "
                f"{hub_dataset_id}, config: {config_name}. Error: {e}"
            )
            return None

    def _calculate_dataset_stats(self, knowledgebase_data: List[Dict[str, Any]], chunker_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics for dataset card."""
        text_lengths = [len(item['text'].split()) for item in knowledgebase_data]
        unique_sources = len(set(item['source'] for item in knowledgebase_data))

        return {
            'num_chunks': len(knowledgebase_data),
            'avg_chunk_size': mean(text_lengths),
            'num_files': unique_sources,
            **chunker_info
        }

    def _create_dataset_card(
        self,
        repository_name: str,
        kb_data: Optional[List[Dict[str, Any]]] = None,
        chunker_info: Optional[Dict[str, Any]] = None,
        train_data: Optional[List[Dict[str, Any]]] = None,
        question_gen_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create dataset card content based on available data."""
        stats = {}
        if kb_data and chunker_info:
            stats = self._calculate_dataset_stats(kb_data, chunker_info)

        return self.card_generator.generate_card(
            dataset_name=repository_name,
            chunker_name=chunker_info.get('chunker_name') if chunker_info else None,
            chunker_params=chunker_info.get('chunker_params') if chunker_info else None,
            num_chunks=stats.get('num_chunks'),
            avg_chunk_size=stats.get('avg_chunk_size'),
            num_files=stats.get('num_files'),
            question_generation=question_gen_info
        )

    def push_dataset(
        self,
        hub_dataset_id: str,
        knowledgebase_path: Optional[str] = None,
        chunker_info: Optional[Dict[str, Any]] = None,
        train_path: Optional[str] = None,
        question_gen_info: Optional[Dict[str, Any]] = None,
        private: bool = True
    ) -> None:
        """Push dataset configurations to the Hugging Face Hub."""
        try:
            repo_exists_flag = self._repository_exists(hub_dataset_id)

            if not repo_exists_flag:
                # Create new repository if it doesn't exist
                create_repo(
                    hub_dataset_id,
                    repo_type="dataset",
                    private=private,
                    token=self.token
                )
                logger.info(f"Created new dataset repository: {hub_dataset_id}")
            else:
                logger.info(f"Dataset repository already exists: {hub_dataset_id}. Updating...")

            # 1) Load existing knowledgebase config
            existing_kb_data = self._load_existing_dataset_config(hub_dataset_id, "knowledgebase")

            # 2) Load existing QA config (**renamed** from "train" to "qa")
            existing_qa_data = self._load_existing_dataset_config(hub_dataset_id, "qa")

            kb_data = self._load_json_file(knowledgebase_path) if knowledgebase_path else None

            # Get repository name from hub_dataset_id
            repository_name = hub_dataset_id.split('/')[-1]

            # Create dataset card content - always regenerate
            card_content = self._create_dataset_card(
                repository_name=repository_name,
                kb_data=kb_data,
                chunker_info=chunker_info,
                train_data=kb_data,  # If you want to pass separate info for QA data, do so here
                question_gen_info=question_gen_info
            )

            # Upload README
            upload_file(
                path_or_fileobj=card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=hub_dataset_id,
                repo_type="dataset",
                token=self.token
            )
            logger.info(f"Uploaded/Updated README.md to {hub_dataset_id}")

            # Prepare merged data references
            merged_kb_data = kb_data
            merged_qa_data = self._load_json_file(train_path) if train_path else None

            # Push knowledgebase config
            if kb_data:
                if existing_kb_data:
                    logger.info("Merging new knowledgebase data with existing data from Hub.")
                    merged_kb_data = existing_kb_data + kb_data
                else:
                    logger.info("No existing knowledgebase data found on Hub, using new data.")

                kb_dataset = Dataset.from_list(merged_kb_data)
                kb_dataset.push_to_hub(
                    hub_dataset_id,
                    config_name="knowledgebase",  # <--- store in config="knowledgebase"
                    token=self.token,
                    private=private,
                )
                logger.info(
                    f"{'Updated' if repo_exists_flag else 'Pushed'} knowledgebase "
                    f"configuration to {hub_dataset_id}"
                )

            # Push QA config
            if train_path:
                if existing_qa_data:
                    logger.info("Merging new QA data with existing data from Hub.")
                    merged_qa_data = existing_qa_data + merged_qa_data
                else:
                    logger.info("No existing QA data found on Hub, using new data.")

                qa_dataset = Dataset.from_list(merged_qa_data)
                qa_dataset.push_to_hub(
                    hub_dataset_id,
                    config_name="qa",  # <--- store in config="qa"
                    token=self.token,
                    private=private,
                )
                logger.info(
                    f"{'Updated' if repo_exists_flag else 'Pushed'} QA "
                    f"configuration to {hub_dataset_id}"
                )

            logger.info(
                f"Successfully {'updated' if repo_exists_flag else 'completed'} "
                f"dataset {'update' if repo_exists_flag else 'upload'} to {hub_dataset_id}"
            )

        except Exception as e:
            logger.error(
                f"Error {'updating' if repo_exists_flag else 'pushing'} "
                f"dataset to Hub: {str(e)}"
            )
            raise