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
        if not file_path: # Check if file_path is None or empty string
            return None # Return None if no path provided

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Expected JSON array in {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

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

    def _create_dataset_card(self,
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
            num_chunks=stats.get('num_chunks') if stats else None, # Handle None case for stats
            avg_chunk_size=stats.get('avg_chunk_size') if stats else None, # Handle None case for stats
            num_files=stats.get('num_files') if stats else None, # Handle None case for stats
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

            # Conditionally load knowledgebase data and calculate stats
            kb_data = self._load_json_file(knowledgebase_path) if knowledgebase_path else None
            chunker_stats = self._calculate_dataset_stats(kb_data, chunker_info) if kb_data and chunker_info else None # Conditional stats calculation

            train_data = self._load_json_file(train_path) if train_path else None

            # Get repository name from hub_dataset_id
            repository_name = hub_dataset_id.split('/')[-1]

            # Create dataset card content - always regenerate based on current push
            card_content = self._create_dataset_card(
                repository_name=repository_name,
                kb_data=kb_data,
                chunker_info=chunker_info,
                train_data=train_data,
                question_gen_info=question_gen_info
            )

            # Upload README - always update README
            upload_file(
                path_or_fileobj=card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=hub_dataset_id,
                repo_type="dataset",
                token=self.token
            )
            logger.info(f"Uploaded/Updated README.md to {hub_dataset_id}")

            # --- Push Configurations Conditionally ---

            # Push knowledgebase config if path is provided
            if kb_data:
                kb_dataset = Dataset.from_list(kb_data)
                kb_dataset.push_to_hub(
                    hub_dataset_id,
                    config_name="knowledgebase",
                    token=self.token,
                    private=private,
                )
                logger.info(f"{'Updated' if repo_exists_flag else 'Pushed'} knowledgebase configuration to {hub_dataset_id}")

            # Push train config if path is provided
            if train_data:
                train_dataset = Dataset.from_list(train_data)
                train_dataset.push_to_hub(
                    hub_dataset_id,
                    config_name="train",
                    token=self.token,
                    private=private,
                )
                logger.info(f"{'Updated' if repo_exists_flag else 'Pushed'} train configuration to {hub_dataset_id}")


            logger.info(f"Successfully {'updated' if repo_exists_flag else 'completed'} dataset {'update' if repo_exists_flag else 'upload'} to {hub_dataset_id}")

        except Exception as e:
            logger.error(f"Error {'updating' if repo_exists_flag else 'pushing'} dataset to Hub: {str(e)}")
            raise