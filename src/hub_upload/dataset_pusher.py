import json
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from statistics import mean
from datasets import Dataset
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
        kb_data: List[Dict[str, Any]],
        chunker_info: Dict[str, Any],
        train_data: Optional[List[Dict[str, Any]]] = None,
        question_gen_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create dataset card content."""
        stats = self._calculate_dataset_stats(kb_data, chunker_info)
        
        return self.card_generator.generate_card(
            dataset_name=repository_name,
            chunker_name=chunker_info['chunker_name'],
            chunker_params=chunker_info['chunker_params'],
            num_chunks=stats['num_chunks'],
            avg_chunk_size=stats['avg_chunk_size'],
            num_files=stats['num_files'],
            question_generation=question_gen_info
        )
            
    def push_dataset(
        self,
        repository_name: str,
        knowledgebase_path: str,
        chunker_info: Dict[str, Any],
        train_path: Optional[str] = None,
        question_gen_info: Optional[Dict[str, Any]] = None,
        private: bool = True
    ) -> None:
        """
        Push dataset to the Hugging Face Hub.
        
        Args:
            repository_name: Name for the repository
            knowledgebase_path: Path to knowledgebase JSON file
            chunker_info: Dictionary containing chunker name and parameters
            train_path: Optional path to training data JSON file
            question_gen_info: Optional dictionary with question generation info
            private: Whether to create a private repository
        """
        # Construct full repository ID
        repo_id = f"{self.username}/{repository_name}"
        
        # Check if repository exists
        if self._repository_exists(repo_id):
            logger.warning(f"Repository {repo_id} already exists, skipping upload")
            return
            
        try:
            # Load data
            kb_data = self._load_json_file(knowledgebase_path)
            train_data = self._load_json_file(train_path) if train_path else None
            
            # Create repository
            create_repo(
                repo_id,
                repo_type="dataset",
                private=private,
                token=self.token
            )
            
            # Create and push README.md
            card_content = self._create_dataset_card(
                repository_name=repository_name,
                kb_data=kb_data,
                chunker_info=chunker_info,
                train_data=train_data,
                question_gen_info=question_gen_info
            )

            # Upload README directly to hub
            upload_file(
                path_or_fileobj=card_content.encode('utf-8'),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
                token=self.token
            )
            logger.info(f"Uploaded README.md to {repo_id}")
            
            # Push knowledgebase configuration
            kb_dataset = Dataset.from_list(kb_data)
            kb_dataset.push_to_hub(
                repo_id,
                config_name="knowledgebase",
                token=self.token,
                private=private
            )
            logger.info(f"Pushed knowledgebase configuration to {repo_id}")
            
            # Push training data if provided
            if train_data:
                train_dataset = Dataset.from_list(train_data)
                train_dataset.push_to_hub(
                    repo_id,
                    config_name="train",
                    token=self.token,
                    private=private
                )
                logger.info(f"Pushed training configuration to {repo_id}")
            
            logger.info(f"Successfully completed dataset upload to {repo_id}")
            
        except Exception as e:
            logger.error(f"Error pushing dataset to Hub: {str(e)}")
            raise