import json
import logging
import os
from pathlib import Path
from typing import Optional
from datasets import Dataset
from huggingface_hub import create_repo
from huggingface_hub import repo_exists

logger = logging.getLogger(__name__)

class DatasetPusher:
    """Handles uploading datasets to the Hugging Face Hub."""
    
    def __init__(self, username: str, token: Optional[str] = None):
        """Initialize with HF credentials."""
        self.username = username
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.token:
            raise ValueError("No Hugging Face token provided or found in environment")
            
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
            
    def push_dataset(
        self,
        repository_name: str,
        knowledgebase_path: str,
        train_path: Optional[str] = None,
        private: bool = True
    ) -> None:
        """
        Push dataset to the Hugging Face Hub.
        
        Args:
            repository_name: Name for the repository
            knowledgebase_path: Path to knowledgebase JSON file
            train_path: Optional path to training data JSON file
            private: Whether to create a private repository
        """
        # Construct full repository ID
        repo_id = f"{self.username}/{repository_name}"
        
        # Check if repository exists
        if self._repository_exists(repo_id):
            logger.warning(f"Repository {repo_id} already exists, skipping upload")
            return
            
        try:
            # Create repository
            create_repo(
                repo_id,
                repo_type="dataset",
                private=private,
                token=self.token
            )
            
            # Load and push knowledgebase
            kb_data = self._load_json_file(knowledgebase_path)
            kb_dataset = Dataset.from_list(kb_data)
            kb_dataset.push_to_hub(
                repo_id,
                config_name="knowledgebase",
                token=self.token,
                private=private
            )
            logger.info(f"Pushed knowledgebase configuration to {repo_id}")
            
            # Load and push training data if provided
            if train_path:
                train_data = self._load_json_file(train_path)
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