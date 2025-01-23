import json
from pathlib import Path
import os
from typing import Dict, Any, Optional, List

class DatasetCardGenerator:
    """Handles dataset card generation for quickb datasets."""
    
    def __init__(self, template_path: str = "src/hub_upload/template.md"):
        """Initialize with path to card template."""
        self.template_path = template_path
        with open(template_path, 'r', encoding='utf-8') as f:
            self.template = f.read()
            
    def _get_size_category(self, num_entries: int) -> str:
        """Determine the size category based on number of entries."""
        if num_entries < 1000:
            return "n<1K"
        elif num_entries < 10000:
            return "1K<n<10K"
        elif num_entries < 100000:
            return "10K<n<100K"
        elif num_entries < 1000000:
            return "100K<n<1M"
        else:
            return "n>1M"
            
    def _format_chunker_params(self, params: Dict[str, Any]) -> str:
        """Simple Markdown-safe parameter formatting"""
        return "\n".join(
            f"- **{key}**: `{repr(value)}`" 
            for key, value in params.items() 
            if value is not None and not key.startswith('_')
        )
        
    def _format_question_generation(self, 
        model_name: str,
        similarity_threshold: float,
        num_questions: int,
        num_deduped: int
    ) -> str:
        """Format question generation section if enabled."""
        return f"""### Question Generation
- Model: {model_name}
- Deduplication threshold: {similarity_threshold}
- Results:
  - Total questions generated: {num_questions}
  - Questions after deduplication: {num_deduped}"""

    def _format_train_config(self) -> str:
        """Format train configuration section."""
        return """2. `train`: Contains generated question-answer pairs
   - Fields: anchor (string), positive (string), question_id (string), chunk_id (string)"""
        
    def generate_card(self,
        dataset_name: str,
        chunker_name: str,
        chunker_params: Dict[str, Any],
        num_chunks: int,
        avg_chunk_size: float,
        num_files: int,
        question_generation: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate a dataset card with the provided information."""
        
        # Load knowledgebase data to determine size category
        size_category = self._get_size_category(num_chunks)
        
        # Determine if question generation is enabled
        gen_tag = "\n- question-generation" if question_generation else ""
        config_count = "two" if question_generation else "one"
        
        # Format question generation section if enabled
        qg_section = ""
        train_config = ""
        if question_generation:
            qg_section = self._format_question_generation(
                model_name=question_generation["model_name"],
                similarity_threshold=question_generation["similarity_threshold"],
                num_questions=question_generation["num_questions"],
                num_deduped=question_generation["num_deduped"]
            )
            train_config = self._format_train_config()
            
        # Fill template
        return self.template.format(
            dataset_name=dataset_name,
            gen_tag=gen_tag,
            size_category=size_category,
            chunker_name=chunker_name,
            chunker_params=self._format_chunker_params(chunker_params),
            num_chunks=num_chunks,
            avg_chunk_size=f"{avg_chunk_size:.1f}",
            num_files=num_files,
            question_generation=qg_section,
            config_count=config_count,
            train_config=train_config
        )