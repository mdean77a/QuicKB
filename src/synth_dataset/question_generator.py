import json
import uuid
from typing import Dict, List
import backoff
from litellm import completion, embedding
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from .deduplicator import QuestionDeduplicator

logger = logging.getLogger(__name__)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

class QuestionGenerator:
    def __init__(
        self, 
        prompt_path: str, 
        api_key: str = None,
        llm_model: str = "openai/gpt-4o-mini",
        embedding_model: str = "text-embedding-3-large",
        dedup_enabled: bool = True,
        similarity_threshold: float = 0.85,
        max_workers: int = 20,
        model_api_base: str = None,
        embedding_api_base: str = None
    ):
        self.api_key = api_key
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.prompt = self._load_prompt(prompt_path)
        self._question_cache: Dict[str, List[Dict]] = {}
        self.dedup_enabled = dedup_enabled
        self.deduplicator = QuestionDeduplicator(similarity_threshold) if dedup_enabled else None
        self.max_workers = max_workers
        self.model_api_base = model_api_base
        self.embedding_api_base = embedding_api_base

    def _load_prompt(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _generate(self, chunk: str) -> str:
        # Prepare completion kwargs
        completion_kwargs = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"Text: {chunk}"}
            ],
            "temperature": 0.7,
            "api_key": self.api_key
        }
        
        # Add model_api_base if provided
        if self.model_api_base:
            completion_kwargs["api_base"] = self.model_api_base
            
        response = completion(**completion_kwargs)
        return response.choices[0].message.content

    def generate_for_chunk(self, chunk: str) -> List[Dict]:
        if chunk in self._question_cache:
            return self._question_cache[chunk]

        try:
            response = self._generate(chunk)
            questions = json.loads(response)["questions"]
            for q in questions:
                q.update({
                    "id": str(uuid.uuid4()),
                    "chunk_text": chunk,
                })
                q.pop("explanation", None)
            self._question_cache[chunk] = questions
            return questions
        except Exception as e:
            logger.error(f"Error generating questions: {str(e)}")
            return []

    def generate_for_chunks(self, chunks: List[str]) -> List[Dict]:
        results = []
        uncached_chunks = []

        # First check cache
        for chunk in chunks:
            if chunk in self._question_cache:
                results.extend(self._question_cache[chunk])
            else:
                uncached_chunks.append(chunk)

        if uncached_chunks:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_chunk = {
                    executor.submit(self.generate_for_chunk, chunk): chunk 
                    for chunk in uncached_chunks
                }
                
                with tqdm(total=len(uncached_chunks), desc="Generating questions") as pbar:
                    for future in as_completed(future_to_chunk):
                        chunk = future_to_chunk[future]
                        try:
                            questions = future.result()
                            results.extend(questions)
                        except Exception as e:
                            logger.error(f"Error processing chunk {chunk[:50]}...: {str(e)}")
                        pbar.update(1)
        
        if self.dedup_enabled and self.deduplicator and results:
            # Get embeddings using LiteLLM
            questions_text = [q["question"] for q in results]
            
            # Prepare embedding kwargs
            embedding_kwargs = {
                "model": self.embedding_model,
                "input": questions_text,
                "api_key": self.api_key
            }
            
            # Add embedding_api_base if provided
            if self.embedding_api_base:
                embedding_kwargs["api_base"] = self.embedding_api_base
            
            response = embedding(**embedding_kwargs)
            embeddings = [data["embedding"] for data in response.data]
            
            # Deduplicate questions
            results = self.deduplicator.deduplicate(results, embeddings)
        
        return results