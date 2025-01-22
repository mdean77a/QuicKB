import json
import uuid
from typing import Dict, List
import backoff
from openai import OpenAI
from tqdm import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from .deduplicator import QuestionDeduplicator

logger = logging.getLogger(__name__)

class QuestionGenerator:
    def __init__(
        self, 
        prompt_path: str, 
        api_key: str = None, 
        model: str = "gpt-4o-mini",
        dedup_enabled: bool = True, 
        similarity_threshold: float = 0.85,
        max_workers: int = 10  # Number of parallel workers
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.prompt = self._load_prompt(prompt_path)
        self._question_cache: Dict[str, List[Dict]] = {}
        self.dedup_enabled = dedup_enabled
        self.deduplicator = QuestionDeduplicator(similarity_threshold) if dedup_enabled else None
        self.max_workers = max_workers

    def _load_prompt(self, path: str) -> str:
        with open(path, 'r') as f:
            return f.read()

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _generate(self, chunk: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": f"Text: {chunk}"}
            ],
            temperature=0.7,
        )
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
                # Submit all chunks to thread pool
                future_to_chunk = {
                    executor.submit(self.generate_for_chunk, chunk): chunk 
                    for chunk in uncached_chunks
                }
                
                # Process completed futures with progress bar
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
            # Get embeddings for questions
            questions_text = [q["question"] for q in results]
            response = self.client.embeddings.create(
                input=questions_text,
                model="text-embedding-3-large"
            )
            embeddings = [data.embedding for data in response.data]
            
            # Deduplicate questions
            results = self.deduplicator.deduplicate(results, embeddings)
        
        return results