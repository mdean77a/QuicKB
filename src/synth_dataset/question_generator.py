import json
import uuid
from typing import Dict, List
import backoff
from openai import OpenAI
from tqdm import tqdm
import logging

class QuestionGenerator:
    def __init__(self, prompt_path: str, api_key: str = None, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.prompt = self._load_prompt(prompt_path)
        self._question_cache: Dict[str, List[Dict]] = {}

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

    # In generate_for_chunk()
    def generate_for_chunk(self, chunk: str) -> List[Dict]:
        if chunk in self._question_cache:
            return self._question_cache[chunk]

        try:
            response = self._generate(chunk)
            questions = json.loads(response)["questions"]
            for q in questions:
                q.update({
                    "id": str(uuid.uuid4()),
                    "chunk_text": chunk,  # Store original chunk text for mapping
                })
                q.pop("explanation", None)
            self._question_cache[chunk] = questions
            return questions
        except Exception as e:
            print(f"Error generating questions: {str(e)}")
            return []

    def generate_for_chunks(self, chunks: List[str]) -> List[Dict]:
        results = []
        for chunk in tqdm(chunks, desc="Generating questions", unit="chunk"):
            questions = self.generate_for_chunk(chunk)
            results.extend(questions)
        return results