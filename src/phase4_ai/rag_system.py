"""RAG-based Q&A system for Fed Minutes."""

import chromadb
from typing import List, Dict
import openai
from src.utils.config import load_config

class FedMinutesQA:
    def __init__(self):
        self.config = load_config()
        self.client = chromadb.PersistentClient(
            path=self.config['paths']['vector_db']
        )
        self.collection = self.client.get_collection("fed_minutes")
        openai.api_key = self.config['llm']['api_key']
    
    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results
    
    def answer_question(self, question: str) -> str:
        """Answer a question using RAG."""
        # Search for relevant context
        results = self.search(question)
        
        # Build context
        context = "\n\n".join(results['documents'][0])
        
        # Create prompt
        prompt = f"""Based on the following Federal Reserve meeting minutes, 
        please answer this question: {question}
        
        Context:
        {context}
        
        Answer:"""
        
        # Get response from LLM
        response = openai.ChatCompletion.create(
            model=self.config['llm']['model'],
            messages=[
                {"role": "system", "content": "You are an expert on Federal Reserve history."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.config['llm']['temperature'],
            max_tokens=self.config['llm']['max_tokens']
        )
        
        return response.choices[0].message.content


