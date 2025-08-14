"""LLM client for AI-powered analysis of Fed Minutes"""

import os
from typing import Dict, List, Optional, Union
from abc import ABC, abstractmethod
import logging
from dataclasses import dataclass

# OpenAI support
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Anthropic support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Ollama support (HTTP requests)
import requests
import json


@dataclass
class LLMResponse:
    """Standard response format from LLM"""
    content: str
    model: str
    usage: Dict[str, int]
    provider: str


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, model: str, temperature: float = 0.1, max_tokens: int = 1000):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if this LLM provider is available"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client"""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model, temperature, max_tokens)
        self.api_key = os.getenv("OPENAI_API_KEY")
        
        if OPENAI_AVAILABLE and self.api_key:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        """Check if OpenAI is available"""
        return OPENAI_AVAILABLE and self.api_key is not None and self.client is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using OpenAI"""
        if not self.is_available():
            raise RuntimeError("OpenAI client not available. Set OPENAI_API_KEY environment variable.")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                provider="openai"
            )
            
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client"""
    
    def __init__(self, model: str = "claude-3-opus-20240229", temperature: float = 0.1, max_tokens: int = 1000):
        super().__init__(model, temperature, max_tokens)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if ANTHROPIC_AVAILABLE and self.api_key:
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Anthropic is available"""
        return ANTHROPIC_AVAILABLE and self.api_key is not None and self.client is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using Anthropic Claude"""
        if not self.is_available():
            raise RuntimeError("Anthropic client not available. Set ANTHROPIC_API_KEY environment variable.")
        
        # Combine system and user prompts for Claude
        full_prompt = ""
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            # Extract text content from response
            content = response.content[0].text if response.content else ""
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                provider="anthropic"
            )
            
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise


class OllamaClient(BaseLLMClient):
    """Ollama local LLM client"""
    
    def __init__(self, model: str = "mistral:7b", temperature: float = 0.1, max_tokens: int = 1000, base_url: str = "http://localhost:11434"):
        super().__init__(model, temperature, max_tokens)
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
    
    def is_available(self) -> bool:
        """Check if Ollama server is running and model is available"""
        try:
            # Check if server is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                return False
                
            # Check if our model is available
            models = response.json().get('models', [])
            model_names = [m.get('name', '') for m in models]
            
            # Check for exact match or base model match
            if self.model in model_names:
                return True
            
            # Check for base model (e.g., 'mistral:7b' matches 'mistral:7b-instruct')
            base_model = self.model.split(':')[0]
            for model_name in model_names:
                if model_name.startswith(f"{base_model}:"):
                    return True
                    
            return False
        except Exception as e:
            self.logger.debug(f"Ollama availability check failed: {e}")
            return False
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate response using Ollama"""
        if not self.is_available():
            raise RuntimeError(f"Ollama server not available or model '{self.model}' not found. "
                             f"Start Ollama and run: ollama pull {self.model}")
        
        # Combine system and user prompts
        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            full_prompt = prompt
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300  # 5 minutes for local generation
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Extract response content
            content = result.get('response', '').strip()
            
            # Extract token usage (Ollama provides these in some versions)
            prompt_tokens = result.get('prompt_eval_count', 0)
            completion_tokens = result.get('eval_count', 0)
            total_tokens = prompt_tokens + completion_tokens
            
            return LLMResponse(
                content=content,
                model=self.model,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                },
                provider="ollama"
            )
            
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Ollama request timeout. Model '{self.model}' may be slow or overloaded.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?")
        except Exception as e:
            self.logger.error(f"Ollama API error: {e}")
            raise


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without API keys"""
    
    def is_available(self) -> bool:
        """Mock client is always available"""
        return True
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Generate mock response"""
        # Provide a reasonable mock response based on prompt content
        if "summarize" in prompt.lower():
            content = "This is a mock summary of the Fed Minutes content. The Federal Reserve discussed monetary policy, interest rates, and economic conditions."
        elif "nixon shock" in prompt.lower():
            content = "This is a mock response about the Nixon Shock. On August 15, 1971, President Nixon announced significant economic policies affecting the Fed's operations."
        elif "bretton woods" in prompt.lower():
            content = "This is a mock response about Bretton Woods. The international monetary system established at Bretton Woods collapsed in the early 1970s."
        else:
            content = f"This is a mock response to your query. In a real implementation, this would analyze the Fed Minutes and provide insights based on: {prompt[:100]}..."
        
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            provider="mock"
        )


def create_llm_client(config: Dict) -> BaseLLMClient:
    """Factory function to create appropriate LLM client based on config"""
    provider = config.get('llm', {}).get('provider', 'openai')
    model = config.get('llm', {}).get('model', 'gpt-4')
    temperature = config.get('llm', {}).get('temperature', 0.1)
    max_tokens = config.get('llm', {}).get('max_tokens', 1000)
    base_url = config.get('llm', {}).get('base_url', 'http://localhost:11434')
    
    # Check environment for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    # Try to create client based on preference and availability
    if provider == "ollama":
        client = OllamaClient(model, temperature, max_tokens, base_url)
        if client.is_available():
            return client
        else:
            logging.warning(f"Ollama server not available or model '{model}' not found.")
            logging.warning(f"Install Ollama and run: ollama pull {model}")
            logging.warning("Falling back to mock client.")
    
    elif provider == "openai" and openai_key:
        client = OpenAIClient(model, temperature, max_tokens)
        if client.is_available():
            return client
    
    elif provider == "anthropic" and anthropic_key:
        client = AnthropicClient(model, temperature, max_tokens)
        if client.is_available():
            return client
    
    # Fallback to mock if no valid provider available
    if provider == "ollama":
        logging.warning(f"Ollama not available. Using mock LLM client.")
    elif provider in ["openai", "anthropic"]:
        logging.warning(f"No API key found for {provider}. Using mock LLM client.")
        logging.warning("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable for real responses.")
    else:
        logging.warning(f"Unknown provider '{provider}'. Using mock LLM client.")
    
    return MockLLMClient(model, temperature, max_tokens)


# System prompts for different types of analysis
SYSTEM_PROMPTS = {
    "general": """You are an expert analyst of Federal Reserve meeting minutes from 1965-1973. 
You have deep knowledge of monetary policy, economic history, and the events surrounding 
the Nixon Shock and Bretton Woods collapse. Provide accurate, insightful analysis based 
on the provided Fed meeting excerpts. Always cite specific meetings or dates when possible.""",
    
    "summary": """You are tasked with summarizing Federal Reserve meeting discussions. 
Focus on key decisions, policy changes, and significant economic discussions. 
Be concise but comprehensive, highlighting the most important points.""",
    
    "research": """You are a historical researcher specializing in Federal Reserve policy 
during the 1960s-1970s. Provide detailed analysis suitable for academic research, 
with attention to historical context and policy implications.""",
    
    "timeline": """You are creating a historical timeline of Federal Reserve actions and discussions. 
Focus on chronological progression, cause-and-effect relationships, and how Fed policies 
evolved over time in response to economic events."""
}


if __name__ == "__main__":
    # Test the LLM clients
    from src.utils.config import load_config
    
    config = load_config()
    client = create_llm_client(config)
    
    print(f"Using LLM client: {client.__class__.__name__}")
    print(f"Available: {client.is_available()}")
    
    if client.is_available():
        response = client.generate(
            "What is the Federal Reserve?",
            system_prompt=SYSTEM_PROMPTS["general"]
        )
        print(f"\nResponse: {response.content[:200]}...")
        print(f"Tokens used: {response.usage}")