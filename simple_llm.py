import os
import openai # type: ignore

def simple_llm_chat():
    # Set OpenAI API key
    openai.api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    
    print("Simple LLM Chat (type 'exit' to quit)")
    print("-" * 40)
    
    # Keep track of conversation history
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    