import torch
from transformers import pipeline

def simple_local_llm_chat():
    print("Simple Local LLM Chat (type 'exit' to quit)")
    print("-" * 40)
    print("Loading model... (this might take a minute)")
    
    # Load a small model suitable for running on a typical computer
    generator = pipeline('text-generation', model='TinyLlama/TinyLlama-1.1B-Chat-v1.0')
    
    print("Model loaded! Ready for chat.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Assistant: Goodbye!")
            break
        
        # Format input for chat-tuned models
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
        
        try:
            # Generate response with the model
            response = generator(prompt, 
                                max_length=200, 
                                num_return_sequences=1,
                                temperature=0.7)
            
            # Extract and clean up the response
            generated_text = response[0]['generated_text']
            
            # Extract just the assistant's response
            assistant_response = generated_text.split("<|assistant|>\n")[-1].split("<|")[0].strip()
            
            # Print the response
            print(f"\nAssistant: {assistant_response}")
            
        except Exception as e:
            print(f"Error: {e}")
    
if __name__ == "__main__":
    simple_local_llm_chat()