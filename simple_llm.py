import os
import openai

def simple_llm_chat():
    # Set your OpenAI API key
    # Replace with your actual API key or set as environment variable
    openai.api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    
    print("Simple LLM Chat (type 'exit' to quit)")
    print("-" * 40)
    
    # Keep track of conversation history
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to exit
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Assistant: Goodbye!")
            break
        
        # Add user message to conversation
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Call the OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # You can change to other models
                messages=messages
            )
            
            # Get the assistant's response
            assistant_response = response.choices[0].message["content"]
            
            # Add assistant response to conversation history
            messages.append({"role": "assistant", "content": assistant_response})
            
            # Print the response
            print(f"\nAssistant: {assistant_response}")
            
        except Exception as e:
            print(f"Error: {e}")
    
if __name__ == "__main__":
    simple_llm_chat()