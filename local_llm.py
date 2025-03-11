#!/usr/bin/env python3
import random
import time
import re
import os
import sys

class SimpleLLMChatBot:
    def __init__(self):
        """Initialize the chat bot with predefined responses."""
        self.responses = [
            {
                "patterns": ["hello", "hi", "hey", "greetings"],
                "replies": [
                    "Hello! How can I help you today?",
                    "Hi there! What can I assist you with?",
                    "Hey! What's on your mind?"
                ]
            },
            {
                "patterns": ["how are you", "how's it going", "how do you do"],
                "replies": [
                    "I'm just a simple bot, but I'm functioning well! How are you?",
                    "I'm here and ready to chat! How can I help?",
                    "All systems operational! What can I do for you?"
                ]
            },
            {
                "patterns": ["name", "who are you", "what are you"],
                "replies": [
                    "I'm a simple chat bot created in Python.",
                    "Just a basic chat bot simulation. No fancy APIs here!",
                    "I'm SimpleBot, a demonstration of a basic chat interface."
                ]
            },
            {
                "patterns": ["bye", "goodbye", "see you", "farewell"],
                "replies": [
                    "Goodbye! Feel free to chat again later.",
                    "See you later! Have a great day!",
                    "Farewell! Come back anytime."
                ]
            },
            {
                "patterns": ["thank", "thanks", "appreciate"],
                "replies": [
                    "You're welcome!",
                    "Happy to help!",
                    "Anytime! That's what I'm here for."
                ]
            },
            {
                "patterns": ["weather", "temperature", "forecast"],
                "replies": [
                    "I'm sorry, I don't have access to real-time weather data. I'm just a simple chat bot simulation.",
                    "As a basic chat bot, I can't check the weather for you. This would require an API integration.",
                    "I don't have weather capabilities. In a real LLM system, this would be handled with a weather API call."
                ]
            }
        ]
        
        self.default_responses = [
            "I'm just a simple chat bot simulation. In a real implementation, this would connect to an actual LLM API.",
            "Interesting. Tell me more about that.",
            "I understand your message, but as a simple demo I have limited responses.",
            "I'm not as smart as a real LLM like GPT or Claude. This is just a demonstration of a basic chat interface.",
            "A real LLM would provide a more meaningful response here. This is just a simple Python demo."
        ]
        
        # For storing conversation history
        self.conversation_history = []
        
        # ASCII art for bot
        self.bot_avatar = """
          _____
         /     \\
        | o   o |
        |   ∧   |
        |  \\_/  |
         \\_____/
        """
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_with_typing_effect(self, text, delay=0.03):
        """Print text with a typing effect to simulate a bot typing."""
        for char in text:
            print(char, end='', flush=True)
            time.sleep(delay)
        print()
    
    def show_typing_indicator(self, duration=1.0):
        """Show a typing indicator (animated dots)."""
        iterations = int(duration / 0.3)
        for _ in range(iterations):
            for dots in [".", "..", "..."]:
                sys.stdout.write("\rThinking" + dots + "   ")
                sys.stdout.flush()
                time.sleep(0.3)
        sys.stdout.write("\r" + " " * 20 + "\r")
        sys.stdout.flush()
    
    def get_response(self, user_input):
        """Generate a response based on user input."""
        # Convert input to lowercase for pattern matching
        input_lower = user_input.lower()
        
        # Check for pattern matches
        for response_set in self.responses:
            for pattern in response_set["patterns"]:
                if pattern in input_lower:
                    return random.choice(response_set["replies"])
        
        # If no patterns match, use a default response
        return random.choice(self.default_responses)
    
    def format_user_message(self, message):
        """Format the user's message for display."""
        return f"\n\033[94mYou: \033[0m{message}\n"
    
    def format_bot_message(self, message):
        """Format the bot's message for display."""
        return f"\033[92mBot: \033[0m{message}"
    
    def add_to_history(self, user_input, bot_response):
        """Add the current exchange to conversation history."""
        self.conversation_history.append({"user": user_input, "bot": bot_response})
    
    def display_welcome_message(self):
        """Display a welcome message when the chat starts."""
        self.clear_screen()
        print("\033[93m" + self.bot_avatar + "\033[0m")
        print("\033[1m" + "=" * 50 + "\033[0m")
        print("\033[1m  Welcome to the Simple Python LLM Chat Bot  \033[0m")
        print("\033[1m" + "=" * 50 + "\033[0m")
        print("\nThis is a basic simulation of an LLM chat bot.")
        print("Type 'exit', 'quit', or 'bye' to end the conversation.\n")
        
        welcome_message = "Hello! I'm a simple chat bot. How can I help you today?"
        print(self.format_bot_message(welcome_message))
    
    def run(self):
        """Run the chat bot in an interactive loop."""
        self.display_welcome_message()
        
        while True:
            # Get user input
            user_input = input("\n\033[94mYou: \033[0m").strip()
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
                # Show a goodbye message
                self.show_typing_indicator(0.8)
                farewell = random.choice([
                    "Goodbye! It was nice chatting with you.",
                    "See you later! Have a great day!",
                    "Farewell! Feel free to come back anytime."
                ])
                print(self.format_bot_message(farewell))
                time.sleep(1)
                break
            
            # Show thinking animation
            self.show_typing_indicator(random.uniform(0.8, 2.0))
            
            # Get and display bot response
            bot_response = self.get_response(user_input)
            self.print_with_typing_effect(self.format_bot_message(bot_response), delay=0.03)
            
            # Add to conversation history
            self.add_to_history(user_input, bot_response)


if __name__ == "__main__":
    # Create and run the chat bot
    chat_bot = SimpleLLMChatBot()
    try:
        chat_bot.run()
    except KeyboardInterrupt:
        print("\n\nChat bot terminated. Goodbye!")