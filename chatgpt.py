import openai

# Set your OpenAI API key
api_key = "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"

# Initialize the OpenAI API client
openai.api_key = api_key

# Define a function to send a message to ChatGPT and get a response
def chat_with_gpt(message):
    response = openai.Completion.create(
        engine="text-davinci-002",  # Use the appropriate engine for ChatGPT
        prompt=message,
        max_tokens=50  # Adjust this to control the response length
    )
    return response.choices[0].text.strip()

# Example usage
user_message = "Tell me a joke."
response = chat_with_gpt(user_message)
print(response)
