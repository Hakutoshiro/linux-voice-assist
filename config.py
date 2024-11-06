from dotenv import load_dotenv
import os
import getpass
# Load environment variables from .env file
load_dotenv()


# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Groq API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Vosk model path
VOSK_MODEL_PATH = os.getenv("VOSK_MODEL_PATH")
EMBEDDINGS_DIR_PATH = os.getenv("EMBEDDINGS_DIR_PATH")



# Prompt template for Gemini API
PROMPT_TEMPLATE = """
You are given a task where you need to provide a Bash command that can be directly executed in a Bash script. The command should resolve the issue described below, and your response should follow these rules:
1. Provide the commands in the form of bash script to solve the problem.
2. Include a small description explaining what each command does.
3. Return the result in JSON format so that it can be parsed easily.
4. Each JSON object should have the following structure:
   - "bash script": The actual Bash script to be executed.
   - "description": Just small explanation of what the script does. Don't mention the word script itself in the description.

Here is the issue you need to resolve: {query}
Please return only the JSON object in your response.
Don't include any additional text like ``` or comments in your response.
Return in stringified JSON format only so that it can be converted using python json.loads().
"""

OUTPUT_INTERPRETATION_PROMPT = """
You are a virtual assistant answering user queries based on the output of a Bash command.
Review the user query and the raw output below, and provide a straightforward, user-friendly interpretation of the results.

User Query: {query}

Raw Command Output:
{output}

Respond concisely to the user’s query based on the command output, using clear language. Explain any technical terms as needed, and provide relevant advice or context if it helps with understanding.

Do not include any extra characters or comments, as your response will be converted to speech.

Dont make reponse too lenght, answer in 2-3 sentences.
"""
