import google.generativeai as genai
from config import settings

# Configure the API
genai.configure(api_key=settings.GEMINI_API_KEY)

# List available models
for model in genai.list_models():
    print(f"Model Name: {model.name}")
    print(f"Display Name: {model.display_name}")
    print(f"Description: {model.description}")
    print(f"Generation Methods: {model.supported_generation_methods}")
    print("-" * 50)