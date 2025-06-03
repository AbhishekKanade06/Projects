from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()  # Make sure .env is loaded

api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("Google API key is missing or not set in .env file.")

genai.configure(api_key=api_key)
