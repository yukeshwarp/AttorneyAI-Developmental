import os
from dotenv import load_dotenv
from openai import AzureOpenAI
import phonetics
from sentence_transformers import SentenceTransformer, util
from rapidfuzz import fuzz


semantic_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
load_dotenv()  # Load environment variables from the .env file

llm_api_key = os.getenv("AZURE_API_KEY")
azure_llm_endpoint = os.getenv("AZURE_ENDPOINT")
llm_model = os.getenv("LLM_MODEL")
llm_api_version = "2024-10-01-preview"

redis_host = os.getenv("REDIS_HOST")
redis_port = 6379
redis_password = os.getenv("REDIS_KEY")

client = AzureOpenAI(
                azure_endpoint=azure_llm_endpoint,
                api_key=llm_api_key,
                api_version="2024-10-01-preview",
            )
llm_headers = {"Content-Type": "application/json", "api-key": llm_api_key}
