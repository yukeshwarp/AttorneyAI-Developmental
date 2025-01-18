import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from the .env file

llm_api_key = os.getenv("AZURE_API_KEY")
azure_llm_endpoint = os.getenv("AZURE_ENDPOINT")
llm_model = os.getenv("LLM_MODEL")
llm_api_version = "2024-10-01-preview"

redis_host = os.getenv("REDIS_HOST")
redis_port = 6379
redis_password = os.getenv("REDIS_KEY")
