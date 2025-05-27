import os
from dotenv import load_dotenv

load_dotenv(override=True)

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
MILVUS_DB_NAME = os.getenv("MILVUS_DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
GROQ_API_TOKEN = os.getenv("GROQ_API_TOKEN")
GROQ_API = os.getenv("GROQ_API")
