from milvus.milvus_init import initialize_milvus 
from config.settings import GROQ_API_TOKEN, GROQ_API
from milvus_img import get_best_image_caption_by_text
from services.rag_pipeline import SafeRAGPipeline

vector_store = initialize_milvus()

rag_pipeline = SafeRAGPipeline(
    vector_store=vector_store,
    image_search_fn=get_best_image_caption_by_text,
    groq_api_url=GROQ_API,
    groq_token=GROQ_API_TOKEN,
    model="llama3-8b-8192",
    embedding_model="BAAI/bge-base-en-v1.5"
)

def process_question(question):
    return rag_pipeline.process(question)
