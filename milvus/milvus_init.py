from langchain_milvus import Milvus
from milvus.milvus_db import create_milvus_db
from embeddings.embedding_model import get_embeddings
from processing.pdf_loader import load_pdfs
from config.settings import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME, COLLECTION_NAME

def initialize_milvus():
  create_milvus_db()

  embeddings = get_embeddings()
  documents = load_pdfs()

  vector_store = Milvus.from_documents(
    documents=documents, 
    embedding=embeddings, 
    connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN, "db_name": MILVUS_DB_NAME},
    collection_name=COLLECTION_NAME,
    index_params={"index_type": "FLAT", "metric_type": "L2"},
    consistency_level="Strong", 
    drop_old=True
  )

  return vector_store
