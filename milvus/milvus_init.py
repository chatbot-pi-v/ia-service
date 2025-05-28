from langchain_milvus import Milvus
from milvus.milvus_db import create_milvus_db
from embeddings.embedding_model import get_embeddings
from processing.pdf_loader import load_pdfs
from config.settings import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME, COLLECTION_NAME
from pymilvus import connections, utility
from pymilvus import Collection

def collection_exists(collection_name, uri, token, db_name):
  connections.connect(
    alias="default",
    uri=uri,
    token=token,
    db_name=db_name,
  )
  return utility.has_collection(collection_name)

def initialize_milvus():
  exists = collection_exists(COLLECTION_NAME, MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME)

  if not exists:
    create_milvus_db()



  embeddings = get_embeddings()
  documents = load_pdfs()

  vector_store = Milvus.from_documents(
    documents=documents, 
    embedding=embeddings, 
    connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN, "db_name": MILVUS_DB_NAME},
    collection_name=COLLECTION_NAME,
    text_field="quotes",
    index_params={"index_type": "IVF_FLAT", "metric_type": "IP"},
    drop_old=False
  )


  collection = Collection("text_embedding")
  collection.load()
    

  print("\nNúmero de documentos:", collection.num_entities)
  print("\nInformações da coleção:")
  print(collection.describe())
  return vector_store
