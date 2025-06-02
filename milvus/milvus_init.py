from langchain_milvus import Milvus
from milvus.milvus_db import create_milvus_db
from embeddings.embedding_model import get_embeddings
from processing.pdf_loader import load_pdfs
from config.settings import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME, COLLECTION_NAME
from pymilvus import connections, utility, list_collections
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
    """
    Conecta ao banco Milvus existente e retorna o vector_store.
    Não cria nem carrega dados - apenas conecta à coleção existente.
    """
    # Conectar ao Milvus
    connections.connect(
        alias="default", 
        uri=MILVUS_URI, 
        token=MILVUS_TOKEN,
        db_name=MILVUS_DB_NAME,
        secure=False, 
        timeout=60
    )
    
    # Verificar se a coleção existe
    if COLLECTION_NAME not in list_collections():
        raise Exception(f"Coleção '{COLLECTION_NAME}' não encontrada no banco de dados.")
    
    # Carregar a coleção existente
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    print(f"Conectado à coleção existente '{COLLECTION_NAME}'.")
    
    # Inicializar embeddings
    embeddings = get_embeddings()
    
    # Conectar ao vector_store existente
    vector_store = Milvus(
        embedding_function=embeddings,
        connection_args={
            "uri": MILVUS_URI, 
            "token": MILVUS_TOKEN, 
            "db_name": MILVUS_DB_NAME
        },
        collection_name=COLLECTION_NAME
    )
    
    return vector_store