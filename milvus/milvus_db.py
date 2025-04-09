from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, list_collections
from config.settings import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME, COLLECTION_NAME

def create_milvus_db():
  #conexão
  connections.connect(
    alias="default", 
    uri=MILVUS_URI, 
    token=MILVUS_TOKEN, 
    db_name=MILVUS_DB_NAME, 
    secure=True, 
    timeout=60
  )

  # Definir os campos do schema
  fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768) #ajustar dimensao
  ]

  schema = CollectionSchema(fields, description="Coleção de embeddings")

  if COLLECTION_NAME not in list_collections():
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="embedding", index_params={
      "index_type": "IVF_FLAT", 
      "metric_type": "L2"
    })
    collection.load()
    print(f"Coleção '{COLLECTION_NAME}' criada e carregada com sucesso.")
  else:
    collection = Collection(name=COLLECTION_NAME)
    collection.load()
    print(f"Coleção '{COLLECTION_NAME}' já existe e foi carregada.")

  return collection
