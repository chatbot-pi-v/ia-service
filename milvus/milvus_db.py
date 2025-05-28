from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, list_collections
from config.settings import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME, COLLECTION_NAME

def create_milvus_db():
  #conexão
  connections.connect(
    alias="default", 
    uri=MILVUS_URI, 
    token=MILVUS_TOKEN, 
    db_name=MILVUS_DB_NAME, 
    timeout=60
  )

  fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=768)
  ]

  schema = CollectionSchema(fields, description="Coleção de embeddings")

  if COLLECTION_NAME not in list_collections():
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="vector", index_params={
      "index_type": "IVF_FLAT", #indexação dos vetores em clusters
      "metric_type": "IP" #similaridade de direção (produto interno)
    })
    collection.load()
  else:
    collection = Collection(name=COLLECTION_NAME)
    collection.load()

  return collection