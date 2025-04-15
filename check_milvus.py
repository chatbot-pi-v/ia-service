from pymilvus import connections, Collection

# Conectar ao Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Nome da coleção
collection_name = "LangChainCollection"

# Obter a coleção
collection = Collection(collection_name)

# Obter a contagem de embeddings
num_embeddings = collection.num_entities  # Total de vetores na coleção

# Estimar tamanho em MB
embedding_dim = 768  # Ajuste para o tamanho correto do seu modelo
size_in_bytes = num_embeddings * embedding_dim * 4  # float32 ocupa 4 bytes
size_in_mb = size_in_bytes / (1024 * 1024)



# Exibir informações
print(f"Número de embeddings: {num_embeddings}")
print(f"Tamanho estimado da coleção: {size_in_mb:.2f} MB")
