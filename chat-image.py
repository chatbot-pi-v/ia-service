from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_milvus import Milvus
from config.settings import MILVUS_URI, MILVUS_TOKEN, MILVUS_DB_NAME

def query_image(question):
    COLLECTION_NAME = "ImageCollection"
    
    # Carregar modelo CLIP
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Gerar embedding da pergunta
    inputs = processor(text=question, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).squeeze(0).numpy()

    # Conectar ao Milvus na nuvem
    vector_store = Milvus(
        embedding=None,
        connection_args={"uri": MILVUS_URI, "token": MILVUS_TOKEN, "db_name": MILVUS_DB_NAME},
        collection_name=COLLECTION_NAME,
    )

    # Buscar a imagem mais semelhante
    results = vector_store.similarity_search_by_vector(query_embedding, k=1)

    if results:
        best_match = results[0].metadata.get("file_name", "desconhecido")
        print(f"Imagem correspondente encontrada: {best_match}")
        
        # Aqui você pode adaptar para servir a imagem via URL se armazenada em um bucket
        return best_match
    else:
        print("Nenhuma imagem correspondente encontrada.")
        return None
