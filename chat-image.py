from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_milvus import Milvus
from PIL import Image
import os

# Importa a função de inicialização
from milvus_img import initialize_milvus_for_images  # Substitua 'your_module' pelo nome correto do arquivo

def query_image(question):
    # Carregar o modelo CLIP
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)

    # Gerar o embedding da pergunta
    inputs = processor(text=question, return_tensors="pt")
    with torch.no_grad():
        query_embedding = model.get_text_features(**inputs).squeeze(0).numpy()

    # Conectar ao Milvus e buscar a imagem mais semelhante
    URI = "http://localhost:19530"
    vector_store = Milvus(
        embedding=None,  # Não precisamos de embeddings aqui, pois já temos um pronto
        connection_args={"uri": URI, "token": "root:Milvus", "db_name": "db_images"},
        collection_name="ImageCollection",
    )

    results = vector_store.similarity_search_by_vector(query_embedding, k=1)  # Retorna a melhor correspondência

    if results:
        best_match = results[0].metadata["file_name"]
        image_path = f"./images/{best_match}"
        
        if os.path.exists(image_path):
            print(f"Imagem correspondente encontrada: {image_path}")
            return Image.open(image_path)
        else:
            print("Imagem encontrada no banco, mas não localizada no diretório.")
            return None
    else:
        print("Nenhuma imagem correspondente encontrada.")
        return None


print("Inicializando Milvus e carregando imagens...")
initialize_milvus_for_images()  # Garante que as imagens foram salvas no Milvus

question = "Um cachorro brincando na praia"
image = query_image(question)

if image:
    image.show()  # Exibe a imagem correspondente
