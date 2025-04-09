import os
import torch
import clip
from PIL import Image
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sklearn.preprocessing import normalize
from pymilvus import utility

#conecta ao milvus
connections.connect(host="localhost", port="19530")
#lista de coleções existentes
all_collections = utility.list_collections()

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)



# criar coleção no milvus
collection_name = 'image_embeddings'

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='image_path', dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512),
]

schema = CollectionSchema(fields, description='Armazena embeddings de imagens')



# Verifica se a coleção já existe
collection_exists = collection_name in all_collections

if not collection_exists:
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "L2"})
else:
    collection = Collection(name=collection_name) #realiza a instancia


def extract_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy()

    return normalize(embedding.reshape(1, -1), norm="l2").flatten()

def insert_images_into_milvus(path):
    image_paths = [
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if filename.endswith((".png", ".jpg", ".jpeg"))
    ]

    entities = [
        {"image_path": p, "embedding": extract_clip_embedding(p)}
        for p in image_paths
    ]

    collection.insert(entities)
    collection.flush()

# Função para buscar imagens no Milvus usando texto
def search_image_by_text(query_text, top_k=1):
    collection.load()
    # Converter texto para embedding
    text_input = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).cpu().numpy()

    text_embedding = normalize(text_embedding.reshape(1, -1), norm="l2").flatten()

    # Fazer busca no Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[text_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["image_path"],
    )

    if results and results[0]:
        best_match = results[0][0]
        return best_match.entity.get("image_path"), best_match.distance
    return None, None

# Testando a inserção e a busca
image_folder = "./images"
insert_images_into_milvus(image_folder)

print(f"Número de imagens no Milvus: {collection.num_entities}")

query_text = "vermelho"
best_image_path, similarity = search_image_by_text(query_text)

if best_image_path:
    print(f"Melhor imagem encontrada: {best_image_path} (Similaridade: {similarity})")
    Image.open(best_image_path).show()
else:
    print("Nenhuma imagem relevante encontrada.")


# def load_images_from_folder(folder_path):
#     images = []
#     image_paths = []
    
#     for filename in os.listdir(folder_path):
#         if filename.endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(folder_path, filename)
#             image = Image.open(img_path).convert("RGB")
#             images.append(preprocess(image).unsqueeze(0).to(device)) 
#             image_paths.append(img_path) 
#     return images, image_paths

# def find_best_matching_image(text, images, image_paths):
#     text_input = clip.tokenize([text]).to(device)
#     text_features = model.encode_text(text_input)

#     best_image = None
#     best_image_path = None
#     best_match_idx = -1

#     image_features_list = [model.encode_image(image) for image in images]
#     image_features = torch.cat(image_features_list, dim=0)

#     image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#     text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

#     similarities = (image_features @ text_features.T)

#     #print("Similaridades:", similarities)

#     max_similarity = similarities.max().item()

#     if max_similarity < 0.1: 
#         print(f"Nenhuma imagem relevante encontrada com uma similaridade suficientemente alta ({max_similarity}).")
#         return None, None
#     else:
#         best_match_idx = similarities.argmax().item()

#         if best_match_idx >= len(images):
#             print(f"Índice inválido: {best_match_idx}. Número total de imagens: {len(images)}")
#             return None, None
#         else:
#             best_image_path = image_paths[best_match_idx]
#             print(f"\nÍndice da imagem mais relevante: {best_match_idx}")
#             print(f"Imagem selecionada: {best_image_path}\n")

#     return best_image_path


# folder_path = "./images"

# images, image_paths = load_images_from_folder(folder_path)

# text_input = "amarelo"

# best_image_path = find_best_matching_image(text_input, images, image_paths)

# if best_image_path is not None:
#     best_image = Image.open(best_image_path)
#     best_image.show()
# else:
#     print("Não foi possível encontrar uma imagem relevante.")


##ADICIONAR BASE DO DRIVE
 #Ver possibilidade de automatizar

##Organizar código e arquivos, tornar modularizado e o código legível