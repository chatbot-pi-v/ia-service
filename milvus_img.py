import os
import torch
import clip
from PIL import Image
import numpy as np
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
from sklearn.preprocessing import normalize
from pymilvus import utility
from config.settings import MILVUS_URI, MILVUS_TOKEN

connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)

all_collections = utility.list_collections()

#carrega o modelo na cpu ou gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

collection_name = 'image_embeddings'

#monta o schema da coleção pro banco
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='image_path', dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name='vector', dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name='captions', dtype=DataType.VARCHAR, max_length=1024)
]

schema = CollectionSchema(fields, description='Armazena embeddings de imagens')

collection_exists = collection_name in all_collections

if not collection_exists:
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="vector", index_params={"metric_type": "IP"})
else:
    collection = Collection(name=collection_name)


def extract_clip_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image).cpu().numpy()

    return normalize(embedding.reshape(1, -1), norm="l2").flatten()

def insert_images_into_milvus(path, captions_map):
    image_paths = [
        os.path.join(path, filename)
        for filename in os.listdir(path)
        if filename.endswith((".png", ".jpg", ".jpeg"))
    ]

    entities = []
    for image_path in image_paths:
        filename = os.path.basename(image_path)
        caption = captions_map.get(filename, "")  # Pega a legenda associada ou string vazia
        vector = extract_clip_embedding(image_path)
        
        entities.append({
            "image_path": image_path,
            "vector": vector,
            "captions": caption
        })

    collection.insert(entities)
    collection.flush()

def search_image_by_text(query_text, top_k=3):
    collection.load()

    text_input = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).cpu().numpy()

    text_embedding = normalize(text_embedding.reshape(1, -1), norm="l2").flatten()

    search_params = {"metric_type": "IP"}
    results = collection.search(
        data=[text_embedding],
        anns_field="vector",
        param=search_params,
        limit=top_k,
        output_fields=["image_path", "captions"],
    )

    if results and results[0]:
        keywords = set(query_text.lower().split())  # ou use self.extract_keywords()

        print("\n🎯 Resultados mais próximos:")
        for hit in results[0]:
            caption = (hit.entity.get("captions") or "").lower()
            caption_words = set(caption.split())
            distance = hit.distance

            print(f"- Legenda: {caption} | Distância: {distance:.4f}")

            if keywords & caption_words and distance < 0.3:
                return (
                    hit.entity.get("image_path"),
                    hit.entity.get("captions"),
                    hit.distance,
                )

    # Se não houver nenhuma imagem boa o suficiente
    return None, None, None


def get_best_image_caption_by_text(query_text):
    print(f'queryText = {query_text}')
    result = search_image_by_text(query_text)
    if result is None or result[0] is None:
        return None, None, None
    return result