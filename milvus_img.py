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
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512),
    FieldSchema(name='captions', dtype=DataType.VARCHAR, max_length=1024)
]

schema = CollectionSchema(fields, description='Armazena embeddings de imagens')

collection_exists = collection_name in all_collections

if not collection_exists:
    collection = Collection(name=collection_name, schema=schema)
    collection.create_index(field_name="embedding", index_params={"metric_type": "L2"})
else:
    collection = Collection(name=collection_name)


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

def search_image_by_text(query_text, top_k=1):
    collection.load()
    text_input = clip.tokenize([query_text]).to(device)
    with torch.no_grad():
        text_embedding = model.encode_text(text_input).cpu().numpy() #transformando textos em valores menores convertidos em numpy

    text_embedding = normalize(text_embedding.reshape(1, -1), norm="l2").flatten()

    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search(
        data=[text_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["image_path", "captions"],
    )

    if results and results[0]:
        best_match = results[0][0] #pega o primeiro resultado (mais proximo)
        return (
            best_match.entity.get("image_path"),
            best_match.entity.get("captions"),
            best_match.distance, #distancia vetorial mais perto de 0
        )
    return None, None, None


def get_best_image_caption_by_text(query_text):
    print(f'queryText = {query_text}')
    result = search_image_by_text(query_text)
    if result is None or result[0] is None:
        return None, None, None
    return result