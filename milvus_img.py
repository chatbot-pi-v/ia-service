import os
import torch
import clip
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

def load_images_from_folder(folder_path):
    images = []
    image_paths = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            image = Image.open(img_path).convert("RGB")
            images.append(preprocess(image).unsqueeze(0).to(device)) 
            image_paths.append(img_path) 
    return images, image_paths

def find_best_matching_image(text, images, image_paths):
    text_input = clip.tokenize([text]).to(device)
    text_features = model.encode_text(text_input)

    best_image = None
    best_image_path = None
    best_match_idx = -1

    image_features_list = [model.encode_image(image) for image in images]
    image_features = torch.cat(image_features_list, dim=0)

    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    similarities = (image_features @ text_features.T)

    #print("Similaridades:", similarities)

    max_similarity = similarities.max().item()

    if max_similarity < 0.1: 
        print(f"Nenhuma imagem relevante encontrada com uma similaridade suficientemente alta ({max_similarity}).")
        return None, None
    else:
        best_match_idx = similarities.argmax().item()

        if best_match_idx >= len(images):
            print(f"Índice inválido: {best_match_idx}. Número total de imagens: {len(images)}")
            return None, None
        else:
            best_image_path = image_paths[best_match_idx]
            print(f"\nÍndice da imagem mais relevante: {best_match_idx}")
            print(f"Imagem selecionada: {best_image_path}\n")

    return best_image_path


folder_path = "./images"

images, image_paths = load_images_from_folder(folder_path)

text_input = "amarelo"

best_image_path = find_best_matching_image(text_input, images, image_paths)

if best_image_path is not None:
    best_image = Image.open(best_image_path)
    best_image.show()
else:
    print("Não foi possível encontrar uma imagem relevante.")
