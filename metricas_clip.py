import os
import pandas as pd
import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
from scipy.stats import norm
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from tqdm import tqdm
from config.settings import MILVUS_URI, MILVUS_TOKEN
from duckduckgo_search import DDGS
from pathlib import Path
import requests
from io import BytesIO
import os
import time


# ========== CONFIGURAÇÕES ==========
COLLECTION_NAME = "image_embeddings"
CSV_PERGUNTAS = "dados/perguntas_classificadas.csv"
XLSX_SAIDA = "dados/distancias_clip.xlsx"
os.makedirs("dados", exist_ok=True)

# ========== CONECTA AO MILVUS ==========
connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device)

# ========== CRIA COLLECTION SE NÃO EXISTIR ==========
def prepara_colecao(nome):
    all_collections = utility.list_collections()

    if nome not in all_collections:
        fields = [
            FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name='image_path', dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=512),
            FieldSchema(name='captions', dtype=DataType.VARCHAR, max_length=1024)
        ]
        schema = CollectionSchema(fields, description="Armazena embeddings CLIP de imagens")
        col = Collection(name=nome, schema=schema)
        col.create_index(field_name="embedding", index_params={"metric_type": "IP"})
    else:
        col = Collection(name=nome)
    return col

collection = prepara_colecao(COLLECTION_NAME)

# ========== GERA BASE DE PERGUNTAS ==========
def gerar_perguntas_base(caminho):
    perguntas = [
        # Relevantes
        ("Qual é a importância da Jurema na espiritualidade?", "relevante"),
        ("Quem são os caboclos na Umbanda?", "relevante"),
        ("O que são buzios?", "relevante"),
        ("Como é uma gira de Umbanda?", "relevante"),
        ("Qual é o papel dos guias espirituais?", "relevante"),
        ("O que é oferenda na religião afro-brasileira?", "relevante"),
        ("Como é um terreiro?", "relevante"),
        ("O que é sincretismo religioso?", "relevante"),
        ("Como são os orixás?", "relevante"),
        ("O que é um atabaque?", "relevante"),
        ("O que representa Exu nas religiões afro?", "relevante"),
        ("Qual é o papel das ervas na Umbanda?", "relevante"),
        ("Como funcionam os rituais com tambor?", "relevante"),
        ("Quem é Ogum na religiosidade afro-brasileira?", "relevante"),
        ("Como é celebrada a Festa de Iemanjá?", "relevante"),

        # Meio do caminho
        ("O que é espiritualidade?", "meio_caminho"),
        ("Como se manifesta a fé em diferentes culturas?", "meio_caminho"),
        ("Quais são os tipos de rituais religiosos no mundo?", "meio_caminho"),
        ("Como as religiões influenciam a sociedade?", "meio_caminho"),
        ("Quais são os elementos simbólicos das religiões?", "meio_caminho"),
        ("Como a música é usada em cerimônias religiosas?", "meio_caminho"),
        ("O que é cultura popular?", "meio_caminho"),
        ("Quais são as religiões mais praticadas no Brasil?", "meio_caminho"),
        ("Como surgem as tradições religiosas?", "meio_caminho"),
        ("O que é preconceito?", "meio_caminho"),
        ("Como é o sincretismo na América Latina?", "meio_caminho"),
        ("Qual a relação entre religião e arte?", "meio_caminho"),
        ("Como surgiram lideres espirituais?", "meio_caminho"),
        ("O que são mitologias africanas?", "meio_caminho"),
        ("O que é um símbolo sagrado?", "meio_caminho"),

        # Não relevantes
        ("Como cozinhar arroz integral?", "nao_relevante"),
        ("O que é inteligência artificial?", "nao_relevante"),
        ("Como fazer login no Instagram?", "nao_relevante"),
        ("Quem descobriu o Brasil?", "nao_relevante"),
        ("Qual é a capital do Canadá?", "nao_relevante"),
        ("Como programar em Python?", "nao_relevante"),
        ("Qual o melhor time de futebol do mundo?", "nao_relevante"),
        ("Como limpar janelas corretamente?", "nao_relevante"),
        ("O que é um átomo?", "nao_relevante"),
        ("Como editar vídeos no celular?", "nao_relevante"),
        ("Qual é a diferença entre HTML e CSS?", "nao_relevante"),
        ("O que é um vulcão?", "nao_relevante"),
        ("Quem inventou a lâmpada?", "nao_relevante"),
        ("Como funciona o WhatsApp?", "nao_relevante"),
        ("O que é uma impressora 3D?", "nao_relevante"),
    ]
    df = pd.DataFrame(perguntas, columns=["texto", "categoria"])
    df.to_csv(caminho, index=False)
    print(f"✅ Perguntas salvas em: {caminho}")

# ========== EXTRAI EMBEDDING DE TEXTO ==========
def extrai_embedding_texto(texto):
    tokenizado = clip.tokenize([texto]).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokenizado).cpu().numpy()
    emb_norm = normalize(emb, norm="l2").flatten()
    return emb_norm

# ========== FAZ A BUSCA NO MILVUS ==========
def buscar_imagem_por_texto(query_text, top_k=1):
    collection.load()
    emb = extrai_embedding_texto(query_text)
    results = collection.search(
        data=[emb],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["image_path", "captions"],
    )
    if results and results[0]:
        best = results[0][0]
        return best.entity.get("image_path"), best.entity.get("captions"), best.distance
    return None, None, None

def baixar_imagens(tema, destino='imagens', limite=10):
    Path(destino).mkdir(parents=True, exist_ok=True)

    with DDGS() as ddgs:
        resultados = ddgs.images(tema, max_results=limite)

        for i, item in enumerate(resultados):
            url = item["image"]
            ext = url.split('.')[-1].split('?')[0][:4]  # extensão segura
            nome_arquivo = f"{tema.replace(' ', '_')}_{i}.{ext}"
            caminho = os.path.join(destino, nome_arquivo)

            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                img.save(caminho)
                print(f"[✔] Imagem salva: {caminho}")
            except Exception as e:
                print(f"[✘] Erro ao baixar {url}: {e}")

def inserir_imagens_no_milvus(pasta="imagens_afro", legenda="sem legenda"):
    arquivos = list(Path(pasta).glob("*.*"))
    if not arquivos:
        print("⚠ Nenhuma imagem encontrada para inserir.")
        return

    embeddings = []
    paths = []
    captions = []

    for caminho in tqdm(arquivos, desc="🧠 Extraindo embeddings das imagens"):
        try:
            imagem = preprocess(Image.open(caminho).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model.encode_image(imagem).cpu().numpy()
            emb_norm = normalize(emb.reshape(1, -1), norm="l2").flatten()
            embeddings.append(emb_norm)
            paths.append(str(caminho))
            captions.append(legenda)  # Você pode gerar isso com BLIP, por exemplo
        except Exception as e:
            print(f"Erro ao processar {caminho}: {e}")

    if embeddings:
        collection.insert([paths, embeddings, captions])
        collection.flush()
        print(f"✅ {len(embeddings)} imagens inseridas no Milvus.")

# Temas
temas = [
    "Exu umbanda", "Oxum candomblé", "Oferenda terreiro", "Búzios jogo",
    "Gira terreiro", "Iemanjá ritual", "Caboclo umbanda", "Guia de proteção umbanda",
    "Ervas sagradas candomblé", "Ponto riscado umbanda"
]

for tema in temas:
    baixar_imagens(tema, destino="imagens_afro", limite=5)
    time.sleep(15)

inserir_imagens_no_milvus(pasta="imagens_afro", legenda="Imagem relacionada às religiões afro-brasileiras")


# ========== AVALIA TODAS AS PERGUNTAS ==========
def avaliar_perguntas(csv_path, xlsx_saida):
    df = pd.read_csv(csv_path)
    registros = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Avaliando perguntas"):
        texto, categoria = row["texto"], row["categoria"]
        img, caption, dist = buscar_imagem_por_texto(texto)

        if img is None:
            continue

        registros.append({
            "Texto": texto,
            "Imagem": img,
            "Caption": caption,
            "Similaridade (IP)": dist,
            "Categoria": categoria
        })

    df_resultado = pd.DataFrame(registros)

    # Classificação por faixas de distância (ajuste conforme distribuição dos seus dados)
    def classificar(sim):
        if sim > 0.29:
            return "relevante"
        elif sim > 0.27 and sim < 0.29:
            return "meio_caminho"
        else:
            return "nao_relevante"

    df_resultado["Classificacao_Automatica"] = df_resultado["Similaridade (IP)"].apply(classificar)

    df_resultado.to_excel(xlsx_saida, index=False)
    print(f"✅ Resultados salvos em: {xlsx_saida}")
    return df_resultado

# ========== GERA GRÁFICO DE DISTRIBUIÇÃO ==========
def plotar_distribuicoes(df_resultado, coluna_dist="Similaridade (IP)"):
    plt.figure(figsize=(10, 6))
    categorias = df_resultado["Categoria"].unique()

    for cat in categorias:
        dados = df_resultado[df_resultado["Categoria"] == cat][coluna_dist]
        sns.kdeplot(dados, label=cat, fill=True)

    plt.title("Distribuição das Similaridades por Categoria")
    plt.xlabel("Similaridade (Produto Interno)")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("distribuicao_similaridade.png")
    print("📊 Gráfico salvo como 'distribuicao_similaridade.png'")
    plt.show()

# ========== EXECUÇÃO ==========
if __name__ == "__main__":
    if not os.path.exists(CSV_PERGUNTAS):
        gerar_perguntas_base(CSV_PERGUNTAS)

    df_resultado = avaliar_perguntas(CSV_PERGUNTAS, XLSX_SAIDA)
    plotar_distribuicoes(df_resultado)
