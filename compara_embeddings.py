from sentence_transformers import SentenceTransformer, util

referencia = "Qual a importância do candomblé na formação da cultura brasileira?"
frase_a = "Como o candomblé influenciou os costumes e tradições do Brasil?"
frase_b = "Quais são os principais rios do Brasil?"

modelos = {
  "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
  "MPNet": "sentence-transformers/all-mpnet-base-v2",
  "BGE": "BAAI/bge-base-en-v1.5",
  "INTFLOAT": "intfloat/e5-base-v2"
}

for nome, modelo_path in modelos.items():
  print(f'\n{nome}')
  modelo = SentenceTransformer(modelo_path)

  emb_ref = modelo.encode(referencia, convert_to_tensor=True)
  emb_a = modelo.encode(frase_a, convert_to_tensor=True)
  emb_b = modelo.encode(frase_b, convert_to_tensor=True)

  sim_a = util.cos_sim(emb_ref, emb_a).item()
  sim_b = util.cos_sim(emb_ref, emb_b).item()

  print(f"Similaridade com frase A: {sim_a:.4f}")
  print(f"Similaridade com frase B: {sim_b:.4f}")
