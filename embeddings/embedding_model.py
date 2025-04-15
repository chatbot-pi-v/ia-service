from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
  return HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2') #testar outros modelo (algum recente)


#proxima semana:
#verificar dimensao de geração de embeddings das imagens
#fazer a relação de coluna de imagem no banco com os textos