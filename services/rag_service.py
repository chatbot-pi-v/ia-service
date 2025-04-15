from milvus.milvus_init import initialize_milvus 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from config.settings import GROQ_API_TOKEN, GROQ_API
import os


os.environ["OPENAI_API_KEY"] = GROQ_API_TOKEN
os.environ["OPENAI_API_BASE"] = GROQ_API

vector_store = initialize_milvus()

llm = ChatOpenAI(
  model="llama3-70b-8192",  # testar "llama3-8b-8192"
  temperature=0,
)

def get_contexts(pergunta, max_docs=3):
  results = vector_store.similarity_search_with_score(pergunta, k=max_docs)
  print(f'results = {results}')

  relevant_documents = []
  for document, score in results:
    if score < 1:
      relevant_documents.append((document.page_content, score))

  print(f'relevant_documents = {relevant_documents}')
  return relevant_documents

def process_question(question):
  documents = get_contexts(question)
  resposta = ''

  if len(documents) == 0:
    return "Nenhuma resposta encontrada com base no contexto"

  prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    # INSTRUÇÃO
    Você é uma pessoa que entende muito sobre religiões de matriz africana e afro-brasileira. Sua missão é ajudar outras pessoas a entenderem esses assuntos de forma simples, leve e direta — como se estivesse trocando uma ideia.

    Sempre que responder:
    1. Evite termos muito técnicos ou linguagem difícil.
    2. Use um tom amigável, como se estivesse explicando pra um amigo ou colega.
    3. Pode usar exemplos ou comparações, se achar que ajuda.
    4. Seja claro e direto, mas mantenha o respeito e profundidade no conteúdo.

    # CONTEXTO
    {context}

    # PERGUNTA
    {question}
    """
  )

  rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "distance": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
  )

  resposta = rag_chain.invoke({"context": documents, "question": question})

  return resposta
