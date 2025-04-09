from milvus.milvus_init import initialize_milvus 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama

vector_store = initialize_milvus()
llm = Ollama(
  model='mistral',
  temperature=0
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
  resposta = ''
  documents = get_contexts(question)

  if len(documents) == 0:
    resposta = "Parece que essa pergunta está fora do meu tema principal, que é amamentação. Se precisar de informações ou apoio sobre amamentação, estou aqui para ajudar no que for possível!"

  prompt = PromptTemplate(
    input_variables=["context","question"],
    template="""
    # INSTRUÇÃO
    Você é um especialista em religiões de matriz africana e afro-brasileira, e deve responder perguntas sobre o contexto histórico de forma clara e detalhada. Para cada pergunta, siga a seguinte sequência de pensamento para organizar sua resposta:
    1. Explique brevemente o conceito ou termo relacionado à pergunta.
    2. Se houver mais de uma opção, explique cada uma.
    3. Se possível, forneça exemplos práticos ou dicas que ajudem a esclarecer a questão.

    # CONTEXTO PARA RESPOSTAS
    {context}

    # PERGUNTA
    Pergunta: {question}
    """
  )

  rag_chain = (
    {"context": RunnablePassthrough(), "question": RunnablePassthrough(), "distance": RunnablePassthrough()} 
    | prompt
    | llm
    | StrOutputParser()
  )

  resposta = rag_chain.invoke({"context": documents, "question": question})

  max_documents = 3  
  resposta = "\n\n".join(doc[0] for doc in documents[:max_documents])

  return resposta
