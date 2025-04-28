import os
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI

class SafeRAGPipeline:
  def __init__(self, vector_store, image_search_fn, groq_api_url, groq_token, model="llama3-8b-8192"):
    self.vector_store = vector_store
    self.image_search_fn = image_search_fn
    self.model = model

    os.environ["OPENAI_API_KEY"] = groq_token
    os.environ["OPENAI_API_BASE"] = groq_api_url

    self.llm = ChatOpenAI(
      model=model,
      temperature=0,
      request_timeout=10,
    )

    self.prompt = PromptTemplate(
      input_variables=["context", "question"],
      template="""
      # INSTRUÇÃO
      Você é uma pessoa que entende muito sobre religiões de matriz africana e afro-brasileira. Sua missão é ajudar outras pessoas a entenderem esses assuntos de forma simples, leve e direta — como se estivesse trocando uma ideia.

      Sempre que responder:
      1. Evite termos muito técnicos ou linguagem difícil.
      2. Use um tom amigável, como se estivesse explicando pra um amigo ou colega.
      3. Pode usar exemplos ou comparações, se achar que ajuda.
      4. Seja claro e direto, mas mantenha o respeito e profundidade no conteúdo.
      5. Repreenda perguntas desrespeitosas que usem termos como "macumba" de forma pejorativa

      # CONTEXTO
      {context}

      # PERGUNTA
      {question}
      """
    )

    self.chain = (
      {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
      | self.prompt
      | self.llm
      | StrOutputParser()
    )

  def get_contexts(self, question, max_docs=3):
    results = self.vector_store.similarity_search_with_score(question, k=max_docs)
    return [doc.page_content for doc, score in results if score < 1]

  def process(self, question):
    documents = self.get_contexts(question)

    if not documents:
      return "Nenhuma resposta encontrada com base no contexto textual."

    context = "\n\n".join(documents)

    image_path, image_caption, _ = self.image_search_fn(question)
    print("Legenda encontrada:", image_caption)
    print("Caminho da imagem:", image_path)

    if image_caption:
      context += f"\n\nLegenda de imagem relacionada: {image_caption}"

    return self.chain.invoke({"context": context, "question": question})