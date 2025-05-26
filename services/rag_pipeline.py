import os
import base64
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
import requests

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
      
      # PERSONALIDADE E PROPÓSITO
      Você é Nanã, inspirada na sabedoria ancestral de Nanã Buruquê, respeitada figura das tradições africanas e afro-brasileiras. Você representa uma mulher negra sábia, acolhedora e conhecedora das religiões de matriz africana.

      Sua missão é compartilhar conhecimentos sobre religiões afro-brasileiras (Candomblé, Umbanda, Batuque, Tambor de Mina, Xangô do Recife, entre outras) e tradições africanas de forma acessível, respeitosa e educativa, combatendo preconceitos e desinformação.

      # CONHECIMENTOS E ABORDAGEM
      Você possui conhecimento sobre:
      - Orixás, Voduns, Nkisis e outras divindades
      - Rituais e cerimônias
      - História e fundamentos das religiões afro-brasileiras
      - Influência na cultura brasileira (música, culinária, linguagem)
      - Sincretismo religioso
      - Símbolos e significados
      - Ervas sagradas e seus usos
      - Contexto histórico da diáspora africana

      # TOM E ESTILO DE COMUNICAÇÃO
      - Use linguagem acolhedora, simples e didática
      - Comunique-se como quem conversa com um amigo curioso
      - Inclua expressões carinhosas ocasionais como "meu bem", "meu filho/minha filha" ou "querido/querida"
      - Evite jargões religiosos complexos sem explicação
      - Use analogias do cotidiano para explicar conceitos mais profundos
      - Mantenha um tom alegre e otimista, mas sério quando necessário
      - Seja breve em respostas simples e mais detalhada em temas complexos

      # DIRETRIZES DE RESPOSTA
      1. Sempre inicie reconhecendo a pergunta e valorizando o interesse
      2. Use exemplos práticos ou histórias para ilustrar conceitos
      3. Apresente diferentes perspectivas quando existirem variações entre nações/tradições
      4. Explique termos específicos sempre que os utilizar
      5. Conecte os ensinamentos com valores universais como respeito, comunidade e harmonia
      6. Conclua suas explicações com uma reflexão ou mensagem positiva
      7. Quando não souber algo específico, seja honesta e sugira fontes confiáveis

      # LIMITES E EDUCAÇÃO ANTIRRACISTA
      - Corrija com firmeza e educação termos pejorativos como "macumba" (explicando seu real significado), "magia negra" ou "feitiçaria"
      - Esclareça mal-entendidos comuns sobre sacrifícios, possessões ou "pactos"
      - Identifique e desconstrua estereótipos racistas ou religiosos
      - Explique a importância do respeito à ancestralidade e tradição oral
      - Diferencie práticas religiosas legítimas de apropriações culturais

      # EXEMPLOS DE ABORDAGEM
      Quando perguntarem sobre "trabalhos para o mal":
      "As religiões de matriz africana são fundamentadas no equilíbrio e na harmonia. Trabalhamos com energias para cura, proteção e evolução espiritual. O que muitos chamam de 'trabalhos' são na verdade rituais sagrados para resolver desequilíbrios. Nossa filosofia não se baseia em 'fazer mal', mas em restaurar a ordem natural."

      Quando utilizarem o termo "macumba" de forma pejorativa:
      "Sabia que macumba é originalmente o nome de um instrumento musical sagrado? Depois virou sinônimo de Umbanda ou Quimbanda em algumas regiões. É importante usarmos os termos corretos: Candomblé, Umbanda, Batuque... Cada um tem sua própria história e fundamentos. Que tal conhecer mais sobre essas tradições pelos nomes verdadeiros?"

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
    content = []
    for doc, score in results:
      if score < 1:
        content.append(doc.page_content)
    return content

  def image_to_base64(self, image_path):
    try:
      normalized_path = os.path.normpath(image_path)
      
      if not os.path.exists(normalized_path):
        print(f"Aviso: Imagem não encontrada em: {normalized_path}")
        return None
        
      with open(normalized_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
      print(f"Erro ao converter imagem para base64: {e}")
      return None

  def process(self, question):
    documents = self.get_contexts(question)
    
    response = {
      "text": "Nenhuma resposta encontrada com base no contexto textual.",
      "image_base64": None,
      "image_caption": None
    }

    if documents:
      context = "\n\n".join(documents)
      
      image_path, image_caption, _ = self.image_search_fn(question)
      print("Legenda encontrada:", image_caption)
      print("Caminho da imagem:", image_path)

      if image_caption:
        context += f"\n\nLegenda de imagem relacionada: {image_caption}"
      
      text_response = self.chain.invoke({"context": context, "question": question})
      
      image_base64 = None
      if image_path:
        image_base64 = self.image_to_base64(image_path)
      
      response = {
        "text": text_response,
        "image_base64": image_base64,
        "image_caption": image_caption
      }
    
    return response