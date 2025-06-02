import os
import base64
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
import math
import re
from sentence_transformers import SentenceTransformer, util

class SafeRAGPipeline:
  def __init__(self, vector_store, image_search_fn, groq_api_url, groq_token, model="llama3-8b-8192", embedding_model="BAAI/bge-base-en-v1.5"):
    self.vector_store = vector_store
    self.image_search_fn = image_search_fn
    self.model = model
    self.embedding_model = SentenceTransformer(embedding_model)

    self.domain_examples = [
      "O que é um Orixá?",
      "Como funciona uma gira na Umbanda?",
      "Qual a origem do Candomblé?",
      "O que significa Exu nas religiões afro?",
      "Para que servem as ervas sagradas no Candomblé?",
      "Quais são os fundamentos do Batuque?",
      "O que é sincretismo religioso?",
      "Quem é Iemanjá?",
      "Como são feitos os rituais de iniciação?",
      "Qual o papel das mulheres nas religiões afro-brasileiras?"
    ]

    self.domain_embeddings = self.embedding_model.encode(
      self.domain_examples,
      convert_to_tensor=True
    )

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
      - Evite se apresentar (não diga "sou a Nanã" ou "eu sou uma IA")
      - Use linguagem acolhedora, simples e didática
      - Comunique-se como quem conversa com um amigo curioso
      - Use expressões carinhosas com moderação: no máximo uma vez por resposta, e evite repetir em perguntas próximas
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

  def score_to_weight(self, score, max_score=1.0):
    # Quanto menor o score, mais similar (ex: score = 0.1 → peso = 0.9)
    weight = max(0.1, 1 - (score / max_score))  # Garante mínimo de 0.1
    return weight

  def crop_context(self, text, weight, max_len=500):
    # Corta o contexto com base no peso e no tamanho máximo permitido
    sentences = text.split(". ")
    num_sentences = max(1, math.ceil(len(sentences) * weight))
    cropped = ". ".join(sentences[:num_sentences])
    return cropped

  def get_contexts(self, question, max_docs=3, max_len=500):
    results = self.vector_store.similarity_search_with_score(question, k=max_docs)
    content = []

    for doc, score in results:
      if score < 1.0:
        weight = self.score_to_weight(score)  # Ex: 0.2 → peso alto
        cropped = self.crop_context(doc.page_content, weight, max_len)
        content.append(cropped)

    return content


  def image_to_base64(self, image_path):
    print(f"image_path: {image_path}")
    try:
        normalized_path = os.path.normpath(image_path)
        print(f"normalized_path: {normalized_path}")
        # Corrige as barras
        clean_path = normalized_path.replace("\\", "/")
        
        # Se detecta duplicação 'images/images', corrige
        parts = clean_path.split('/')
        fixed_parts = []
        for i, part in enumerate(parts):
            if i == 0 or part != parts[i-1]:
                fixed_parts.append(part)

        fixed_path = Path(*fixed_parts).resolve()
        print(f"fixed_path: {fixed_path}")
        
        absolute_path = os.path.abspath(normalized_path)
        print(f"Local completo da imagem: {absolute_path}")

        if not os.path.exists(fixed_path):
            print(f"Aviso: Imagem não encontrada em: {fixed_path}")
            return None

        with open(fixed_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')

    except Exception as e:
        print(f"Erro ao converter imagem para base64: {e}")
        return None


  def extract_keywords(self, text):
    # Extrai palavras significativas com pelo menos 3 letras
    return set(re.findall(r'\b\w{4,}\b', text.lower()))


  def is_question_relevant(self, question, threshold=0.74):
    question_embedding = self.embedding_model.encode(question, convert_to_tensor=True)
    cos_similarities = util.cos_sim(question_embedding, self.domain_embeddings)
    max_score = cos_similarities.max().item()
    print(f"[RELEVÂNCIA] Score da pergunta: {max_score:.3f}")
    return max_score >= threshold


  def process(self, question):
    if not self.is_question_relevant(question):
      return {
          "text": (
              "Olá! Essa pergunta não parece relacionada às religiões afro-brasileiras "
              "ou tradições africanas. Posso te ajudar com temas como Candomblé, Umbanda, "
              "Orixás, sincretismo religioso, ervas sagradas e história da ancestralidade africana."
          ),
          "image_base64": None,
          "image_caption": None
      }
    
    documents = self.get_contexts(question)
    
    if documents:
        context = "\n\n".join(documents)

        image_path, image_caption, distance = self.image_search_fn(question)

        image_base64 = None
        if image_caption and image_path:
            context += f"\n\nLegenda de imagem relacionada: {image_caption}"
            corrected_path = os.path.join("docs/images", os.path.basename(image_path))
            image_base64 = self.image_to_base64(corrected_path)
        else:
            print("⚠️ Nenhuma imagem relevante encontrada.")

        text_response = self.chain.invoke({"context": context, "question": question})
        response = {
            "text": text_response,
            "image_base64": image_base64,
            "image_caption": image_caption if image_base64 else None
        }

    return response
