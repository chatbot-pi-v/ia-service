from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from flask import Flask, request, jsonify

app = Flask(__name__)

load_dotenv()

DB_NAME = os.getenv("MILVUS_DB_NAME")
URI = os.getenv("MILVUS_URI")
TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = "LangChainCollection"

if not URI or not TOKEN:
    raise ValueError("As credenciais do Milvus não foram carregadas corretamente.")


def initialize_milvus():
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)

    documents = []
    
    for file_name in os.listdir('./data'):
        file_path = os.path.join('./data', file_name)
        if file_name.endswith('.pdf'):
            print(f"Carregando: {file_path}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load_and_split(text_splitter=text_splitter))

    print("Todos os arquivos foram carregados.")

    vector_store_saved = Milvus.from_documents(
        documents=documents, 
        embedding=embeddings, 
        connection_args={"uri": URI, "token": TOKEN, "db_name": DB_NAME, "timeout": 60},
        collection_name=COLLECTION_NAME,
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        consistency_level="Strong", 
        drop_old=True
    )

    from pymilvus import connections, list_collections

    connections.connect(alias="default", uri=URI, token=TOKEN, secure=True)
    print("Coleções disponíveis:", list_collections())

    return vector_store_saved

vector_store = initialize_milvus()

def get_contexts(pergunta):
    results = vector_store.similarity_search_with_score(pergunta)
    #print(f"RESULTADOS BD: {results}")

    relevantDocuments = []
    for document, score in results:
        if score < 1:
            relevantDocuments.append((document.page_content, score))

    #print(f"Tamanho relevantDocuments {len(relevantDocuments)}")

    return relevantDocuments

def process_question(question):
    """Busca a pergunta no Milvus e retorna uma resposta relevante."""
    
    # Obtém os contextos relevantes da base
    relevant_documents = get_contexts(question)

    if not relevant_documents:
        return "Nenhuma resposta relevante encontrada."

    # Junta os trechos recuperados para compor o contexto da resposta
    context_text = []
    for doc in relevant_documents:
        context_text.append(doc[0])

    return f"Baseado nas informações disponíveis:\n\n{context_text}"

@app.route('/ask', methods=['POST'])
def answer_question():
    """Recebe uma pergunta do usuário e retorna uma resposta baseada no RAG."""
    
    data = request.get_json()
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Pergunta vazia"}), 400

    response = process_question(question)
    
    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)


## ORGANIZAR CÓDIGO
## CRIAR COLLECTION PARA A BASE DE RELIGIOES 
## CRIAR COLLECTION PARA A BASE DE IMAGENS
## AJUSTAR CONTEXTO etc