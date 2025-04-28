#pip install langchain-huggingface
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

#%pip install -qU  langchain_milvus
from langchain_milvus import BM25BuiltInFunction, Milvus

from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

from create_milvus_db import createMilvusDB

def initialize_milvus():

    createMilvusDB("db_text")

    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    # Configurando o TextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)

    documents = []

    # Listar todos os arquivos PDF na pasta './data'
    for file_name in os.listdir('./data'):

        file_path = os.path.join('./data', file_name)

        if file_name.endswith('.pdf'):  # Verifica se o arquivo é um PDF
            print(f"Carregando arquivo: {file_path}\n")
            
            # Carrega o conteúdo do PDF e divide em partes
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load_and_split(text_splitter=text_splitter))  # Adiciona as páginas à lista

            print(f"Arquivo carregado: {file_path}\n")
        
        else:
            print(f"{file_path} não é um arquivo de formato aceito (PDF)\n")

    print("Todos os arquivos foram carregados!\n")
    
    URI = "http://localhost:19530"
    vector_store_saved = Milvus.from_documents(
        documents=documents, 
        embedding=embeddings, 
        connection_args={"uri": URI, "token": "root:Milvus", "db_name": "db_text"},
        collection_name="LangChainCollection",
        index_params={"index_type": "FLAT", "metric_type": "L2"},
        consistency_level="Strong", 
        drop_old=True
    )

    vector_store_loaded = Milvus(
        embeddings,
        connection_args={"uri": URI},
        collection_name="LangChainCollection",
    )

    return vector_store_saved