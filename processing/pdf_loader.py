import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(data_path="./data"):
  documents = []
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=300)

  for file_name in os.listdir(data_path):
    file_path = os.path.join(data_path, file_name)
    if file_name.endswith('.pdf'):
      print(f"Carregando: {file_path}")
      loader = PyPDFLoader(file_path)
      documents.extend(loader.load_and_split(text_splitter=text_splitter))
  
  print("Todos os PDFs foram carregados.")
  return documents
