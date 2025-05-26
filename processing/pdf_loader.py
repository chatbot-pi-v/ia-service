import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdfs(data_path="./docs/pdf"):
  documents = []
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)

  for file_name in os.listdir(data_path):
    file_path = os.path.join(data_path, file_name)
    if file_name.endswith('.pdf'):
      loader = PyPDFLoader(file_path)
      documents.extend(loader.load_and_split(text_splitter=text_splitter))
  
  return documents
