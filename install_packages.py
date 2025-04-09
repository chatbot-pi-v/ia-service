import subprocess
import sys

# Lista das bibliotecas a serem instaladas
packages = [
    "langchain-huggingface",
    "langchain_community",
    "langchain_core",
    "langchain_milvus",
    "Flask",
    "pypdf",
    "torch",
    "torchvision",
    "clip-by-openai",
    "pymilvus",
    "pillow",
    "pymilvus",
    "python-dotenv"
]

def install_packages():
    for package in packages:
        try:
            print(f"Instalando {package}...\n")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} instalado com sucesso!\n")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao instalar {package}: {e}")

if __name__ == "__main__":
    install_packages()
    print("Instalação concluída!")
