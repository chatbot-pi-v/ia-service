# Instruções para rodar a IA

# 1. Intalar o Docker
  - `https://www.docker.com/products/docker-desktop/`

# 2. Baixar e executar o container do Milvus    
  ## Em sistemas Windows
    1. Configurar o WSL
      - `https://learn.microsoft.com/pt-br/windows/wsl/install`
    2. Rodar o comando no powershell como administador:
      - `Invoke-WebRequest https://raw.githubusercontent.com/milvus-io/milvus/refs/heads/master/scripts/standalone_embed.bat -OutFile standalone.bat`
      - `standalone.bat start`
       
  ## Em sistemas unix (MacOs ou Linux)
    1. Executar os comandos no terminal
      - `curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh`
      - `bash standalone_embed.sh start`
     
# 3. Instalar o Ollama:
  - `https://ollama.com/`
# 4. Abrir o CMD como administrador (Windows) ou Terminal (MacOs ou Linux) e rodar o comando:
  - `ollama pull llama3`

# 5. Rodar o script que instala as bibliotecas necessárias:
  - `python install_packages.py`

# 6. Após todas as bibliotecas instaladas, basta digitar e rodar o seguinte comando no terminal na pasta do projeto
  - `python app.py`
    Esse comando irá iniciar o servidor python na porta 5050

