from flask import Flask
from api.routes import api

app = Flask(__name__)
app.register_blueprint(api)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5050, debug=True)

##CRIAR COLEÇÃO DE IMAGENS
  ##Humanizar respostas
  ##Criar coleção de imagens