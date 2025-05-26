from flask import Blueprint, request, jsonify
from services.rag_service import process_question 

api = Blueprint('api', __name__)

@api.route('/question', methods=['POST'])
def answer_question():
  data = request.get_json()
  question = data.get('question', '')

  if not question:
    return jsonify({"error": "Pergunta vazia"}), 400

  response = process_question(question)
  
  print(f'response = {response}')
  return jsonify({
    "answer": response["text"],
    "image_base64": response["image_base64"],
    "image_caption": response["image_caption"]
  })