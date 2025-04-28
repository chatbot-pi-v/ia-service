from flask import Blueprint, request, jsonify
from services.rag_service import process_question 

api = Blueprint('api', __name__)

@api.route('/ask', methods=['POST'])
def answer_question():
    
  data = request.get_json()
  question = data.get('question', '')

  if not question:
    return jsonify({"error": "Pergunta vazia"}), 400

  response = process_question(question)

  return jsonify({"answer": response})
