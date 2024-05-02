from flask import Flask, request, jsonify, abort
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import nltk

app = Flask(__name__)
CORS(app, origins='*')

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text).lower()
    stop_words = set(stopwords.words('portuguese'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)

def extract_details(text):
    patterns = {
        'cause_by': r'Causa raiz: (.*)',
        'testes_realizados_by': r'Testes Realizados: (.*)',
        'solution_by': r'Solução aplicada: (.*)',
        'validated_by': r'Validado por: (.*)'
    }
    details = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            details[key] = match.group(1).strip()
    return details

@app.route('/process', methods=['POST'])
def process_notes():
    if not request.json or 'notes' not in request.json:
        abort(400, description="Bad Request: JSON body must contain 'notes' key with a list of strings.")

    notes = request.json['notes']
    processed_data = []

    for note in notes:
        details = extract_details(note)
        processed_data.append(details)

    return jsonify(processed_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)

