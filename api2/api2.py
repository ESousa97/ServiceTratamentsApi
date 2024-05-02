from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import numpy as np
import unicodedata
from collections import defaultdict
import json
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')

def normalize_text(text):
    """
    Função para normalizar o texto removendo acentos e convertendo para minúsculas.
    """
    normalized_text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return normalized_text.lower()

def convert_to_serializable(obj):
    """
    Função para converter numpy.int64 em int para serialização JSON.
    """
    if isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {convert_to_serializable(key): convert_to_serializable(value) for key, value in obj.items()}
    return obj

@app.route('/cluster-tickets', methods=['POST'])
def cluster_tickets():
    tickets = request.get_json()
    texts = [
        " ".join([
            t.get('cause_by', ''),
            t.get('testes_realizados_by', ''),
            t.get('solution_by', ''),
            t.get('validated_by', '')
        ]).strip()
        for t in tickets
        if any(t.get(field) for field in ['cause_by', 'testes_realizados_by', 'solution_by', 'validated_by'])
    ]

    if not texts:
        return jsonify({})

    # Normalização dos textos
    normalized_texts = [normalize_text(text) for text in texts]

    # Vetorização dos textos utilizando TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(normalized_texts)

    # Cálculo da similaridade de cosseno entre os textos
    cosine_sim = cosine_similarity(X)

    # Normalização da matriz de similaridade entre 0 e 1
    normalized_cosine_sim = (cosine_sim - cosine_sim.min()) / (cosine_sim.max() - cosine_sim.min())

    # Agrupamento com DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
    labels = clustering.fit_predict(1 - normalized_cosine_sim)

    # Agrupando os tickets com base nos rótulos
    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(tickets[idx])

    # Normalizando os indicadores e agrupando os valores semelhantes
    separated_indicators = defaultdict(dict)
    for cluster_id, tickets in clusters.items():
        indicators = defaultdict(list)
        for ticket in tickets:
            for key, value in ticket.items():
                if isinstance(value, str):
                    normalized_value = normalize_text(value)
                    indicators[key].append(normalized_value)

        # Agrupando os valores semelhantes
        for key, values in indicators.items():
            unique_values = set(values)
            for value in unique_values:
                count = values.count(value)
                if count > 1:
                    separated_indicators[str(cluster_id)].setdefault(key, []).append(f"{value} ({count})")
                else:
                    separated_indicators[str(cluster_id)].setdefault(key, []).append(value)

    # Verificando se há indicadores semelhantes para combinar
    for cluster_id, indicators in separated_indicators.items():
        for key, values in indicators.items():
            unique_values = set(values)
            for value in unique_values:
                count = values.count(value)
                if count > 1:
                    new_value = f"{value} {count}"
                    separated_indicators[cluster_id][key] = [new_value if v == value else v for v in separated_indicators[cluster_id][key]]

    # Convertendo o dicionário para tornar as chaves serializáveis
    serializable_clusters = convert_to_serializable(separated_indicators)

    # Retornando a resposta JSON
    return jsonify(serializable_clusters)

if __name__ == '__main__':
    app.run(debug=True, port=5001)

