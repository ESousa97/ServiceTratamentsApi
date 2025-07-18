# analysis/semantic.py
from analysis.stopwords import clean_text, remove_stopwords
import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Modelo carregado uma vez (pode ser movido para singleton)
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def embed_terms(terms):
    return model.encode(terms, convert_to_tensor=False, show_progress_bar=True)

def cluster_terms_by_embedding(terms, threshold=0.8):
    embeddings = embed_terms(terms)
    clusters = []
    used = set()
    for i, emb_i in enumerate(embeddings):
        if i in used:
            continue
        cluster = [terms[i]]
        used.add(i)
        for j in range(i+1, len(terms)):
            if j in used:
                continue
            sim = cosine_similarity([emb_i], [embeddings[j]])[0][0]
            if sim >= threshold:
                cluster.append(terms[j])
                used.add(j)
        clusters.append(cluster)
    return clusters

def get_terms_frequency(df, text_columns, custom_stopwords=None):
    term_freq = defaultdict(int)
    term_ids = defaultdict(set)
    for idx, row in df.iterrows():
        row_id = row['id']
        for col in text_columns:
            val = clean_text(row[col])
            words = remove_stopwords(val.split(), custom_stopwords)
            for w in words:
                if w.strip():
                    term_freq[w] += 1
                    term_ids[w].add(row_id)

    freq_df = pd.DataFrame(
        [(t, term_freq[t], len(term_ids[t])) for t in term_freq],
        columns=['termo', 'frequencia', 'ids_unicos']
    ).sort_values(by=['frequencia', 'termo'], ascending=[False, True]).reset_index(drop=True)

    # Agrupar semanticamente os termos
    termos = freq_df['termo'].tolist()
    clusters = cluster_terms_by_embedding(termos)

    return freq_df, term_ids, clusters
