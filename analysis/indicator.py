import pandas as pd
from rapidfuzz import process, fuzz
import re

# Instale rapidfuzz se não tiver: pip install rapidfuzz

STOPWORDS = {
    "de", "da", "do", "das", "dos", "a", "e", "em", "na", "no", "para", "por", "com", "sem", "o", "os", "as", "um", "uma"
}

def normalize_term(term):
    term = term.lower().strip()
    term = re.sub(r'[^\w\s-]', '', term)  # remove pontuação exceto hífen
    term = re.sub(r'\s+', ' ', term)
    palavras = [w for w in term.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(palavras)

def detect_id_column(df):
    for col in df.columns:
        if df[col].is_unique:
            return col
    return None

def detect_group_columns(df, id_col):
    group_cols = []
    for col in df.columns:
        if col == id_col:
            continue
        nunique = df[col].nunique(dropna=True)
        if nunique < len(df) * 0.8:
            group_cols.append(col)
    return group_cols

def fuzzy_group_terms(terms, threshold=90):
    """Agrupa termos similares com fuzzy matching."""
    clusters = []
    used = set()
    for term in terms:
        if term in used:
            continue
        cluster = [term]
        used.add(term)
        for candidate in terms:
            if candidate in used:
                continue
            if fuzz.ratio(term, candidate) >= threshold:
                cluster.append(candidate)
                used.add(candidate)
        clusters.append(cluster)
    return clusters

def generate_indicators(df, custom_stopwords=None):
    id_col = detect_id_column(df)
    if not id_col:
        df['_id'] = [str(i) for i in range(1, len(df)+1)]
        id_col = '_id'

    indicators = {
        "id_coluna": id_col,
        "total_linhas": len(df),
        "total_colunas": len(df.columns),
        "agrupamentos": []
    }
    group_cols = detect_group_columns(df, id_col)
    for col in group_cols:
        # Normaliza e mapeia termos originais para normalizados
        valores = df[[col, id_col]].dropna()
        mapping = {}
        for idx, row in valores.iterrows():
            original = str(row[col]).strip()
            norm = normalize_term(original)
            if norm not in mapping:
                mapping[norm] = {"originais": set(), "ids": set()}
            mapping[norm]["originais"].add(original)
            mapping[norm]["ids"].add(str(row[id_col]))

        # Fuzzy group nos termos normalizados
        termos_normais = list(mapping.keys())
        clusters = fuzzy_group_terms(termos_normais, threshold=90)  # pode ajustar para 85 ou 80 se quiser mais agressivo

        tabela = []
        for cluster in clusters:
            variantes = set()
            ids = set()
            for norm in cluster:
                variantes.update(mapping[norm]["originais"])
                ids.update(mapping[norm]["ids"])
            tabela.append({
                "termo_base": max(cluster, key=len).upper(),
                "variantes": "; ".join(sorted(variantes)),
                "frequencia": len(ids),
                "ids": ",".join(sorted(ids))
            })
        tabela = pd.DataFrame(tabela).sort_values(by="frequencia", ascending=False)
        indicators["agrupamentos"].append({
            "coluna": col,
            "tabela": tabela
        })
    return indicators
