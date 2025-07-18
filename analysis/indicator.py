import pandas as pd
from rapidfuzz import fuzz
from analysis.detector import detect_column_types
import unidecode
import re

def is_id_column(col, df):
    # Detecta se é uma coluna de id pelo nome ou se todos valores são únicos
    return str(col).lower() in ['id', 'codigo', 'identificacao', 'code'] or df[col].is_unique

def is_numerical(col, df):
    return pd.api.types.is_numeric_dtype(df[col])

def is_categorical(col, df):
    if is_numerical(col, df):
        return df[col].nunique() < min(30, len(df) // 5)
    return False

def is_date_candidate(col):
    # Considera candidato a data só se o nome indicar
    keywords = ['data', 'date', 'day', 'dia']
    return any(k in unidecode.unidecode(str(col)).lower() for k in keywords)

def normalize_generic(val):
    s = str(val).lower().strip()
    s = unidecode.unidecode(s)
    s = re.sub(r'[^\w\s-]', '', s)
    return s

def safe_to_datetime(series):
    try:
        sample = series.dropna().astype(str).head(50)
        if sample.str.match(r'\d{4}-\d{2}-\d{2}').all():
            return pd.to_datetime(series, format='%Y-%m-%d', errors='coerce')
        else:
            return pd.to_datetime(series, errors='coerce')
    except:
        return pd.to_datetime(series, errors='coerce')

def fuzzy_cluster_terms(terms, threshold=90, max_terms=500):
    if len(terms) > max_terms:
        return [[term] for term in terms]
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

def generate_indicators(df):
    col_types = detect_column_types(df)
    id_col = None
    for col in df.columns:
        if is_id_column(col, df):
            id_col = col
            break
    if not id_col:
        df['_id'] = [str(i) for i in range(1, len(df)+1)]
        id_col = '_id'

    indicators = {
        "id_coluna": id_col,
        "total_linhas": len(df),
        "total_colunas": len(df.columns),
        "agrupamentos": []
    }
    skip_cols = set([id_col])

    for col in df.columns:
        if col in skip_cols:
            continue

        label_tipo = col_types.get(col) or "desconhecido"

        if is_date_candidate(col):
            conv = safe_to_datetime(df[col])
            indicators["agrupamentos"].append({
                "coluna": col,
                "tipo": label_tipo,
                "estatisticas": {
                    "min": str(conv.min()),
                    "max": str(conv.max())
                }
            })
            continue

        if is_numerical(col, df) and not is_categorical(col, df):
            indicators["agrupamentos"].append({
                "coluna": col,
                "tipo": label_tipo,
                "estatisticas": {
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "media": float(df[col].mean())
                }
            })
            continue

        valores = df[[col, id_col]].dropna()
        value_counts = valores[col].value_counts()
        if len(value_counts) > 200:
            top_vals = value_counts.head(100).index
            valores = valores[valores[col].isin(top_vals)]

        mapping = {}
        for idx, row in valores.iterrows():
            original = str(row[col]).strip()
            norm = normalize_generic(original)
            if norm not in mapping:
                mapping[norm] = {"originais": set(), "ids": set()}
            mapping[norm]["originais"].add(original)
            mapping[norm]["ids"].add(str(row[id_col]))

        termos_normais = list(mapping.keys())
        clusters = fuzzy_cluster_terms(termos_normais, threshold=88, max_terms=500)

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
        tabela_df = pd.DataFrame(tabela).sort_values(by="frequencia", ascending=False)
        indicators["agrupamentos"].append({
            "coluna": col,
            "tipo": label_tipo,
            "tabela": tabela_df
        })

    return indicators
